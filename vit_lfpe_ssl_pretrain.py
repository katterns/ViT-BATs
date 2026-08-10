import argparse
import math
import warnings

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import Callback

import config as cfg
from bat.data import load_paper_trainval, make_loaders
from bat.data.audio import CLIP_SEC, SPEC_CHANNELS, SPEC_H, SPEC_W, mix_specs
from bat.lightning_utils import EpochLRScheduler, load_weights, log_dir, make_trainer, resolve_resume
from vit_bat import BatViTPatchMAE, EMBED_DIM, PATCH_SIZE

NOISE_PATCH_PERCENTILE = 25.0
CONTRASTIVE_TEMP = 0.07
WEIGHT_DECAY = 0.05
ADAMW_BETAS = (0.9, 0.95)
MIN_LR = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", nargs="?", const="__auto__", default=None,
                   help="без пути: last.ckpt; .ckpt — полный resume; .pt — только веса")
    return p.parse_args()


def set_epoch_lr(epoch, optimizer):
    if epoch <= cfg.WARMUP_EPOCHS:
        lr = cfg.BASE_LR * epoch / max(cfg.WARMUP_EPOCHS, 1)
    else:
        t = (epoch - cfg.WARMUP_EPOCHS) / max(cfg.MAX_EPOCHS - cfg.WARMUP_EPOCHS, 1)
        lr = MIN_LR + 0.5 * (cfg.BASE_LR - MIN_LR) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


class SaveBestSSL(Callback):
    def __init__(self, initial_best=float("inf"), data_tag="nabat_paper_31"):
        self.best = float(initial_best)
        self.data_tag = data_tag

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        self.best = float(state_dict.get("best", self.best))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val = trainer.callback_metrics.get("val_loss")
        if val is None or float(val) >= self.best:
            return
        self.best = float(val)
        cfg.BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": pl_module.model.state_dict(),
            "encoder_state": pl_module.model.encoder_state_dict(),
            "model_name": "vit_lfpe_mae",
            "config": {
                "clip_sec": CLIP_SEC,
                "spec_channels": SPEC_CHANNELS,
                "spec_hw": (SPEC_H, SPEC_W),
                "patch_size": PATCH_SIZE,
                "embed_dim": EMBED_DIM,
                "sem_contrastive_weight": cfg.SEM_CONTRASTIVE_WEIGHT,
                "sem_sep_weight": cfg.SEM_SEP_WEIGHT,
                "sem_jigsaw_weight": cfg.SEM_JIGSAW_WEIGHT,
                "jigsaw_parts": cfg.JIGSAW_PARTS,
                "preprocess": "nabat_v2",
                "data": self.data_tag,
            },
            "val_recon_loss": self.best,
            "epoch": trainer.current_epoch,
        }, cfg.BEST_CKPT)


class SSLModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BatViTPatchMAE(SPEC_H, SPEC_W)

    def _sample_gain_ratio(self, batch_size, device, dtype):
        lo, hi = cfg.SEP_GAIN_RATIO_MIN, cfg.SEP_GAIN_RATIO_MAX
        return torch.empty(batch_size, device=device, dtype=dtype).uniform_(lo, hi)

    def _sep_pair(self, x):
        b = x.shape[0]
        if b < 2:
            return None
        perm = torch.randperm(b, device=x.device)
        for _ in range(16):
            if not (perm == torch.arange(b, device=x.device)).any():
                break
            perm = torch.randperm(b, device=x.device)
        else:
            perm = (torch.arange(b, device=x.device) + 1) % b
        s1, s2 = x, x[perm]
        gain = self._sample_gain_ratio(b, x.device, x.dtype)
        mix = mix_specs(s1, s2, gain)
        return mix, s1, s2

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                return batch[0], batch[1], None
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
        return batch, None, None

    @staticmethod
    def _due(batch_idx, every, offset=0):
        return every > 0 and (batch_idx % every) == (offset % every)

    def _loss(self, x, peer, batch_idx):
        loss, _, _, _, sem = self.model(
            x,
            mask_ratio=cfg.MASK_RATIO,
            noise_percentile=NOISE_PATCH_PERCENTILE,
            utterance_weight=cfg.SEM_UTTERANCE_WEIGHT,
        )
        con = 0.0
        if (
            cfg.SEM_CONTRASTIVE_WEIGHT > 0
            and self._due(batch_idx, cfg.CONTRASTIVE_EVERY, cfg.CONTRASTIVE_EVERY_OFFSET)
            and peer is not None
        ):
            real = (x - peer).abs().flatten(1).max(dim=1).values > 1e-5
            if int(real.sum()) >= 2:
                c = self.model.contrastive_loss(x[real], peer[real], CONTRASTIVE_TEMP)
                loss = loss + cfg.SEM_CONTRASTIVE_WEIGHT * c
                con = float(c.detach())

        sep = 0.0
        if cfg.SEM_SEP_WEIGHT > 0 and self._due(batch_idx, cfg.SEP_EVERY, cfg.SEP_EVERY_OFFSET):
            pair = self._sep_pair(x)
            if pair is not None:
                mix, s1, s2 = pair
                s = self.model.separation_loss(mix, s1, s2)
                loss = loss + cfg.SEM_SEP_WEIGHT * s
                sep = float(s.detach())

        jig = 0.0
        if cfg.SEM_JIGSAW_WEIGHT > 0 and self._due(batch_idx, cfg.JIGSAW_EVERY, cfg.JIGSAW_EVERY_OFFSET):
            j = self.model.jigsaw_loss(x, n_parts=cfg.JIGSAW_PARTS)
            loss = loss + cfg.SEM_JIGSAW_WEIGHT * j
            jig = float(j.detach())

        return loss, sem, con, sep, jig

    def training_step(self, batch, batch_idx):
        x, peer, _ = self._unpack_batch(batch)
        loss, sem, con, sep, jig = self._loss(x, peer, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_recon", sem.get("recon", 0.0), on_epoch=True)
        self.log("train_utterance", sem.get("utterance", 0.0), on_epoch=True)
        if con:
            self.log("train_contrastive", con, on_epoch=True)
        if sep:
            self.log("train_sep", sep, on_epoch=True)
        if jig:
            self.log("train_jigsaw", jig, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = self._unpack_batch(batch)
        loss, _, _, _, _ = self.model(
            x, mask_ratio=cfg.MASK_RATIO, noise_percentile=NOISE_PATCH_PERCENTILE, utterance_weight=0.0
        )
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=cfg.BASE_LR, weight_decay=WEIGHT_DECAY, betas=ADAMW_BETAS)


def main():
    args = parse_args()

    pl.seed_everything(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cache = cfg.NABAT_PAPER_SPEC_CACHE
    train_df, val_df, _, _, _ = load_paper_trainval()
    cache_dir = cache
    print(f"SSL data: nabat_paper_31 pulses ({len(train_df)} train + {len(val_df)} val)", flush=True)

    if cfg.USE_SPEC_CACHE:
        from bat.data.spec_cache import cache_stats
        hits, total = cache_stats(train_df, cache_dir=cache_dir)
        print(f"spec cache: {hits}/{total} in {cache_dir}", flush=True)

    print(f"SSL: {len(train_df)} train + {len(val_df)} val pulses", flush=True)
    mode = "recording" if cfg.SEM_CONTRASTIVE_WEIGHT > 0 else "ssl"
    train_loader, val_loader = make_loaders(
        train_df, val_df, mode=mode, val_mode="ssl", cache_dir=cache_dir,
    )

    module = SSLModule()
    weights_ckpt, pl_ckpt, completed_epochs = resolve_resume(args.resume, "ssl_pretrain", cfg.BEST_CKPT)
    initial_best = float("inf")
    if weights_ckpt is not None:
        meta = load_weights(module.model, weights_ckpt, strict=False)
        initial_best = float(meta.get("val_recon_loss", float("inf")))
        print(
            f"resume weights: {weights_ckpt} (epoch={meta.get('epoch', '?')}, next epoch={completed_epochs})",
            flush=True,
        )
    elif pl_ckpt is not None:
        print(f"resume trainer: {pl_ckpt}", flush=True)

    data_tag = "nabat_paper_31"
    trainer = make_trainer(
        "ssl_pretrain",
        max_epochs=cfg.MAX_EPOCHS,
        monitor="val_loss",
        mode="min",
        patience=cfg.PATIENCE,
        extra_callbacks=[SaveBestSSL(initial_best, data_tag=data_tag), EpochLRScheduler(set_epoch_lr)],
        continuing_run=args.resume is not None,
        restore_completed_epochs=completed_epochs if pl_ckpt is None else 0,
    )
    print(f"logs: {log_dir('ssl_pretrain')}")
    print(
        f"SSL losses: MAE+utt×{cfg.SEM_UTTERANCE_WEIGHT}"
        f" + same-rec con×{cfg.SEM_CONTRASTIVE_WEIGHT}/every {cfg.CONTRASTIVE_EVERY}+{cfg.CONTRASTIVE_EVERY_OFFSET}"
        f" + sep×{cfg.SEM_SEP_WEIGHT}/every {cfg.SEP_EVERY}+{cfg.SEP_EVERY_OFFSET}"
        f" + jigsaw×{cfg.SEM_JIGSAW_WEIGHT}/every {cfg.JIGSAW_EVERY}+{cfg.JIGSAW_EVERY_OFFSET} (parts={cfg.JIGSAW_PARTS})",
        flush=True,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(pl_ckpt) if pl_ckpt else None)


if __name__ == "__main__":
    main()
