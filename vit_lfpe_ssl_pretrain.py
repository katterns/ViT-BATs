import argparse
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import Callback

import config as cfg
from bat.data import load_paper_trainval, load_spec, make_loaders
from bat.data.audio import CLIP_SEC, SPEC_CHANNELS, SPEC_H, SPEC_W
from bat.lightning_utils import EpochLRScheduler, load_weights, log_dir, make_trainer, resolve_resume
from vit_bat import EMBED_DIM, PATCH_SIZE, BatViTPatchMAE, unpatchify_2d

NOISE_PATCH_PERCENTILE = 25.0
CONTRASTIVE_TEMP = 0.07
CONTRASTIVE_EVERY = 4
WEIGHT_DECAY = 0.05
ADAMW_BETAS = (0.9, 0.95)
MIN_LR = 1e-6
RECON_DEMO_PNG = cfg.CHECKPOINT_DIR / "vit_lfpe_ssl_recon_demo.png"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", nargs="?", const="__auto__", default=None,
                   help="без пути: last.ckpt; .ckpt — полный resume; .pt — только веса")
    p.add_argument(
        "--full",
        action="store_true",
        help="pretrain на full cleaned/ (нечестно vs subset CNN; только ablation)",
    )
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
    def __init__(self, initial_best=float("inf"), data_tag="subset_200"):
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

    def _loss(self, x, left, right, batch_idx):
        loss, _, _, _, sem = self.model(
            x,
            mask_ratio=cfg.MASK_RATIO,
            noise_percentile=NOISE_PATCH_PERCENTILE,
            utterance_weight=cfg.SEM_UTTERANCE_WEIGHT,
        )
        con = 0.0
        if (
            cfg.SEM_CONTRASTIVE_WEIGHT > 0
            and batch_idx % CONTRASTIVE_EVERY == 0
            and left is not None
        ):
            c = self.model.contrastive_loss(left, right, CONTRASTIVE_TEMP)
            loss = loss + cfg.SEM_CONTRASTIVE_WEIGHT * c
            con = float(c.detach())
        return loss, sem, con

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, left, right = batch
        else:
            x, left, right = batch, None, None
        loss, sem, con = self._loss(x, left, right, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_recon", sem.get("recon", 0.0), on_epoch=True)
        self.log("train_utterance", sem.get("utterance", 0.0), on_epoch=True)
        if con:
            self.log("train_contrastive", con, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        loss, _, _, _, _ = self.model(
            x, mask_ratio=cfg.MASK_RATIO, noise_percentile=NOISE_PATCH_PERCENTILE, utterance_weight=0.0
        )
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=cfg.BASE_LR, weight_decay=WEIGHT_DECAY, betas=ADAMW_BETAS)


@torch.no_grad()
def save_recon_demo(model, val_df, device, out_path):
    model.eval()
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    row = val_df.iloc[int(rng.integers(len(val_df)))]
    x = load_spec(row["path"], False, rng, pulse_center=row["pulse_center"]).unsqueeze(0).to(device)
    _, pred, mask, targets, _ = model(
        x, mask_ratio=cfg.MASK_RATIO, noise_percentile=NOISE_PATCH_PERCENTILE, utterance_weight=0.0
    )
    pf, pt = PATCH_SIZE
    gf, gt = model.grid_shape
    pred_p = pred.view(1, gf, gt, SPEC_CHANNELS, pf, pt)
    tgt_p = targets.view(1, gf, gt, SPEC_CHANNELS, pf, pt)
    m = mask.view(1, gf, gt).bool()
    recon_p = tgt_p.clone()
    recon_p[m] = pred_p[m]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].imshow(x[0].permute(1, 2, 0).cpu().clamp(0, 1), origin="lower", aspect="auto")
    axes[0].set_title("target")
    axes[0].axis("off")
    recon = unpatchify_2d(recon_p, PATCH_SIZE, SPEC_CHANNELS)[0]
    axes[1].imshow(recon.permute(1, 2, 0).cpu().clamp(0, 1), origin="lower", aspect="auto")
    axes[1].set_title("reconstruction")
    axes[1].axis("off")
    fig.suptitle(f"{row['species']}/{row['filename']}", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    pl.seed_everything(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if args.full:
        print("SSL --full: legacy cleaned/ через load_split (медленный пересчёт импульсов)", flush=True)
        from bat.data import load_split
        train_df, val_df, _, _, _ = load_split(cfg.METADATA_PATH, cfg.DATA_DIR)
        cache_dir = None
    else:
        train_df, val_df, _, _, _ = load_paper_trainval()
        cache_dir = cfg.NABAT_PAPER_SPEC_CACHE
        print(f"SSL data: nabat_paper_31 pulses ({len(train_df)} train + {len(val_df)} val)", flush=True)

    print(f"SSL: {len(train_df)} train + {len(val_df)} val pulses", flush=True)
    mode = "dual" if cfg.SEM_CONTRASTIVE_WEIGHT > 0 else "ssl"
    train_loader, val_loader = make_loaders(
        train_df, val_df, mode=mode, val_mode="ssl", cache_dir=cache_dir,
    )

    module = SSLModule()
    weights_ckpt, pl_ckpt, completed_epochs = resolve_resume(args.resume, "ssl_pretrain", cfg.BEST_CKPT)
    initial_best = float("inf")
    if weights_ckpt is not None:
        meta = load_weights(module.model, weights_ckpt)
        initial_best = float(meta.get("val_recon_loss", float("inf")))
        print(
            f"resume weights: {weights_ckpt} (epoch={meta.get('epoch', '?')}, next epoch={completed_epochs})",
            flush=True,
        )
    elif pl_ckpt is not None:
        print(f"resume trainer: {pl_ckpt}", flush=True)

    data_tag = "full_cleaned" if args.full else "nabat_paper_31"
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
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(pl_ckpt) if pl_ckpt else None)

    if cfg.BEST_CKPT.is_file():
        load_weights(module.model, cfg.BEST_CKPT)
    save_recon_demo(module.model, val_df, next(module.parameters()).device, RECON_DEMO_PNG)


if __name__ == "__main__":
    main()
