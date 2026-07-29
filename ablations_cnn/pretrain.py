import argparse
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import Callback

import config as cfg
from ablations_cnn.cnn_ssl import CONTRASTIVE_TEMP, NOISE_PATCH_PERCENTILE, BatCNNSSL
from ablations_cnn.presets import ABLATION_ROOT, parse_preset, ssl_ckpt_path, ssl_monitor, ssl_run_name, use_recording_loader
from bat.data import load_paper_trainval, make_loaders
from bat.data.audio import SPEC_H, SPEC_W, _apply_spec_aug, mix_specs
from bat.lightning_utils import EpochLRScheduler, load_weights, log_dir, make_trainer, resolve_resume

WEIGHT_DECAY = 0.05
ADAMW_BETAS = (0.9, 0.95)
MIN_LR = 1e-6


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
    def __init__(self, ckpt_path, preset, monitor, initial_best=float("inf")):
        self.ckpt_path = ckpt_path
        self.preset = preset
        self.monitor = monitor
        self.best = float(initial_best)

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        self.best = float(state_dict.get("best", self.best))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val = trainer.callback_metrics.get(self.monitor)
        if val is None or float(val) >= self.best:
            return
        self.best = float(val)
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": pl_module.model.state_dict(),
                "encoder_state": pl_module.model.encoder_state_dict(),
                "model_name": "cnn_ssl",
                "preset": self.preset.id,
                "monitor": self.monitor,
                "val_metric": self.best,
                "epoch": trainer.current_epoch,
            },
            self.ckpt_path,
        )


class SSLModule(pl.LightningModule):
    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks
        self.con_only = tasks.con and not (tasks.mae or tasks.sep or tasks.jig)
        self.model = BatCNNSSL(tasks, SPEC_H, SPEC_W)

    def _sample_gain_ratio(self, batch_size, device, dtype):
        lo, hi = cfg.SEP_GAIN_RATIO_MIN, cfg.SEP_GAIN_RATIO_MAX
        return torch.empty(batch_size, device=device, dtype=dtype).uniform_(lo, hi)

    def _aug_batch(self, x):
        rng = np.random.default_rng()
        views = [_apply_spec_aug(x[i].detach().cpu().clone(), rng) for i in range(x.shape[0])]
        return torch.stack(views, dim=0).to(device=x.device, dtype=x.dtype)

    def _sep_pair(self, x):
        b = x.shape[0]
        if b < 2:
            return None
        perm = (torch.arange(b, device=x.device) + 1) % b
        s1, s2 = x, x[perm]
        gain = self._sample_gain_ratio(b, x.device, x.dtype)
        return mix_specs(s1, s2, gain), s1, s2

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        return batch, None

    @staticmethod
    def _due(batch_idx, every, offset=0):
        return every > 0 and (batch_idx % every) == (offset % every)

    def _run_task(self, task, batch_idx):
        if not self.tasks[task]:
            return False
        if sum(self.tasks[t] for t in ("mae", "con", "sep", "jig")) == 1:
            return True
        if task == "mae":
            return True
        if task == "con":
            return self._due(batch_idx, cfg.CONTRASTIVE_EVERY, cfg.CONTRASTIVE_EVERY_OFFSET)
        if task == "sep":
            return self._due(batch_idx, cfg.SEP_EVERY, cfg.SEP_EVERY_OFFSET)
        return self._due(batch_idx, cfg.JIGSAW_EVERY, cfg.JIGSAW_EVERY_OFFSET)

    def _contrastive_pair(self, x, peer):
        if self.con_only:
            return self._aug_batch(x), self._aug_batch(x)
        real = (x - peer).abs().flatten(1).max(dim=1).values > 1e-5
        if int(real.sum()) >= 2:
            return x[real], peer[real]
        return None, None

    def _step_losses(self, x, peer, batch_idx, utterance_weight):
        loss = x.new_tensor(0.0)
        recon = utt = con = sep = jig = None

        if self._run_task("mae", batch_idx):
            mae, sem = self.model.mae_loss(
                x, cfg.MASK_RATIO, NOISE_PATCH_PERCENTILE, utterance_weight,
            )
            loss = loss + mae
            recon = sem["recon"]
            utt = sem.get("utterance")

        if self._run_task("con", batch_idx):
            x1, x2 = self._contrastive_pair(x, peer)
            if x1 is not None:
                c = self.model.contrastive_loss(x1, x2, CONTRASTIVE_TEMP)
                loss = loss + cfg.SEM_CONTRASTIVE_WEIGHT * c
                con = c.detach()

        if self._run_task("sep", batch_idx):
            pair = self._sep_pair(x)
            if pair is not None:
                mix, s1, s2 = pair
                s = self.model.separation_loss(mix, s1, s2)
                loss = loss + cfg.SEM_SEP_WEIGHT * s
                sep = s.detach()

        if self._run_task("jig", batch_idx):
            j = self.model.jigsaw_loss(x, n_parts=cfg.JIGSAW_PARTS)
            loss = loss + cfg.SEM_JIGSAW_WEIGHT * j
            jig = j.detach()

        return loss, recon, utt, con, sep, jig

    def _log_parts(self, prefix, recon, utt, con, sep, jig, prog_bar=False):
        if recon is not None:
            self.log(f"{prefix}_recon", recon, prog_bar=(prog_bar and self.tasks.mae), on_epoch=True)
        if utt is not None:
            self.log(f"{prefix}_utterance", utt, on_epoch=True)
        if con is not None:
            self.log(f"{prefix}_con", con, prog_bar=(prog_bar and self.tasks.con), on_epoch=True)
        if sep is not None:
            self.log(f"{prefix}_sep", sep, prog_bar=(prog_bar and self.tasks.sep), on_epoch=True)
        if jig is not None:
            self.log(f"{prefix}_jig", jig, prog_bar=(prog_bar and self.tasks.jig), on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, peer = self._unpack_batch(batch)
        loss, recon, utt, con, sep, jig = self._step_losses(
            x, peer, batch_idx, cfg.SEM_UTTERANCE_WEIGHT,
        )
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self._log_parts("train", recon, utt, con, sep, jig)
        return loss

    def validation_step(self, batch, batch_idx):
        x, peer = self._unpack_batch(batch)
        # как ViT: val MAE без utterance; остальные задачи — тот же протокол, что train
        loss, recon, utt, con, sep, jig = self._step_losses(
            x, peer, batch_idx, utterance_weight=0.0,
        )
        self.log("val_loss", loss, on_epoch=True)
        self._log_parts("val", recon, utt, con, sep, jig, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=cfg.BASE_LR, weight_decay=WEIGHT_DECAY, betas=ADAMW_BETAS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", required=True)
    p.add_argument(
        "--resume", nargs="?", const="__auto__", default=None,
        help="без пути: last.ckpt; .ckpt — полный resume; .pt — только веса",
    )
    args = p.parse_args()

    tasks = parse_preset(args.preset)
    best_ckpt = ssl_ckpt_path(tasks)
    run_name = ssl_run_name(tasks)
    monitor = ssl_monitor(tasks)

    pl.seed_everything(cfg.RANDOM_SEED)
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    train_df, val_df, _, _, _ = load_paper_trainval()
    cache = cfg.NABAT_PAPER_SPEC_CACHE
    loader_mode = "recording" if use_recording_loader(tasks) else "ssl"
    train_loader, val_loader = make_loaders(
        train_df, val_df, mode=loader_mode, val_mode=loader_mode, cache_dir=cache,
    )

    module = SSLModule(tasks)
    weights_ckpt, pl_ckpt, completed_epochs = resolve_resume(args.resume, run_name, best_ckpt)
    initial_best = float("inf")
    if weights_ckpt is not None:
        meta = load_weights(module.model, weights_ckpt, strict=False)
        initial_best = float(meta.get("val_metric", meta.get("val_loss", float("inf"))))
        print(
            f"resume weights: {weights_ckpt} (epoch={meta.get('epoch', '?')}, "
            f"{monitor}={initial_best:.4f}, next epoch={completed_epochs})",
            flush=True,
        )
    elif pl_ckpt is not None:
        print(f"resume trainer: {pl_ckpt}", flush=True)

    trainer = make_trainer(
        run_name,
        max_epochs=cfg.MAX_EPOCHS,
        monitor=monitor,
        mode="min",
        patience=cfg.PATIENCE,
        extra_callbacks=[
            SaveBestSSL(best_ckpt, tasks, monitor, initial_best=initial_best),
            EpochLRScheduler(set_epoch_lr),
        ],
        continuing_run=args.resume is not None,
        restore_completed_epochs=completed_epochs if pl_ckpt is None else 0,
    )
    print(f"preset: {tasks.id}", flush=True)
    print(f"monitor: {monitor}", flush=True)
    print(f"loader: {loader_mode}", flush=True)
    print(f"logs: {log_dir(run_name)}", flush=True)
    print(f"best ckpt: {best_ckpt}", flush=True)
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(pl_ckpt) if pl_ckpt else None)


if __name__ == "__main__":
    main()
