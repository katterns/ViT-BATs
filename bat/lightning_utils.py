from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config as cfg


def load_weights(model, path, key="model_state"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get(key) or ckpt.get("classifier_state") or ckpt.get("model_state")
    if state is None:
        raise KeyError(f"В чекпоинте {path} нет {key}")
    model.load_state_dict(state, strict=True)
    return ckpt


def log_dir(run_name: str) -> Path:
    return cfg.CHECKPOINT_DIR / "lightning_logs" / run_name


def last_ckpt_path(run_name: str) -> Path:
    return log_dir(run_name) / "pl_ckpt" / "last.ckpt"


def find_trainer_ckpt(run_name: str) -> Path | None:
    last = last_ckpt_path(run_name)
    return last if last.is_file() else None


def _completed_epochs_from_pt(path: Path) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    if epoch is None:
        return 0
    return int(epoch) + 1


def resolve_resume(resume, run_name, best_ckpt):
    """__auto__ / .ckpt -> полный resume; .pt -> веса + эпоха из meta."""
    if resume is None:
        return None, None, 0

    path = Path(resume)
    if str(resume) == "__auto__":
        pl_ckpt = find_trainer_ckpt(run_name)
        if pl_ckpt is not None:
            return None, pl_ckpt, 0
        best = Path(best_ckpt)
        if best.is_file():
            completed = _completed_epochs_from_pt(best)
            print(
                f"WARNING: {last_ckpt_path(run_name)} не найден — resume с best.pt (next epoch={completed})",
                flush=True,
            )
            return best, None, completed
        return None, None, 0

    if path.suffix == ".ckpt":
        if not path.is_file():
            raise FileNotFoundError(f"Lightning checkpoint не найден: {path}")
        return None, path, 0

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint не найден: {path}")
    return path, None, _completed_epochs_from_pt(path)


def _latest_log_version(root: Path) -> int:
    versions = []
    for sub in ("csv", "tb"):
        base = root / sub
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.name.startswith("version_"):
                try:
                    versions.append(int(p.name.split("_", 1)[1]))
                except ValueError:
                    pass
    return max(versions) if versions else 0


def make_trainer(
    run_name,
    *,
    max_epochs,
    monitor,
    mode,
    patience=None,
    extra_callbacks=None,
    epoch_log=None,
    continuing_run=False,
    restore_completed_epochs=0,
):
    root = log_dir(run_name)
    root.mkdir(parents=True, exist_ok=True)
    log_version = _latest_log_version(root) if continuing_run else None
    callbacks = list(extra_callbacks or [])
    if restore_completed_epochs > 0:
        callbacks.append(RestoreEpoch(restore_completed_epochs))
    callbacks.append(EpochSummary(epoch_log or root / "epoch_log.txt"))
    callbacks.append(SaveLast(run_name))
    if patience:
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, mode=mode))
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=[
            CSVLogger(save_dir=root, name="csv", version=log_version),
            TensorBoardLogger(save_dir=root, name="tb", version=log_version),
        ],
        callbacks=callbacks,
        gradient_clip_val=1.0,
        default_root_dir=root,
        enable_progress_bar=True,
    )


class EpochLRScheduler(Callback):
    def __init__(self, set_lr_fn):
        self.set_lr_fn = set_lr_fn

    def on_train_epoch_start(self, trainer, pl_module):
        lr = self.set_lr_fn(trainer.current_epoch + 1, pl_module.optimizers())
        pl_module.log("lr", lr, on_epoch=True, prog_bar=True)


class RestoreEpoch(Callback):
    def __init__(self, completed_epochs: int = 0):
        self.completed_epochs = int(completed_epochs)

    def on_fit_start(self, trainer, pl_module):
        if self.completed_epochs <= 0:
            return
        trainer.fit_loop.epoch_progress.current.completed = self.completed_epochs
        trainer.fit_loop.epoch_progress.current.processed = self.completed_epochs
        print(f"resume epoch: continue from {self.completed_epochs}", flush=True)


class SaveLast(Callback):
    def __init__(self, run_name: str):
        self.path = last_ckpt_path(run_name)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.path)


class EpochSummary(Callback):
    def __init__(self, log_path=None):
        self.log_path = Path(log_path) if log_path else None

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        line = (
            f"epoch={epoch:>2d}  "
            f"train_loss={float(m.get('train_loss', float('nan'))):.4f}  "
            f"val_loss={float(m.get('val_loss', float('nan'))):.4f}  "
            f"macro_f1={float(m.get('macro_f1', float('nan'))):.4f}  "
            f"acc={float(m.get('acc', float('nan'))):.4f}"
        )
        print(line, flush=True)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class SaveBest(Callback):
    def __init__(self, ckpt_path, model_name, label2id, id2label, extra=None, initial_best_f1=-1.0):
        self.ckpt_path = Path(ckpt_path)
        self.model_name = model_name
        self.label2id = label2id
        self.id2label = id2label
        self.extra = extra or {}
        self.best_f1 = float(initial_best_f1)

    def state_dict(self):
        return {"best_f1": self.best_f1}

    def load_state_dict(self, state_dict):
        self.best_f1 = float(state_dict.get("best_f1", self.best_f1))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        f1 = trainer.callback_metrics.get("macro_f1")
        if f1 is None or not float(f1) == float(f1):  # None or NaN
            return
        if float(f1) <= self.best_f1:
            return
        self.best_f1 = float(f1)
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": pl_module.model.state_dict(),
            "model_name": self.model_name,
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "val_macro_f1": self.best_f1,
            "epoch": trainer.current_epoch,
            "config": self.extra,
        }, self.ckpt_path)


class ClassifierModule(pl.LightningModule):
    def __init__(
        self,
        model,
        param_groups,
        *,
        weight_decay,
        plateau_patience,
        lr_factor,
        lr_min,
        label_smoothing=0.1,
        mixup_alpha=0.0,
    ):
        super().__init__()
        self.model = model
        self.param_groups = param_groups
        self.weight_decay = weight_decay
        self.plateau_patience = plateau_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mixup_alpha = mixup_alpha

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mixup_alpha > 0:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            perm = torch.randperm(x.shape[0], device=x.device)
            logits = self.model(lam * x + (1.0 - lam) * x[perm])
            loss = lam * self.loss_fn(logits, y) + (1.0 - lam) * self.loss_fn(logits, y[perm])
        else:
            loss = self.loss_fn(self.model(x), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self._y, self._pred = [], []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        self.log("val_loss", self.loss_fn(logits, y), prog_bar=True, on_epoch=True, batch_size=len(y))
        self._y.append(y.cpu())
        self._pred.append(logits.argmax(-1).cpu())

    def on_validation_epoch_end(self):
        y = torch.cat(self._y)
        pred = torch.cat(self._pred)
        self.log("macro_f1", f1_score(y, pred, average="macro", zero_division=0), prog_bar=True, on_epoch=True)
        self.log("acc", accuracy_score(y, pred), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        opt = optim.AdamW(self.param_groups, weight_decay=self.weight_decay)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=self.lr_factor, patience=self.plateau_patience, min_lr=self.lr_min
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "macro_f1", "interval": "epoch"}}


def save_confusion(y_true, y_pred, labels, path):
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(confusion_matrix(y_true, y_pred), cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def final_eval(module, val_loader, species, confusion_path):
    module.eval()
    y_true, y_pred = [], []
    device = next(module.parameters()).device
    for x, y in val_loader:
        x = x.to(device)
        y_pred.extend(module.model(x).argmax(-1).cpu().tolist())
        y_true.extend(y.tolist())
    print(classification_report(y_true, y_pred, target_names=species, zero_division=0))
    save_confusion(np.array(y_true), np.array(y_pred), species, confusion_path)
