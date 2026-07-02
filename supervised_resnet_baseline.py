import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch.nn as nn
from torchvision.models import resnet18

import config as cfg
from bat.data.audio import SPEC_CHANNELS
from bat.data import load_split, make_loaders
from bat.lightning_utils import ClassifierModule, SaveBest, final_eval, load_weights, log_dir, make_trainer, resolve_resume

BEST_CKPT = cfg.CHECKPOINT_DIR / "resnet18_bat_best.pt"
CONFUSION_PNG = cfg.CHECKPOINT_DIR / "resnet18_bat_confusion_matrix.png"
LR, WD, LS = 1e-3, 1e-4, 0.1
MAX_EPOCHS, PATIENCE = 40, 10
PLATEAU_PATIENCE, LR_FACTOR, LR_MIN = 5, 0.5, 1e-7


class ResNet18Bat(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(SPEC_CHANNELS, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, n_classes))
        self.net = net

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", nargs="?", const="__auto__", default=None,
                   help="without path: last.ckpt; .ckpt — full resume; .pt — only weights")
    args = p.parse_args()

    pl.seed_everything(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, label2id, id2label, species = load_split(cfg.FT_METADATA_PATH, cfg.FT_DATA_DIR)
    train_loader, val_loader = make_loaders(train_df, val_df, mode="supervised", balanced=True)

    model = ResNet18Bat(len(species))
    module = ClassifierModule(
        model, [{"params": model.parameters(), "lr": LR}],
        weight_decay=WD, plateau_patience=PLATEAU_PATIENCE, lr_factor=LR_FACTOR, lr_min=LR_MIN,
        label_smoothing=LS,
    )

    weights_ckpt, pl_ckpt, completed_epochs = resolve_resume(args.resume, "resnet_baseline", BEST_CKPT)
    initial_best = -1.0
    if weights_ckpt is not None:
        meta = load_weights(model, weights_ckpt)
        initial_best = float(meta.get("val_macro_f1", -1.0))
        print(
            f"resume weights: {weights_ckpt} (saved epoch={meta.get('epoch', '?')}, "
            f"macro_f1={initial_best:.4f}, next epoch={completed_epochs})",
            flush=True,
        )
    elif pl_ckpt is not None:
        print(f"resume trainer: {pl_ckpt}", flush=True)

    trainer = make_trainer(
        "resnet_baseline", max_epochs=MAX_EPOCHS, monitor="macro_f1", mode="max", patience=PATIENCE,
        extra_callbacks=[SaveBest(BEST_CKPT, "resnet18_bat", label2id, id2label, initial_best_f1=initial_best)],
        continuing_run=args.resume is not None,
        restore_completed_epochs=completed_epochs if pl_ckpt is None else 0,
    )
    print(f"logs: {log_dir('resnet_baseline')}")
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(pl_ckpt) if pl_ckpt else None)

    if BEST_CKPT.is_file():
        load_weights(model, BEST_CKPT)
    final_eval(module, val_loader, species, CONFUSION_PNG)


if __name__ == "__main__":
    main()
