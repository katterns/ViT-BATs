import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch.nn as nn

import config as cfg
from bat.data import load_split, make_loaders
from bat.lightning_utils import ClassifierModule, SaveBest, final_eval, load_weights, log_dir, make_trainer

BEST_CKPT = cfg.CHECKPOINT_DIR / "cnn_bat_a0_best.pt"
CONFUSION_PNG = cfg.CHECKPOINT_DIR / "cnn_bat_a0_confusion_matrix.png"
LR, WD, LS = 1e-3, 1e-4, 0.1
MAX_EPOCHS, PATIENCE = 40, 10
PLATEAU_PATIENCE, LR_FACTOR, LR_MIN = 5, 0.5, 1e-7


def _gn(ch):
    g = 8
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


def _block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        _gn(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BatCNNA0(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            _block(1, 32), _block(32, 64), _block(64, 128),
            _block(128, 256), _block(256, 256), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", nargs="?", const=str(BEST_CKPT), default=None)
    args = p.parse_args()
    resume = Path(args.resume) if args.resume else None

    pl.seed_everything(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, label2id, id2label, species = load_split(cfg.FT_METADATA_PATH, cfg.FT_DATA_DIR)
    train_loader, val_loader = make_loaders(train_df, val_df, mode="supervised", balanced=True)

    model = BatCNNA0(len(species))
    module = ClassifierModule(
        model, [{"params": model.parameters(), "lr": LR}],
        weight_decay=WD, plateau_patience=PLATEAU_PATIENCE, lr_factor=LR_FACTOR, lr_min=LR_MIN,
        label_smoothing=LS,
    )
    if resume and resume.is_file():
        load_weights(model, resume)

    trainer = make_trainer(
        "cnn_baseline", max_epochs=MAX_EPOCHS, monitor="macro_f1", mode="max", patience=PATIENCE,
        extra_callbacks=[SaveBest(BEST_CKPT, "cnn_bat_a0", label2id, id2label)],
    )
    print(f"logs: {log_dir('cnn_baseline')}")
    trainer.fit(module, train_loader, val_loader)

    if BEST_CKPT.is_file():
        load_weights(model, BEST_CKPT)
    final_eval(module, val_loader, species, CONFUSION_PNG)


if __name__ == "__main__":
    main()
