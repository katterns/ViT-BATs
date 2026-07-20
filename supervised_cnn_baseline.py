import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch.nn as nn

import config as cfg
from bat.data import load_paper_trainval, make_loaders
from bat.data.audio import SPEC_CHANNELS
from bat.lightning_utils import ClassifierModule, SaveBest, final_eval, last_ckpt_path, load_weights, log_dir, make_trainer, resolve_resume

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
            _block(SPEC_CHANNELS, 32), _block(32, 64), _block(64, 128),
            _block(128, 256), _block(256, 256), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", nargs="?", const="__auto__", default=None,
                   help="без пути: last.ckpt; .ckpt — полный resume; .pt — только веса")
    args = p.parse_args()

    pl.seed_everything(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, label2id, id2label, species = load_paper_trainval()
    cache = cfg.NABAT_PAPER_SPEC_CACHE
    train_loader, val_loader = make_loaders(
        train_df, val_df, mode="supervised", balanced=True, cache_dir=cache,
    )

    if cfg.USE_SPEC_CACHE:
        from bat.data.spec_cache import cache_stats
        hits, total = cache_stats(train_df, cache_dir=cache)
        if hits < total:
            print(
                f"spec cache: {hits}/{total} — дождитесь build --precompute-cache "
                f"или: uv run python scripts/precompute_specs.py --paper",
                flush=True,
            )

    model = BatCNNA0(len(species))
    module = ClassifierModule(
        model, [{"params": model.parameters(), "lr": LR}],
        weight_decay=WD, plateau_patience=PLATEAU_PATIENCE, lr_factor=LR_FACTOR, lr_min=LR_MIN,
        label_smoothing=LS,
    )

    weights_ckpt, pl_ckpt, completed_epochs = resolve_resume(args.resume, "cnn_baseline", BEST_CKPT)
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
    print(f"logs: {log_dir('cnn_baseline')}")
    print(f"last ckpt: {last_ckpt_path('cnn_baseline')}", flush=True)
    trainer = make_trainer(
        "cnn_baseline", max_epochs=MAX_EPOCHS, monitor="macro_f1", mode="max", patience=PATIENCE,
        extra_callbacks=[SaveBest(BEST_CKPT, "cnn_bat_a0", label2id, id2label, initial_best_f1=initial_best)],
        continuing_run=args.resume is not None,
        restore_completed_epochs=completed_epochs if pl_ckpt is None else 0,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(pl_ckpt) if pl_ckpt else None)

    if BEST_CKPT.is_file():
        load_weights(model, BEST_CKPT)
    final_eval(module, val_loader, species, CONFUSION_PNG)


if __name__ == "__main__":
    main()
