import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pytorch_lightning as pl

import config as cfg
from bat.data import load_split, make_loaders
from bat.data.audio import SPEC_H, SPEC_W
from bat.lightning_utils import ClassifierModule, SaveBest, final_eval, load_weights, log_dir, make_trainer
from vit_bat import BatViTClassifier, load_ssl_encoder

CONFUSION_PNG = cfg.CHECKPOINT_DIR / "vit_lfpe_bat_confusion_matrix.png"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", nargs="?", const=str(cfg.FT_BEST_CKPT), default=None)
    p.add_argument("--no-ssl", action="store_true")
    args = p.parse_args()
    resume = Path(args.resume) if args.resume else None

    pl.seed_everything(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, label2id, id2label, species = load_split(cfg.FT_METADATA_PATH, cfg.FT_DATA_DIR)
    train_loader, val_loader = make_loaders(train_df, val_df, mode="supervised", balanced=True)

    ssl_ok = not args.no_ssl and cfg.LOAD_SSL_PRETRAIN
    enc_lr = cfg.FT_ENCODER_LR_SSL if ssl_ok else cfg.FT_ENCODER_LR_NO_SSL
    head_lr = cfg.FT_HEAD_LR
    model = BatViTClassifier(len(species), SPEC_H, SPEC_W)
    module = ClassifierModule(
        model,
        [
            {"params": model.encoder_parameters(), "lr": enc_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        weight_decay=1e-4, plateau_patience=6, lr_factor=0.5, lr_min=1e-7, label_smoothing=0.1,
        mixup_alpha=cfg.FT_MIXUP_ALPHA,
    )
    if ssl_ok:
        load_ssl_encoder(model, cfg.BEST_CKPT, next(model.parameters()).device, (SPEC_H, SPEC_W))
    if resume and resume.is_file():
        load_weights(model, resume)

    extra = {"ssl_pretrain_loaded": ssl_ok, "encoder_lr": enc_lr, "head_lr": head_lr, "mixup_alpha": cfg.FT_MIXUP_ALPHA}
    trainer = make_trainer(
        "finetune", max_epochs=cfg.FT_MAX_EPOCHS, monitor="macro_f1", mode="max", patience=cfg.FT_PATIENCE,
        extra_callbacks=[SaveBest(cfg.FT_BEST_CKPT, "vit_lfpe_bat", label2id, id2label, extra=extra)],
    )
    print(f"logs: {log_dir('finetune')}")
    trainer.fit(module, train_loader, val_loader)

    if cfg.FT_BEST_CKPT.is_file():
        load_weights(model, cfg.FT_BEST_CKPT)
    final_eval(module, val_loader, species, CONFUSION_PNG)


if __name__ == "__main__":
    main()
