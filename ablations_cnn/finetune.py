import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import pytorch_lightning as pl

import config as cfg
from ablations_cnn.cnn_ssl import BatCNNClassifier, load_ssl_encoder
from ablations_cnn.presets import ABLATION_ROOT, confusion_path, ft_ckpt_path, ft_run_name, parse_preset, ssl_ckpt_path
from bat.data import load_paper_trainval, make_loaders
from bat.lightning_utils import ClassifierModule, SaveBest, final_eval, load_weights, log_dir, make_trainer, resolve_resume

LR, WD, LS = 1e-3, 1e-4, 0.1
MAX_EPOCHS, PATIENCE = 40, 10
PLATEAU_PATIENCE, LR_FACTOR, LR_MIN = 5, 0.5, 1e-7


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", required=True)
    p.add_argument("--ssl-ckpt", default=None)
    p.add_argument(
        "--resume", nargs="?", const="__auto__", default=None,
        help="без пути: last.ckpt; .ckpt — полный resume; .pt — только веса",
    )
    p.add_argument(
        "--mixup", action="store_true",
        help=f"MixUp при train (α={cfg.FT_MIXUP_ALPHA}, как ViT finetune)",
    )
    args = p.parse_args()

    tasks = parse_preset(args.preset)
    tag = "mixup" if args.mixup else ""
    mixup_alpha = cfg.FT_MIXUP_ALPHA if args.mixup else 0.0
    best_ckpt = ft_ckpt_path(tasks, tag)
    run_name = ft_run_name(tasks, tag)
    ssl_path = args.ssl_ckpt or ssl_ckpt_path(tasks)

    pl.seed_everything(cfg.RANDOM_SEED)
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    train_df, val_df, label2id, id2label, species = load_paper_trainval()
    train_loader, val_loader = make_loaders(
        train_df, val_df, mode="supervised", balanced=True, cache_dir=cfg.NABAT_PAPER_SPEC_CACHE,
    )

    model = BatCNNClassifier(len(species))
    load_ssl_encoder(model, ssl_path)
    print(f"loaded SSL encoder: {ssl_path}", flush=True)

    module = ClassifierModule(
        model, [{"params": model.parameters(), "lr": LR}],
        weight_decay=WD, plateau_patience=PLATEAU_PATIENCE, lr_factor=LR_FACTOR, lr_min=LR_MIN,
        label_smoothing=LS, mixup_alpha=mixup_alpha,
    )

    weights_ckpt, pl_ckpt, completed_epochs = resolve_resume(args.resume, run_name, best_ckpt)
    initial_best = -1.0
    if weights_ckpt is not None:
        meta = load_weights(model, weights_ckpt)
        initial_best = float(meta.get("val_macro_f1", -1.0))
        print(
            f"resume weights: {weights_ckpt} (epoch={meta.get('epoch', '?')}, "
            f"macro_f1={initial_best:.4f}, next epoch={completed_epochs})",
            flush=True,
        )
    elif pl_ckpt is not None:
        print(f"resume trainer: {pl_ckpt}", flush=True)

    trainer = make_trainer(
        run_name, max_epochs=MAX_EPOCHS, monitor="macro_f1", mode="max", patience=PATIENCE,
        extra_callbacks=[
            SaveBest(
                best_ckpt, f"cnn_ssl_{tasks.id.replace('+', '_')}",
                label2id, id2label,
                extra={"preset": tasks.id, "ssl_ckpt": str(ssl_path), "mixup_alpha": mixup_alpha},
                initial_best_f1=initial_best,
            )
        ],
        continuing_run=args.resume is not None,
        restore_completed_epochs=completed_epochs if pl_ckpt is None else 0,
    )
    print(f"preset: {tasks.id}  mixup_alpha={mixup_alpha}", flush=True)
    print(f"logs: {log_dir(run_name)}", flush=True)
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(pl_ckpt) if pl_ckpt else None)

    if best_ckpt.is_file():
        load_weights(model, best_ckpt)
    final_eval(module, val_loader, species, confusion_path(tasks, tag))


if __name__ == "__main__":
    main()
