"""Test inference + метрики (pulse/file level) для CNN и ablations_cnn."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from ablations_cnn.cnn_ssl import BatCNNClassifier
from ablations_cnn.presets import ALL_PRESETS, FT_CKPT_DIR, parse_preset
from bat.data import load_paper_test, load_paper_trainval
from bat.data.audio import load_spec
from bat.lightning_utils import load_weights
from supervised_cnn_baseline import BatCNNA0

CONF_THRESHOLD = 0.57
BATCH_SIZE = 64


def load_cnn_classifier(ckpt_path, n_classes=None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if n_classes is None:
        n_classes = len(ckpt["label2id"])
    model = BatCNNClassifier(n_classes)
    load_weights(model, ckpt_path)
    return model, ckpt


def load_supervised_cnn(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BatCNNA0(len(ckpt["label2id"]))
    load_weights(model, ckpt_path)
    return model, ckpt


@torch.no_grad()
def predict_df(model, df, cache_dir, device):
    model.eval()
    probs, labels, paths, filenames = [], [], [], []
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]
        xs, ys = [], []
        for _, row in batch.iterrows():
            xs.append(load_spec(row["path"], False, None, pulse_center=int(row["pulse_center"]), cache_dir=cache_dir))
            ys.append(int(row["label"]))
        x = torch.stack(xs).to(device)
        logits = model(x)
        probs.append(F.softmax(logits, dim=-1).cpu().numpy())
        labels.extend(ys)
        paths.extend(batch["path"].tolist())
        filenames.extend(batch["filename"].tolist())
    return np.vstack(probs), np.array(labels), paths, filenames


def file_metrics(probs, labels, paths, id2label):
    by_file = defaultdict(list)
    file_true = {}
    for i, path in enumerate(paths):
        by_file[path].append(probs[i])
        file_true[path] = int(labels[i])

    file_pred, file_conf, file_true_l = [], [], []
    for path, pulse_probs in by_file.items():
        mean_prob = np.mean(np.stack(pulse_probs), axis=0)
        file_pred.append(int(mean_prob.argmax()))
        file_conf.append(float(mean_prob.max()))
        file_true_l.append(file_true[path])

    file_pred = np.array(file_pred)
    file_true_l = np.array(file_true_l)
    file_conf = np.array(file_conf)

    majority = []
    for path, pulse_probs in by_file.items():
        arr = np.stack(pulse_probs)
        preds = arr.argmax(axis=1)
        majority.append(int(Counter(preds).most_common(1)[0][0]))

    kept = file_conf >= CONF_THRESHOLD
    per_class = {}
    for cls_id in sorted(set(file_true_l)):
        mask = file_true_l == cls_id
        if mask.sum() == 0:
            continue
        per_class[id2label[cls_id]] = float((file_pred[mask] == cls_id).mean())

    return {
        "n_files": len(by_file),
        "file_accuracy_mean_prob": float(accuracy_score(file_true_l, file_pred)),
        "file_macro_f1_mean_prob": float(f1_score(file_true_l, file_pred, average="macro", zero_division=0)),
        "file_weighted_precision_mean_prob": float(
            precision_score(file_true_l, file_pred, average="weighted", zero_division=0)
        ),
        "file_accuracy_majority": float(accuracy_score(file_true_l, np.array(majority))),
        "file_conf_ge_057_accuracy": float(accuracy_score(file_true_l[kept], file_pred[kept])) if kept.any() else float("nan"),
        "file_conf_ge_057_kept": int(kept.sum()),
        "file_conf_ge_057_dropped": int((~kept).sum()),
        "classes_ge_90pct_file_id": int(sum(1 for v in per_class.values() if v >= 0.9)),
        "per_class_file_id_rate": per_class,
    }


def eval_checkpoint(name, ckpt_path, test_df, cache_dir, device, *, supervised=False):
    if supervised:
        model, ckpt = load_supervised_cnn(ckpt_path)
    else:
        model, ckpt = load_cnn_classifier(ckpt_path)
    id2label = {int(k): v for k, v in ckpt["id2label"].items()}

    probs, labels, paths, filenames = predict_df(model, test_df, cache_dir, device)
    pulse_pred = probs.argmax(axis=1)

    fm = file_metrics(probs, labels, paths, id2label)
    return {
        "name": name,
        "checkpoint": str(ckpt_path),
        "val_macro_f1": float(ckpt.get("val_macro_f1", float("nan"))),
        "n_pulses": int(len(labels)),
        "pulse_accuracy": float(accuracy_score(labels, pulse_pred)),
        "pulse_macro_f1": float(f1_score(labels, pulse_pred, average="macro", zero_division=0)),
        "pulse_weighted_precision": float(
            precision_score(labels, pulse_pred, average="weighted", zero_division=0)
        ),
        **fm,
    }


def discover_ablation_checkpoints(*, include_mixup=False):
    found = []
    for preset in ALL_PRESETS:
        path = FT_CKPT_DIR / f"{preset.id}_best.pt"
        if path.is_file():
            found.append((f"CNN+{preset.id}", path))
        if include_mixup:
            mix_path = FT_CKPT_DIR / f"{preset.id}_mixup_best.pt"
            if mix_path.is_file():
                found.append((f"CNN+{preset.id}+mixup", mix_path))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=cfg.CHECKPOINT_DIR / "test_eval_results.json")
    p.add_argument("--skip-cnn", action="store_true")
    p.add_argument("--preset", default=None, help="single ablation preset id")
    p.add_argument("--mixup", action="store_true", help="eval *_mixup_best.pt вместо обычного")
    p.add_argument("--include-mixup", action="store_true", help="в discover добавить *_mixup_best.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, label2id, id2label, _ = load_paper_trainval()
    test_df = load_paper_test(label2id=label2id)
    cache_dir = cfg.NABAT_PAPER_TEST_SPEC_CACHE

    models = []
    if not args.skip_cnn:
        cnn_ckpt = cfg.CHECKPOINT_DIR / "cnn_bat_a0_best.pt"
        if cnn_ckpt.is_file():
            models.append(("CNN", cnn_ckpt, True))
    if args.preset:
        pid = parse_preset(args.preset).id
        suffix = "_mixup" if args.mixup else ""
        path = FT_CKPT_DIR / f"{pid}{suffix}_best.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        name = f"CNN+{args.preset}+mixup" if args.mixup else f"CNN+{args.preset}"
        models.append((name, path, False))
    else:
        for name, path in discover_ablation_checkpoints(include_mixup=args.include_mixup):
            models.append((name, path, False))

    if not models:
        raise RuntimeError("no checkpoints found")

    results = []
    for name, ckpt_path, supervised in models:
        print(f"eval {name} ...", flush=True)
        results.append(eval_checkpoint(name, ckpt_path, test_df, cache_dir, device, supervised=supervised))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {args.out}", flush=True)
    for r in results:
        print(
            f"{r['name']:22s}  pulse F1={r['pulse_macro_f1']:.3f}  "
            f"file acc={r['file_accuracy_mean_prob']:.1%}  conf≥0.57={r['file_conf_ge_057_accuracy']:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
