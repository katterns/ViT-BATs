"""Inference официальной NABat ML модели (gottbat, TF SavedModel m-1) на test split.

Использует предобработанные спектрограммы из spec_cache (nabat_paper_31/test).
Сравнивает метрики с Khalighifar et al., 2022 (NABat ML).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)
from tensorflow.python.saved_model import tag_constants

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from bat.data.paper_splits import load_paper_test, load_paper_trainval
from bat.data.spec_cache import load_base_spec

GOTTBAT_DIR = ROOT / "third_party" / "gottbat_full"
MODEL_DIR = GOTTBAT_DIR / "prediction" / "tf-models" / "m-1"
HISTORY_PATH = GOTTBAT_DIR / "prediction" / "tf-models" / "training_history_m-1.p"
REPORT_PATH = cfg.CHECKPOINT_DIR / "nabat_official_test_eval.md"

PAPER_METRICS = {
    "pulse_accuracy": 0.83,
    "pulse_weighted_precision": 0.80,
    "file_weighted_accuracy": 0.92,
    "conf_threshold": 0.57,
}


class NabatOfficialPredictor:
    """Официальная CNN NABat ML (SavedModel m-1), inference через TF1 compat API."""

    def __init__(self, model_dir: Path = MODEL_DIR, history_path: Path = HISTORY_PATH):
        with history_path.open("rb") as fp:
            self.class_names: list[str] = pickle.load(fp)[1]
        self.name_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self._graph = tf.Graph()
        self._sess = tf.compat.v1.Session(graph=self._graph)
        with self._graph.as_default():
            meta = tf.compat.v1.saved_model.loader.load(
                self._sess, [tag_constants.SERVING], str(model_dir)
            )
            sig = meta.signature_def["serving_default"]
            self._input = self._graph.get_tensor_by_name(sig.inputs["input_1"].name)
            self._output = self._graph.get_tensor_by_name(
                list(sig.outputs.values())[0].name
            )

    def predict_batch(self, images_hwc: np.ndarray) -> np.ndarray:
        return self._sess.run(self._output, feed_dict={self._input: images_hwc})

    def close(self) -> None:
        self._sess.close()


def chw_to_hwc(spec_chw: np.ndarray) -> np.ndarray:
    return np.transpose(spec_chw, (1, 2, 0)).astype(np.float32, copy=False)


def load_specs_batch(
    rows: pd.DataFrame,
    *,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    indices: list[int] = []
    for i, row in enumerate(rows.itertuples()):
        spec = load_base_spec(row.path, int(row.pulse_center), cache_dir=cache_dir)
        if spec is None:
            continue
        images.append(chw_to_hwc(spec))
        indices.append(i)
    if not images:
        return np.empty((0, 100, 100, 3), dtype=np.float32), np.array([], dtype=int)
    return np.stack(images, axis=0), np.array(indices, dtype=int)


def predict_all_pulses(
    df: pd.DataFrame,
    predictor: NabatOfficialPredictor,
    *,
    cache_dir: Path,
    batch_size: int,
) -> np.ndarray:
    probs = np.zeros((len(df), len(predictor.class_names)), dtype=np.float32)
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        batch, local_idx = load_specs_batch(chunk, cache_dir=cache_dir)
        if len(batch) == 0:
            continue
        pred = predictor.predict_batch(batch)
        global_idx = start + local_idx
        probs[global_idx] = pred
    return probs


def species_to_model_idx(
    species: str,
    predictor: NabatOfficialPredictor,
) -> int | None:
    return predictor.name_to_idx.get(species)


def pulse_metrics(
    y_true_species: np.ndarray,
    probs: np.ndarray,
    predictor: NabatOfficialPredictor,
) -> dict:
    pred_idx = probs.argmax(axis=1)
    pred_species = np.array([predictor.class_names[i] for i in pred_idx])
    acc = accuracy_score(y_true_species, pred_species)
    macro_f1 = f1_score(y_true_species, pred_species, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true_species, pred_species, average="weighted", zero_division=0
    )
    weighted_prec = precision_score(
        y_true_species, pred_species, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_prec,
        "pred_species": pred_species,
        "report": classification_report(
            y_true_species, pred_species, zero_division=0, digits=4
        ),
    }


def file_level_mean_prob(
    df: pd.DataFrame,
    probs: np.ndarray,
    predictor: NabatOfficialPredictor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    file_keys = df["path"].values
    unique_files = pd.unique(file_keys)
    y_true: list[str] = []
    y_pred: list[str] = []
    conf: list[float] = []

    for path in unique_files:
        mask = file_keys == path
        file_df = df.loc[mask]
        mean_prob = probs[mask].mean(axis=0)
        pred_idx = int(mean_prob.argmax())
        y_true.append(file_df["species"].iloc[0])
        y_pred.append(predictor.class_names[pred_idx])
        conf.append(float(mean_prob[pred_idx]))

    return np.array(y_true), np.array(y_pred), np.array(conf)


def file_level_majority(
    df: pd.DataFrame,
    pred_species: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    work = df[["path", "species"]].copy()
    work["_pred"] = pred_species
    y_true: list[str] = []
    y_pred: list[str] = []
    for _, group in work.groupby("path", sort=False):
        y_true.append(group["species"].iloc[0])
        y_pred.append(Counter(group["_pred"]).most_common(1)[0][0])
    return np.array(y_true), np.array(y_pred)


def summarize_file_level(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def per_class_file_id_rate(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for species in sorted(np.unique(y_true)):
        mask = y_true == species
        n = int(mask.sum())
        correct = int((y_pred[mask] == species).sum())
        rows.append(
            {
                "species": species,
                "id_rate": correct / n if n else float("nan"),
                "correct": correct,
                "total": n,
            }
        )
    return pd.DataFrame(rows).sort_values("id_rate", ascending=False)


def _df_to_md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for row in df.itertuples(index=False):
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.1%}" if 0 <= val <= 1 else f"{val:.4f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def format_comparison(ours: dict, paper: dict) -> str:
    lines = [
        "## Сравнение с NABat ML (Khalighifar et al., 2022)",
        "",
        "| Метрика | NABat ML (статья) | Официальная модель (наш test) | Δ |",
        "|---------|-------------------|-------------------------------|---|",
    ]
    rows = [
        ("Pulse-level accuracy", paper["pulse_accuracy"], ours["pulse_accuracy"]),
        (
            "Pulse-level weighted precision",
            paper["pulse_weighted_precision"],
            ours["pulse_weighted_precision"],
        ),
        (
            "File-level weighted accuracy (mean prob)",
            paper["file_weighted_accuracy"],
            ours["file_weighted_accuracy"],
        ),
    ]
    for name, paper_val, our_val in rows:
        delta = our_val - paper_val
        lines.append(
            f"| {name} | {paper_val:.1%} | {our_val:.1%} | {delta:+.1%} |"
        )
    lines.extend(
        [
            "",
            "**Оговорки:**",
            "- Holdout test из статьи не входит в USGS release — используется локальный proxy split (80/10/10).",
            "- 92% file-level accuracy в статье получена **с range maps**; здесь range maps не применялись.",
            "- Официальная модель обучена на 31 классе; test split — 28 классов (без EUMA/EUPE/NYMA и редких видов).",
        ]
    )
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    summary: dict,
    pulse_report: str,
    per_class: pd.DataFrame,
    comparison_md: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "# NABat ML official model — test inference",
        "",
        f"Модель: `{MODEL_DIR.relative_to(ROOT)}`",
        f"Test: `{cfg.NABAT_PAPER_TEST_DIR.relative_to(ROOT)}/pulses_test.csv`",
        "",
        "## Сводка",
        "",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "",
        comparison_md,
        "",
        "## Per-class file-level ID rate (mean prob)",
        "",
        _df_to_md_table(per_class),
        "",
        "## Pulse-level classification report",
        "",
        "```",
        pulse_report.strip(),
        "```",
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Eval official NABat ML model on test split")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="Ограничить число импульсов (0 = все)")
    p.add_argument("--out", type=Path, default=REPORT_PATH)
    args = p.parse_args()

    tf.get_logger().setLevel("ERROR")

    train_df, _, label2id, _, _ = load_paper_trainval()
    test_df = load_paper_test(label2id=label2id)
    if args.limit > 0:
        test_df = test_df.head(args.limit).copy()

    cache_dir = cfg.NABAT_PAPER_TEST_SPEC_CACHE
    n_files = test_df["path"].nunique()
    n_species = test_df["species"].nunique()
    print(
        f"Test: {len(test_df)} pulses, {n_files} files, {n_species} species",
        flush=True,
    )

    predictor = NabatOfficialPredictor()
    print(f"Loaded model m-1 ({len(predictor.class_names)} classes)", flush=True)

    probs = predict_all_pulses(
        test_df,
        predictor,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
    )
    y_true_species = test_df["species"].values

    pulse = pulse_metrics(y_true_species, probs, predictor)
    print(f"Pulse-level accuracy: {pulse['accuracy']:.1%}", flush=True)
    print(f"Pulse-level macro-F1: {pulse['macro_f1']:.4f}", flush=True)

    file_true, file_pred, file_conf = file_level_mean_prob(test_df, probs, predictor)
    file_mean = summarize_file_level(file_true, file_pred)
    print(f"File-level accuracy (mean prob): {file_mean['accuracy']:.1%}", flush=True)

    maj_true, maj_pred = file_level_majority(test_df, pulse["pred_species"])
    file_maj = summarize_file_level(maj_true, maj_pred)

    conf_mask = file_conf >= PAPER_METRICS["conf_threshold"]
    conf_subset = summarize_file_level(file_true[conf_mask], file_pred[conf_mask])

    per_class = per_class_file_id_rate(file_true, file_pred)
    ge90 = int((per_class["id_rate"] >= 0.90).sum())

    summary = {
        "n_pulses": len(test_df),
        "n_files": int(n_files),
        "n_species": int(n_species),
        "model_classes": len(predictor.class_names),
        "pulse_accuracy": round(pulse["accuracy"], 4),
        "pulse_macro_f1": round(pulse["macro_f1"], 4),
        "pulse_weighted_precision": round(pulse["weighted_precision"], 4),
        "file_accuracy_mean_prob": round(file_mean["accuracy"], 4),
        "file_macro_f1_mean_prob": round(file_mean["macro_f1"], 4),
        "file_weighted_precision_mean_prob": round(file_mean["weighted_precision"], 4),
        "file_accuracy_majority": round(file_maj["accuracy"], 4),
        "file_conf_ge_057_accuracy": round(conf_subset["accuracy"], 4),
        "file_conf_ge_057_kept": int(conf_mask.sum()),
        "file_conf_ge_057_dropped": int((~conf_mask).sum()),
        "classes_ge_90pct_file_id": ge90,
    }

    comparison_md = format_comparison(
        {
            "pulse_accuracy": pulse["accuracy"],
            "pulse_weighted_precision": pulse["weighted_precision"],
            "file_weighted_accuracy": file_mean["accuracy"],
        },
        PAPER_METRICS,
    )

    write_report(
        args.out,
        summary=summary,
        pulse_report=pulse["report"],
        per_class=per_class,
        comparison_md=comparison_md,
    )
    print(comparison_md, flush=True)
    print(f"\nReport saved: {args.out}", flush=True)

    predictor.close()


if __name__ == "__main__":
    main()
