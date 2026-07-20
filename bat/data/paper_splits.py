"""Загрузка готовых split'ов NABat paper-style (trainval / test)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as cfg

PULSE_COLUMNS = ["path", "species", "filename", "label", "pulse_center"]


def _read_pulses(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "pulse_center" not in df.columns and "window_offset_ms" in df.columns:
        df["pulse_center"] = df["window_offset_ms"].astype(int)
    missing = [c for c in PULSE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[PULSE_COLUMNS].copy()


def load_paper_trainval(
    base_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[int, str], list[str]]:
    """train + val из trainval/; label2id строится по train."""
    root = base_dir or cfg.NABAT_PAPER_TRAINVAL_DIR
    train_df = _read_pulses(root / "pulses_train.csv")
    val_df = _read_pulses(root / "pulses_val.csv")

    species = sorted(train_df["species"].unique())
    label2id = {name: i for i, name in enumerate(species)}
    id2label = {i: name for name, i in label2id.items()}

    for df in (train_df, val_df):
        unknown = set(df["species"].unique()) - set(label2id)
        if unknown:
            raise ValueError(f"species not in train label map: {sorted(unknown)}")

    train_df["label"] = train_df["species"].map(label2id)
    val_df["label"] = val_df["species"].map(label2id)
    return train_df, val_df, label2id, id2label, species


def load_paper_test(
    *,
    base_dir: Path | None = None,
    label2id: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Test split; label2id должен совпадать с trainval."""
    root = base_dir or cfg.NABAT_PAPER_TEST_DIR
    test_df = _read_pulses(root / "pulses_test.csv")
    if label2id is None:
        _, _, label2id, _, _ = load_paper_trainval()
    unknown = set(test_df["species"].unique()) - set(label2id)
    if unknown:
        raise ValueError(f"test species not in label map: {sorted(unknown)}")
    test_df["label"] = test_df["species"].map(label2id)
    return test_df
