import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg
from bat.data.audio import find_pulses, read_wav


def expand_by_pulses(df):
    """Один wav -> несколько строк (по одной на импульс)."""
    rows = []
    for _, row in df.iterrows():
        y = read_wav(row["path"])
        centers = find_pulses(y)
        if not centers:
            centers = [int(np.argmax(y ** 2))]

        for center in centers:
            rows.append({
                "path": row["path"],
                "species": row["species"],
                "filename": row["filename"],
                "label": row["label"],
                "pulse_center": center,
            })
    return pd.DataFrame(rows)


def load_split(metadata_path, data_dir, expand_pulses=True):
    df = pd.read_csv(metadata_path)
    df["path"] = df.apply(lambda r: data_dir / r["species"] / r["filename"], axis=1)
    df = df[df["path"].apply(lambda p: p.exists())].copy()

    species = sorted(df["species"].unique())
    label2id = {name: i for i, name in enumerate(species)}
    id2label = {i: name for name, i in label2id.items()}
    df["label"] = df["species"].map(label2id)

    train_files, val_files = train_test_split(
        df,
        test_size=cfg.VAL_TEST_SIZE,
        random_state=cfg.RANDOM_SEED,
        stratify=df["label"],
    )
    if expand_pulses:
        train_df = expand_by_pulses(train_files)
        val_df = expand_by_pulses(val_files)
    else:
        train_df = train_files.reset_index(drop=True)
        val_df = val_files.reset_index(drop=True)
    return train_df, val_df, label2id, id2label, species
