import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg
from bat.data.audio import find_pulses

PULSE_CACHE_DIR = cfg.CHECKPOINT_DIR / "pulse_cache"


def _pulse_cache_path(wav_path):
    p = Path(wav_path).resolve()
    st = p.stat()
    key = f"{p}|{st.st_mtime_ns}|{st.st_size}|gottbat_v2"
    digest = hashlib.sha1(key.encode()).hexdigest()
    return PULSE_CACHE_DIR / f"{digest}.json"


def get_pulse_centers(wav_path):
    p = Path(wav_path)
    cache = _pulse_cache_path(p)
    if cache.is_file():
        data = json.loads(cache.read_text())
        st = p.stat()
        if data.get("mtime_ns") == st.st_mtime_ns and data.get("size") == st.st_size:
            return data["centers"]

    centers = find_pulses(p)

    PULSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    st = p.stat()
    cache.write_text(json.dumps({"mtime_ns": st.st_mtime_ns, "size": st.st_size, "centers": centers}))
    return centers


def expand_by_pulses(df, desc="pulses"):
    """Один wav -> несколько строк (по одной на импульс gottbat)."""
    rows = []
    n = len(df)
    skipped_files = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        if i == 1 or i % 25 == 0 or i == n:
            print(f"{desc}: {i}/{n}", flush=True)
        centers = get_pulse_centers(row.path)
        if not centers:
            skipped_files += 1
            continue
        for center in centers:
            rows.append({
                "path": row.path,
                "species": row.species,
                "filename": row.filename,
                "label": row.label,
                "pulse_center": center,
            })
    if skipped_files:
        print(f"{desc}: skipped {skipped_files} files with no pulses", flush=True)
    return pd.DataFrame(rows)


def load_split(metadata_path, data_dir, expand_pulses=True, expand_train=True):
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
        parts = []
        if expand_train:
            parts.append(f"{len(train_files)} train")
        parts.append(f"{len(val_files)} val")
        print(f"load_split: {' + '.join(parts)} files, pulse_cache + spec_cache...", flush=True)
        if expand_train:
            train_df = expand_by_pulses(train_files, desc="train pulses")
        else:
            train_df = train_files.reset_index(drop=True)
        val_df = expand_by_pulses(val_files, desc="val pulses")
        train_unit = "examples" if expand_train else "files"
        print(f"load_split: {len(train_df)} train {train_unit} + {len(val_df)} val examples", flush=True)
    else:
        train_df = train_files.reset_index(drop=True)
        val_df = val_files.reset_index(drop=True)
    return train_df, val_df, label2id, id2label, species
