"""Сборка датасета, максимально близкого к Khalighifar et al. (2022).

Классы: без CORA/EUFL/LAXA/NYFE. Сколько видов есть на диске — столько и в корпусе
(сейчас часто 28; после докачки EUMA/NYMA/EUPE → до 31).

Структура:
  data/nabat_paper_31/
    trainval/  files_*.csv, pulses_*.csv, spec_cache/
    test/      files_test.csv, pulses_test.csv, spec_cache/
    summary.json

Split: 80/10/10 на уровне файлов → pulse expansion (gottbat).

Обновление после докачки:
  uv run python scripts/build_nabat_paper_dataset.py \\
    --wav-dir ./data --scan-extracted --no-extract --update --precompute-cache

Первая сборка:
  uv run python scripts/build_nabat_paper_dataset.py \\
    --wav-dir ./data --scan-extracted --no-extract --precompute-cache
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_cleaned import (  # noqa: E402
    EXCLUDED_PAPER,
    extract_files,
    resolve_member,
    scan_raw,
    write_manifest,
)
from bat.data.audio import precompute_specs  # noqa: E402
from bat.data.nabat import process_file  # noqa: E402
from bat.data.spec_cache import cache_stats  # noqa: E402

PAPER_TRAIN_RATIO = 0.80
PAPER_VAL_RATIO = 0.10
PAPER_TEST_RATIO = 0.10
PAPER_CLASSES = 31
PAPER_FILES_TARGET = 23_835
PAPER_TRAIN_PULSES_TARGET = 611_637
DEFAULT_SEED = 42
SPECIES_DIR_RE = re.compile(r"^[A-Z]{4}$|^NOISE$")

FILE_COLUMNS = ["species", "filename", "duration", "sample_rate", "archive", "split", "path"]
PULSE_COLUMNS = [
    "species",
    "filename",
    "archive",
    "duration",
    "sample_rate",
    "window_offset_ms",
    "pulse_center",
    "peak_frequency_hz",
    "peak_time_ms",
    "snr",
    "amplitude",
    "split",
    "path",
    "label",
]
FILE_KEY = ["species", "filename"]


def scan_usgs_release(raw_dir: Path) -> pd.DataFrame:
    return scan_raw(raw_dir, excluded=EXCLUDED_PAPER, duration_filter=False)


def scan_extracted_wavs(wav_root: Path) -> pd.DataFrame:
    """Файлы из wav_root/<SPECIES>/*.wav."""
    import soundfile as sf

    rows: list[dict] = []
    for species_dir in sorted(wav_root.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name.upper()
        if not SPECIES_DIR_RE.match(species) or species in EXCLUDED_PAPER:
            continue
        for wav_path in sorted(species_dir.glob("*.wav")):
            try:
                info = sf.info(str(wav_path))
            except Exception as exc:
                print(f"skip {wav_path}: {exc}", flush=True)
                continue
            rows.append({
                "species": species,
                "filename": wav_path.name,
                "duration": float(info.duration),
                "sample_rate": int(info.samplerate),
                "archive": f"{species}.zip",
            })

    if not rows:
        raise RuntimeError(f"no WAV in {wav_root}/<SPECIES>/")

    return (
        pd.DataFrame(rows, columns=FILE_COLUMNS[:-2])
        .sort_values(["species", "filename"])
        .reset_index(drop=True)
    )


def _abs_paths(files_df: pd.DataFrame, wav_dir: Path) -> pd.DataFrame:
    out = files_df.copy()
    out["path"] = out.apply(
        lambda r: str((wav_dir / r["species"] / r["filename"]).resolve()),
        axis=1,
    )
    return out


def assign_file_splits(
    files_df: pd.DataFrame,
    seed: int,
    *,
    train_ratio: float = PAPER_TRAIN_RATIO,
    val_ratio: float = PAPER_VAL_RATIO,
    test_ratio: float = PAPER_TEST_RATIO,
) -> pd.DataFrame:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1")
    if len(files_df) == 0:
        out = files_df.copy()
        out["split"] = pd.Series(dtype=str)
        return out

    # мало файлов у вида → без stratify
    counts = files_df["species"].value_counts()
    can_stratify = counts.min() >= 2 and len(files_df) >= 3

    if not can_stratify:
        train_val, test_files = train_test_split(
            files_df, test_size=test_ratio, random_state=seed,
        )
        val_share = val_ratio / (train_ratio + val_ratio)
        train_files, val_files = train_test_split(
            train_val, test_size=val_share, random_state=seed,
        )
    else:
        train_val, test_files = train_test_split(
            files_df,
            test_size=test_ratio,
            random_state=seed,
            stratify=files_df["species"],
        )
        val_share = val_ratio / (train_ratio + val_ratio)
        # после первого split у редких видов может остаться 1 файл
        if train_val["species"].value_counts().min() < 2:
            train_files, val_files = train_test_split(
                train_val, test_size=val_share, random_state=seed,
            )
        else:
            train_files, val_files = train_test_split(
                train_val,
                test_size=val_share,
                random_state=seed,
                stratify=train_val["species"],
            )

    out = files_df.copy()
    split_map = {
        **{p: "train" for p in train_files["path"]},
        **{p: "val" for p in val_files["path"]},
        **{p: "test" for p in test_files["path"]},
    }
    out["split"] = out["path"].map(split_map)
    return out


def load_existing_file_splits(trainval_dir: Path, test_dir: Path) -> pd.DataFrame | None:
    paths = [
        trainval_dir / "files_train.csv",
        trainval_dir / "files_val.csv",
        test_dir / "files_test.csv",
    ]
    if not all(p.is_file() for p in paths):
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def merge_splits_update(
    files_df: pd.DataFrame,
    old_files: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Сохраняет split для уже известных файлов; новым — стратифицированный assign."""
    old_key = old_files.drop_duplicates(FILE_KEY).set_index(FILE_KEY)["split"]
    out = files_df.copy()
    keys = list(zip(out["species"], out["filename"]))
    out["split"] = [old_key.get(k) for k in keys]

    new_mask = out["split"].isna()
    n_new = int(new_mask.sum())
    n_kept = len(out) - n_new
    print(f"update: keep split for {n_kept} files, assign {n_new} new", flush=True)

    if n_new:
        assigned = assign_file_splits(out.loc[new_mask].drop(columns=["split"]), seed)
        out.loc[new_mask, "split"] = assigned["split"].values
    return out


def _attach_labels(pulses_df: pd.DataFrame, label2id: dict[str, int]) -> pd.DataFrame:
    out = pulses_df.copy()
    out["label"] = out["species"].map(label2id)
    unknown = out["label"].isna()
    if unknown.any():
        bad = sorted(out.loc[unknown, "species"].unique())
        raise ValueError(f"species without label id: {bad}")
    out["label"] = out["label"].astype(int)
    return out


def _pulse_rows_for_wav(path: Path, file_row: pd.Series) -> list[dict]:
    data = process_file(path)
    if data is None or not data.metadata:
        return []

    rows = []
    for m in data.metadata:
        rows.append({
            "species": file_row["species"],
            "filename": file_row["filename"],
            "archive": file_row["archive"],
            "duration": file_row["duration"],
            "sample_rate": file_row["sample_rate"],
            "window_offset_ms": int(m.offset),
            "pulse_center": int(m.offset),
            "peak_frequency_hz": float(m.frequency),
            "peak_time_ms": float(m.time),
            "snr": float(m.snr),
            "amplitude": float(m.amplitude),
            "split": file_row["split"],
            "path": str(Path(path).resolve()),
        })
    return rows


def build_pulse_manifest(
    files_df: pd.DataFrame,
    wav_dir: Path,
    *,
    existing: pd.DataFrame | None = None,
    checkpoint: Path | None = None,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """Считает импульсы; пишет checkpoint CSV, чтобы не терять прогресс при обрыве."""
    parts: list[pd.DataFrame] = []
    if existing is not None and len(existing):
        parts.append(existing[PULSE_COLUMNS[:-1]])
    if checkpoint is not None and checkpoint.is_file():
        parts.append(pd.read_csv(checkpoint))
        print(f"pulses: loaded checkpoint {checkpoint} ({len(parts[-1])} rows)", flush=True)

    if parts:
        kept = pd.concat(parts, ignore_index=True).drop_duplicates(
            subset=["species", "filename", "pulse_center"],
            keep="last",
        )
        done_keys = set(zip(kept["species"], kept["filename"]))
        todo = files_df[
            ~files_df.apply(lambda r: (r["species"], r["filename"]) in done_keys, axis=1)
        ].copy()
        keep_keys = set(zip(files_df["species"], files_df["filename"]))
        kept = kept[
            kept.apply(lambda r: (r["species"], r["filename"]) in keep_keys, axis=1)
        ].copy()
        if len(kept):
            meta = files_df.set_index(FILE_KEY)[["split", "path", "duration", "sample_rate", "archive"]]
            idx = pd.MultiIndex.from_frame(kept[FILE_KEY])
            for col in ("split", "path", "duration", "sample_rate", "archive"):
                kept[col] = meta.loc[idx, col].to_numpy()
        print(
            f"pulses: reuse {kept.groupby(FILE_KEY).ngroups if len(kept) else 0} files, "
            f"todo {len(todo)}",
            flush=True,
        )
    else:
        todo = files_df
        kept = pd.DataFrame(columns=PULSE_COLUMNS[:-1])

    rows: list[dict] = []
    n = len(todo)
    skipped = 0
    for i, row in enumerate(todo.itertuples(index=False), 1):
        if i == 1 or i % 50 == 0 or i == n:
            print(f"pulses: {i}/{n}", flush=True)

        wav_path = Path(row.path) if Path(row.path).is_file() else (wav_dir / row.species / row.filename)
        if not wav_path.is_file():
            skipped += 1
            continue
        rows.extend(_pulse_rows_for_wav(wav_path, pd.Series(row._asdict())))

        if checkpoint is not None and (i % checkpoint_every == 0 or i == n) and rows:
            chunk = pd.DataFrame(rows, columns=PULSE_COLUMNS[:-1])
            merged = pd.concat([kept, chunk], ignore_index=True) if len(kept) else chunk
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(checkpoint, index=False)
            print(f"pulses: checkpoint → {checkpoint} ({len(merged)} rows)", flush=True)

    if skipped:
        print(f"pulses: skipped {skipped} files (WAV missing)", flush=True)

    new_df = pd.DataFrame(rows, columns=PULSE_COLUMNS[:-1]) if rows else pd.DataFrame(columns=PULSE_COLUMNS[:-1])
    out = pd.concat([kept, new_df], ignore_index=True) if len(kept) else new_df
    if out.empty:
        raise RuntimeError("no pulses after gottbat filter")
    if checkpoint is not None:
        out.to_csv(checkpoint, index=False)
    return out


def build_pulse_manifest_from_zip(files_df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    n = len(files_df)
    open_zips: dict[Path, zipfile.ZipFile] = {}

    def get_zip(archive: str) -> zipfile.ZipFile:
        zip_path = raw_dir / archive
        if zip_path not in open_zips:
            open_zips[zip_path] = zipfile.ZipFile(zip_path, "r")
        return open_zips[zip_path]

    try:
        for i, row in enumerate(files_df.itertuples(index=False), 1):
            if i == 1 or i % 100 == 0 or i == n:
                print(f"pulses(zip): {i}/{n}", flush=True)
            zf = get_zip(row.archive)
            member = resolve_member(zf, row.filename)
            if member is None:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(zf.read(member))
                tmp_path = Path(tmp.name)
            try:
                rows.extend(_pulse_rows_for_wav(tmp_path, pd.Series(row._asdict())))
            finally:
                tmp_path.unlink(missing_ok=True)
    finally:
        for zf in open_zips.values():
            zf.close()

    if not rows:
        raise RuntimeError("no pulses after gottbat filter")
    return pd.DataFrame(rows, columns=PULSE_COLUMNS[:-1])


def build_label_map(train_pulses: pd.DataFrame) -> dict[str, int]:
    species = sorted(train_pulses["species"].unique())
    return {name: i for i, name in enumerate(species)}


def load_existing_pulses(trainval_dir: Path, test_dir: Path) -> pd.DataFrame | None:
    paths = [
        trainval_dir / "pulses_train.csv",
        trainval_dir / "pulses_val.csv",
        test_dir / "pulses_test.csv",
    ]
    if not all(p.is_file() for p in paths):
        return None
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def precompute_split_cache(pulses_df: pd.DataFrame, cache_dir: Path, name: str) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"precompute {name} → {cache_dir}", flush=True)
    built = precompute_specs(pulses_df, desc=f"{name} specs", cache_dir=cache_dir)
    hits, total = cache_stats(pulses_df, cache_dir=cache_dir)
    return {"built": built, "cached": hits, "total": total, "dir": str(cache_dir)}


def print_summary(files_df: pd.DataFrame, pulses_df: pd.DataFrame) -> dict:
    summary = {
        "paper_reference": {
            "files": PAPER_FILES_TARGET,
            "train_pulses": PAPER_TRAIN_PULSES_TARGET,
            "classes": PAPER_CLASSES,
            "file_split": "80/10/10 (train/val/test)",
        },
        "excluded_species": sorted(EXCLUDED_PAPER),
        "limitations": [
            "round-robin по species×grid cell недоступен без NABat DB",
            "holdout test из статьи не в USGS release — test split локальный proxy",
            "корпус обновляемый: недостающие виды появятся после --update",
        ],
        "file_level": {
            "n_files": int(len(files_df)),
            "n_species": int(files_df["species"].nunique()),
            "n_train_files": int((files_df["split"] == "train").sum()),
            "n_val_files": int((files_df["split"] == "val").sum()),
            "n_test_files": int((files_df["split"] == "test").sum()),
            "files_per_species": files_df["species"].value_counts().to_dict(),
        },
        "pulse_level": {
            "filter": "gottbat spectrogram_v2 (§2.2.2)",
            "n_pulses": int(len(pulses_df)),
            "n_train": int((pulses_df["split"] == "train").sum()),
            "n_val": int((pulses_df["split"] == "val").sum()),
            "n_test": int((pulses_df["split"] == "test").sum()),
            "pulses_per_species": pulses_df.groupby("species").size().to_dict(),
        },
    }
    print("\n=== NABat paper-style dataset summary ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Build NABat paper-style dataset")
    p.add_argument("--raw", type=Path, default=ROOT / "data")
    p.add_argument("--wav-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "nabat_paper_31")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--limit-files", type=int, default=0)
    p.add_argument("--files-only", action="store_true")
    p.add_argument("--pulses-only", action="store_true")
    p.add_argument("--no-extract", action="store_true")
    p.add_argument("--from-zip", action="store_true")
    p.add_argument(
        "--scan-extracted",
        action="store_true",
        help="сканировать wav-dir/<SPECIES>/*.wav вместо zip",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="сохранить split старых файлов; досчитать только новые WAV/импульсы/кэш",
    )
    p.add_argument(
        "--precompute-cache",
        action="store_true",
        help="spec_cache (float16) в trainval/ и test/",
    )
    args = p.parse_args()

    trainval_dir = args.out / "trainval"
    test_dir = args.out / "test"
    trainval_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    files_train_path = trainval_dir / "files_train.csv"
    files_val_path = trainval_dir / "files_val.csv"
    files_test_path = test_dir / "files_test.csv"
    pulses_train_path = trainval_dir / "pulses_train.csv"
    pulses_val_path = trainval_dir / "pulses_val.csv"
    pulses_test_path = test_dir / "pulses_test.csv"
    summary_path = args.out / "summary.json"

    if args.pulses_only:
        files_df = pd.concat([
            pd.read_csv(files_train_path),
            pd.read_csv(files_val_path),
            pd.read_csv(files_test_path),
        ], ignore_index=True)
    else:
        print(
            f"scan: exclude {sorted(EXCLUDED_PAPER)}; "
            f"{'extracted ' + str(args.wav_dir) if args.scan_extracted else args.raw}",
            flush=True,
        )
        if args.scan_extracted:
            files_df = scan_extracted_wavs(args.wav_dir)
        else:
            files_df = scan_usgs_release(args.raw)
        if args.limit_files > 0:
            files_df = files_df.head(args.limit_files).copy()
        files_df = _abs_paths(files_df, args.wav_dir)

        old = load_existing_file_splits(trainval_dir, test_dir) if args.update else None
        if old is not None:
            files_df = merge_splits_update(files_df, old, args.seed)
        else:
            if args.update:
                print("update: no previous files_*.csv — full split", flush=True)
            files_df = assign_file_splits(files_df, args.seed)

        for split_name, path in (
            ("train", files_train_path),
            ("val", files_val_path),
            ("test", files_test_path),
        ):
            write_manifest(files_df[files_df["split"] == split_name][FILE_COLUMNS], path)

        print(
            f"files: {len(files_df)} ({files_df['species'].nunique()} species)  "
            f"train={int((files_df['split']=='train').sum())} "
            f"val={int((files_df['split']=='val').sum())} "
            f"test={int((files_df['split']=='test').sum())}  "
            f"(paper target: {PAPER_FILES_TARGET})",
            flush=True,
        )

    if args.files_only:
        print(f"saved files → {trainval_dir}, {test_dir}", flush=True)
        return

    if not args.pulses_only and not args.no_extract and not args.from_zip and not args.scan_extracted:
        print(f"extract WAV → {args.wav_dir}", flush=True)
        n_ok = extract_files(files_df, args.raw, args.wav_dir, raw_fallback=args.raw)
        print(f"extracted: {n_ok}/{len(files_df)}", flush=True)

    existing_pulses = load_existing_pulses(trainval_dir, test_dir) if args.update else None
    checkpoint = args.out / "pulses_partial.csv"
    if args.from_zip:
        pulses_df = build_pulse_manifest_from_zip(files_df, args.raw)
    else:
        pulses_df = build_pulse_manifest(
            files_df,
            args.wav_dir,
            existing=existing_pulses,
            checkpoint=checkpoint,
        )

    train_pulses = pulses_df[pulses_df["split"] == "train"]
    train_sp = sorted(train_pulses["species"].unique())
    rest = sorted(set(pulses_df["species"].unique()) - set(train_sp))
    label2id = {name: i for i, name in enumerate(train_sp + rest)}

    pulses_df = _attach_labels(pulses_df, label2id)
    pulses_df = pulses_df.sort_values(
        ["split", "species", "filename", "pulse_center"],
    ).reset_index(drop=True)

    for split_name, path in (
        ("train", pulses_train_path),
        ("val", pulses_val_path),
        ("test", pulses_test_path),
    ):
        write_manifest(pulses_df[pulses_df["split"] == split_name], path)

    summary = print_summary(files_df, pulses_df)
    summary["label2id"] = label2id
    import config as cfg
    summary["cache"] = {
        "version": cfg.SPEC_CACHE_VERSION,
        "dtype": getattr(cfg, "SPEC_CACHE_DTYPE", "float16"),
        "approx_bytes_per_pulse": 3 * 100 * 100 * 2,
    }

    if args.precompute_cache:
        summary["spec_cache"] = {
            "train": precompute_split_cache(
                pulses_df[pulses_df["split"] == "train"],
                trainval_dir / "spec_cache",
                "train",
            ),
            "val": precompute_split_cache(
                pulses_df[pulses_df["split"] == "val"],
                trainval_dir / "spec_cache",
                "val",
            ),
            "test": precompute_split_cache(
                pulses_df[pulses_df["split"] == "test"],
                test_dir / "spec_cache",
                "test",
            ),
        }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved trainval={trainval_dir}\n      test={test_dir}\n      {summary_path}", flush=True)


if __name__ == "__main__":
    main()
