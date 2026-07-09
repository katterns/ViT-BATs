"""Сборка локального корпуса cleaned/ из zip-архивов NABat ML training release."""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import soundfile as sf

# Исключены в USGS release (малый N): CORA, EUFL, LAXA, NYFE
EXCLUDED_USGS_LOW_N = frozenset({"CORA", "EUFL", "LAXA", "NYFE"})
# Дополнительно не вошли в локальный cleaned/
EXCLUDED_EXTRA = frozenset({"EUPE", "IDPH", "MYAU", "MYVE", "NYHU"})
EXCLUDED_SPECIES = EXCLUDED_USGS_LOW_N | EXCLUDED_EXTRA

MIN_DURATION_SEC = 0.52
MAX_DURATION_SEC = 8.09

CSV_COLUMNS = ["species", "filename", "duration", "sample_rate", "archive"]


def resolve_member(zf: zipfile.ZipFile, filename: str) -> str | None:
    if filename in zf.namelist():
        return filename
    matches = [
        n for n in zf.namelist()
        if not n.endswith("/") and n.rsplit("/", 1)[-1] == filename
    ]
    return matches[0] if len(matches) == 1 else None


def probe_wav_bytes(data: bytes) -> tuple[float, int]:
    info = sf.info(io.BytesIO(data))
    return float(info.duration), int(info.samplerate)


def scan_zip(zip_path: Path) -> list[dict]:
    species = zip_path.stem.upper()
    if species in EXCLUDED_SPECIES:
        return []

    archive = zip_path.name
    rows: list[dict] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("/") or not member.lower().endswith(".wav"):
                continue
            filename = member.rsplit("/", 1)[-1]
            try:
                duration, sample_rate = probe_wav_bytes(zf.read(member))
            except Exception as exc:
                print(f"skip {species}/{filename}: {exc}", flush=True)
                continue
            if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
                continue
            rows.append({
                "species": species,
                "filename": filename,
                "duration": duration,
                "sample_rate": sample_rate,
                "archive": archive,
            })
    return rows


def scan_raw(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    zips = sorted(raw_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"no *.zip in {raw_dir}")

    for i, zip_path in enumerate(zips, 1):
        species = zip_path.stem.upper()
        print(f"scan {i}/{len(zips)} {zip_path.name}", flush=True)
        rows.extend(scan_zip(zip_path))

    if not rows:
        raise RuntimeError("no WAV passed filters")

    return (
        pd.DataFrame(rows, columns=CSV_COLUMNS)
        .sort_values(["species", "filename"])
        .reset_index(drop=True)
    )


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def extract_files(
    df: pd.DataFrame,
    raw_dir: Path,
    out_dir: Path,
    *,
    raw_fallback: Path | None = None,
) -> int:
    """Распаковывает WAV в out_dir/<species>/; возвращает число готовых файлов."""
    n_ok = 0
    open_zips: dict[Path, zipfile.ZipFile] = {}

    def get_zip(path: Path) -> zipfile.ZipFile | None:
        if not path.is_file() or path.stat().st_size <= 64:
            return None
        if path not in open_zips:
            open_zips[path] = zipfile.ZipFile(path, "r")
        return open_zips[path]

    try:
        for i, row in enumerate(df.itertuples(index=False), 1):
            if i == 1 or i % 1000 == 0 or i == len(df):
                print(f"extract {i}/{len(df)}", flush=True)

            dest = out_dir / row.species / row.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file():
                n_ok += 1
                continue

            for base in (raw_dir, raw_fallback):
                if base is None:
                    continue
                zip_path = base / row.archive
                zf = get_zip(zip_path)
                if zf is None:
                    continue
                member = resolve_member(zf, row.filename)
                if member is None:
                    continue
                zf.extract(member, path=dest.parent)
                extracted = dest.parent / member
                if extracted != dest and extracted.is_file():
                    extracted.rename(dest)
                n_ok += 1
                break
            else:
                raise FileNotFoundError(f"{row.filename} not found for {row.species}")
    finally:
        for zf in open_zips.values():
            zf.close()

    return n_ok


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Build cleaned/ from NABat ML zip archives")
    p.add_argument(
        "--raw",
        type=Path,
        default=root / "data" / "nabat_raw",
        help="каталог со скачанными zip (ANPA.zip, COTO.zip, …)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "cleaned",
        help="выходной каталог cleaned/",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=root / "data" / "audio_metadata_cleaned.csv",
        help="индекс после фильтрации (хранится в git)",
    )
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="только пересобрать CSV из --raw, без распаковки WAV",
    )
    p.add_argument(
        "--from-manifest",
        action="store_true",
        help="не сканировать zip; взять готовый --manifest и распаковать в --out",
    )
    args = p.parse_args()

    if args.from_manifest:
        if not args.manifest.is_file():
            raise FileNotFoundError(args.manifest)
        df = pd.read_csv(args.manifest)
        print(f"manifest {args.manifest}: {len(df)} rows, {df['species'].nunique()} classes", flush=True)
    else:
        df = scan_raw(args.raw)
        write_manifest(df, args.manifest)
        write_manifest(df, args.out / "audio_metadata_cleaned.csv")
        print(
            f"manifest: {len(df)} files, {df['species'].nunique()} classes "
            f"(duration {MIN_DURATION_SEC}–{MAX_DURATION_SEC} s)",
            flush=True,
        )
        print(f"saved {args.manifest}", flush=True)

    if args.manifest_only:
        return

    n_ok = extract_files(df, args.raw, args.out, raw_fallback=args.out)
    write_manifest(df, args.out / "audio_metadata_cleaned.csv")
    print(f"{args.out}: {n_ok} wav files ready", flush=True)


if __name__ == "__main__":
    main()
