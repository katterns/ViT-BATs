import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

CAP_DEFAULT = 200
RNG = 42


def build_subset(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    parts = []
    for _, sub in df.groupby("species", sort=True):
        parts.append(sub.sample(min(len(sub), cap), random_state=RNG) if len(sub) > cap else sub)
    return pd.concat(parts, ignore_index=True).sort_values(["species", "filename"]).reset_index(drop=True)


def resolve_member(zf: zipfile.ZipFile, fn: str) -> str | None:
    if fn in zf.namelist():
        return fn
    matches = [n for n in zf.namelist() if not n.endswith("/") and n.rsplit("/", 1)[-1] == fn]
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cleaned", type=Path, default=Path(__file__).resolve().parents[1] / "cleaned")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "cleaned_subset_200")
    p.add_argument("--cap", type=int, default=CAP_DEFAULT)
    args = p.parse_args()

    subset = build_subset(pd.read_csv(args.cleaned / "audio_metadata_cleaned.csv"), args.cap)
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "audio_metadata_cleaned.csv"
    subset.to_csv(out_csv, index=False)

    n_ok = 0
    for species in sorted(subset["species"].unique()):
        grp = subset[subset["species"] == species]
        zip_path = args.cleaned / str(grp["archive"].iloc[0])
        zf = zipfile.ZipFile(zip_path, "r") if zip_path.is_file() and zip_path.stat().st_size > 64 else None
        try:
            for _, row in grp.iterrows():
                dest = args.out / species / row["filename"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_file():
                    n_ok += 1
                    continue
                disk_src = args.cleaned / species / row["filename"]
                if disk_src.is_file():
                    shutil.copy2(disk_src, dest)
                elif zf is None:
                    raise FileNotFoundError(disk_src)
                else:
                    member = resolve_member(zf, row["filename"])
                    if member is None:
                        raise KeyError(f"{row['filename']} not in {zip_path}")
                    zf.extract(member, path=dest.parent)
                n_ok += 1
        finally:
            if zf is not None:
                zf.close()

    print(f"{out_csv}: {n_ok} files, {subset['species'].nunique()} classes")


if __name__ == "__main__":
    main()
