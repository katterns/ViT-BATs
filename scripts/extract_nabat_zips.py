"""Распаковка NABat zip в data/<SPECIES>/; архив удаляется только если все WAV на месте."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROKEN = frozenset({"EUMA", "NYMA"})


def list_wav_members(zf: zipfile.ZipFile) -> list[str]:
    return [m for m in zf.namelist() if not m.endswith("/") and m.lower().endswith(".wav")]


def extract_zip(zip_path: Path, out_dir: Path) -> tuple[int, int, int, int]:
    """new, skipped, failed, total"""
    new = skipped = failed = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = list_wav_members(zf)
        total = len(members)
        for i, member in enumerate(members, 1):
            dest = out_dir / member.rsplit("/", 1)[-1]
            if dest.is_file():
                skipped += 1
                continue
            if i == 1 or i % 100 == 0 or i == total:
                print(f"  {zip_path.stem}: {i}/{total}", flush=True)
            try:
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                new += 1
            except Exception as exc:
                failed += 1
                if failed <= 3:
                    print(f"  skip {dest.name}: {exc}", flush=True)
                elif failed == 4:
                    print("  ... further read errors suppressed", flush=True)
    return new, skipped, failed, total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    zips = sorted(args.data.glob("*.zip"))
    if not zips:
        print(f"no zip in {args.data}", flush=True)
        return

    removed = kept = 0
    for zip_path in zips:
        species = zip_path.stem.upper()
        if species in BROKEN:
            print(f"skip broken download {zip_path.name}", flush=True)
            kept += 1
            continue

        out_dir = args.data / species
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = list_wav_members(zf)
        except zipfile.BadZipFile as exc:
            print(f"skip unreadable {zip_path.name}: {exc}", flush=True)
            kept += 1
            continue

        expected = {m.rsplit("/", 1)[-1] for m in members}
        existing = {p.name for p in out_dir.glob("*.wav")} if out_dir.is_dir() else set()
        missing = expected - existing

        print(
            f"{zip_path.name}: expected={len(expected)} "
            f"in_folder={len(existing & expected)} missing={len(missing)}",
            flush=True,
        )

        if args.dry_run:
            if not missing:
                print(f"  would remove {zip_path.name}", flush=True)
            else:
                print(f"  would extract {len(missing)}, keep archive", flush=True)
            continue

        if missing:
            out_dir.mkdir(parents=True, exist_ok=True)
            new, skipped, failed, total = extract_zip(zip_path, out_dir)
            present = len([n for n in expected if (out_dir / n).is_file()])
            print(
                f"  extracted new={new} failed={failed} "
                f"present={present}/{total}",
                flush=True,
            )
        else:
            present = len(expected)
            print(f"  folder complete ({present} wav)", flush=True)

        still_missing = [n for n in expected if not (out_dir / n).is_file()]
        if not still_missing:
            zip_path.unlink()
            print(f"removed {zip_path.name}", flush=True)
            removed += 1
        else:
            print(
                f"keep {zip_path.name} ({len(still_missing)} wav still missing — "
                f"перекачайте архив)",
                flush=True,
            )
            kept += 1

    if not args.dry_run:
        print(f"done: removed={removed} kept={kept}", flush=True)


if __name__ == "__main__":
    main()
