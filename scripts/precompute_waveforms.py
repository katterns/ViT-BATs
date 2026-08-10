import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from bat.data.paper_splits import load_paper_trainval
from bat.data.waveform_cache import CLIP_SAMPLES, build_waveform_cache


def main():
    parser = argparse.ArgumentParser(
        description="Build compact 50 ms waveform cache for physical sep mixtures",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=cfg.NABAT_PAPER_WAVEFORM_CACHE,
    )
    args = parser.parse_args()

    train_df, val_df, *_ = load_paper_trainval()
    frame = pd.concat([train_df, val_df], ignore_index=True)
    size_gib = len(frame) * CLIP_SAMPLES * 2 / 1024**3
    print(
        f"build waveform cache: {len(frame)} clips, {size_gib:.2f} GiB -> {args.cache_dir}",
        flush=True,
    )
    build_waveform_cache(frame, args.cache_dir)
    print("waveform cache complete", flush=True)


if __name__ == "__main__":
    main()
