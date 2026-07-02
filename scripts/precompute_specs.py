"""Один раз прогоняет log-STFT кэш для всех (path, pulse_center) в split."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from bat.data import load_split
from bat.data.audio import precompute_specs
from bat.data.spec_cache import SPEC_CACHE_DIR, cache_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", action="store_true", help="cleaned_subset_200 (supervised)")
    p.add_argument("--full", action="store_true", help="cleaned (SSL)")
    p.add_argument("--limit", type=int, default=0, help="только первые N примеров train/val (0 = все)")
    args = p.parse_args()

    if args.full:
        meta, data = cfg.METADATA_PATH, cfg.DATA_DIR
        name = "full cleaned"
    else:
        meta, data = cfg.FT_METADATA_PATH, cfg.FT_DATA_DIR
        name = "subset_200"

    train_df, val_df, *_ = load_split(meta, data)
    if args.limit > 0:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(args.limit)
    print(f"precompute {name}: train={len(train_df)} val={len(val_df)}", flush=True)
    print(f"cache dir: {SPEC_CACHE_DIR}", flush=True)
    print("оценка диска: ~130 KB на spec", flush=True)

    built_train = precompute_specs(train_df, desc="train specs")
    built_val = precompute_specs(val_df, desc="val specs")
    hits, total = cache_stats(train_df)
    vhits, vtotal = cache_stats(val_df)
    print(
        f"done: built train={built_train}, val={built_val}; "
        f"cached train={hits}/{total}, val={vhits}/{vtotal}",
        flush=True,
    )


if __name__ == "__main__":
    main()
