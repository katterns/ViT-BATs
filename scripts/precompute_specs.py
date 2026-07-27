import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from bat.data.paper_splits import load_paper_test, load_paper_trainval
from bat.data.spec_cache import cache_stats, sync_spec_cache


def main():
    p = argparse.ArgumentParser(
        description="Sync spec_cache to nabat_paper_31 pulses_*.csv (prune + build)",
    )
    p.add_argument("--limit", type=int, default=0, help="только первые N примеров (0 = все)")
    args = p.parse_args()

    train_df, val_df, *_ = load_paper_trainval()
    test_df = load_paper_test()
    if args.limit > 0:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(args.limit)
        test_df = test_df.head(args.limit)

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    trainval_cache = cfg.NABAT_PAPER_SPEC_CACHE
    test_cache = cfg.NABAT_PAPER_TEST_SPEC_CACHE

    print(
        f"sync cache: trainval={len(trainval_df)} (train={len(train_df)} val={len(val_df)}) "
        f"test={len(test_df)}",
        flush=True,
    )
    trainval_stats = sync_spec_cache(trainval_df, cache_dir=trainval_cache, desc="trainval")
    test_stats = sync_spec_cache(test_df, cache_dir=test_cache, desc="test")

    th, tt = cache_stats(train_df, cache_dir=trainval_cache)
    vh, vt = cache_stats(val_df, cache_dir=trainval_cache)
    xeh, xet = cache_stats(test_df, cache_dir=test_cache)
    print(
        f"done: trainval removed={trainval_stats['removed']} built={trainval_stats['built']}; "
        f"cached train={th}/{tt}, val={vh}/{vt}, test={xeh}/{xet}",
        flush=True,
    )


if __name__ == "__main__":
    main()
