import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from bat.data.audio import precompute_specs
from bat.data.paper_splits import load_paper_test, load_paper_trainval
from bat.data.spec_cache import cache_stats


def main():
    p = argparse.ArgumentParser(description="Precompute float16 spec_cache for nabat_paper_31")
    p.add_argument("--limit", type=int, default=0, help="только первые N примеров (0 = все)")
    args = p.parse_args()

    train_df, val_df, *_ = load_paper_trainval()
    test_df = load_paper_test()
    if args.limit > 0:
        train_df = train_df.head(args.limit)
        val_df = val_df.head(args.limit)
        test_df = test_df.head(args.limit)

    trainval_cache = cfg.NABAT_PAPER_SPEC_CACHE
    test_cache = cfg.NABAT_PAPER_TEST_SPEC_CACHE
    print(
        f"precompute paper: train={len(train_df)} val={len(val_df)} test={len(test_df)}",
        flush=True,
    )
    built_train = precompute_specs(train_df, desc="train specs", cache_dir=trainval_cache)
    built_val = precompute_specs(val_df, desc="val specs", cache_dir=trainval_cache)
    built_test = precompute_specs(test_df, desc="test specs", cache_dir=test_cache)
    th, tt = cache_stats(train_df, cache_dir=trainval_cache)
    vh, vt = cache_stats(val_df, cache_dir=trainval_cache)
    xeh, xet = cache_stats(test_df, cache_dir=test_cache)
    print(
        f"done: built train={built_train}, val={built_val}, test={built_test}; "
        f"cached train={th}/{tt}, val={vh}/{vt}, test={xeh}/{xet}",
        flush=True,
    )


if __name__ == "__main__":
    main()
