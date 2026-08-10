import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from bat.data.audio import compute_base_spec, load_spec
from bat.data.nabat import CLIP_MS, IMG_SIZE, MIN_FREQ_HZ, MAX_FREQ_HZ
from bat.data.paper_splits import load_paper_trainval

OUT_DEFAULT = cfg.CHECKPOINT_DIR / "preprocessed_input_preview.png"


def _rgb_imshow(ax, spec, title):
    img = spec.numpy() if hasattr(spec, "numpy") else spec
    img = np.transpose(img, (1, 2, 0))
    ax.imshow(img, origin="lower", aspect="auto")
    ax.set_xlabel("время, px")
    ax.set_ylabel("частота, px")
    ax.set_title(title, fontsize=10)


def pick_examples(df, n=6, seed=42):
    rng = np.random.default_rng(seed)
    picks = []
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        picks.append(sub.iloc[rng.integers(len(sub))])
        if len(picks) >= n:
            break
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, default=OUT_DEFAULT)
    p.add_argument("--n-examples", type=int, default=6)
    args = p.parse_args()

    _, val_df, _, id2label, _ = load_paper_trainval()
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    cache = cfg.NABAT_PAPER_SPEC_CACHE

    row = val_df.iloc[0]
    path = row["path"]
    offset = int(row["pulse_center"])
    label = id2label[int(row["label"])]

    base = compute_base_spec(path, offset)
    val_spec = load_spec(
        path, training=False, rng=rng, spec_aug=False, pulse_center=offset, cache_dir=cache,
    )
    train_spec = load_spec(
        path, training=True, rng=rng, spec_aug=cfg.SUPERVISED_SPEC_AUG,
        pulse_center=offset, cache_dir=cache,
    )
    examples = pick_examples(val_df, n=args.n_examples)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"NABat v2 (gottbat)  •  tensor [3, {IMG_SIZE}, {IMG_SIZE}] float32\n"
        f"окно {CLIP_MS} ms  •  offset={offset} ms  •  "
        f"полоса {MIN_FREQ_HZ // 1000}–{MAX_FREQ_HZ // 1000} kHz",
        fontsize=11,
        y=0.995,
    )
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.6], hspace=0.4)

    mid = outer[0].subgridspec(1, 3, wspace=0.28)
    ax_base = fig.add_subplot(mid[0, 0])
    _rgb_imshow(ax_base, base, "base spec (кэш)")
    ax_val = fig.add_subplot(mid[0, 1])
    _rgb_imshow(ax_val, val_spec, f"validation  •  {Path(path).name}  •  {label}")
    ax_train = fig.add_subplot(mid[0, 2])
    _rgb_imshow(ax_train, train_spec, "train (+ spec aug если включён)")

    bottom = outer[1].subgridspec(2, 3, hspace=0.35, wspace=0.25)
    for i, ex in enumerate(examples):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(bottom[r, c])
        spec = load_spec(
            ex["path"], training=False, rng=rng,
            pulse_center=int(ex["pulse_center"]), cache_dir=cache,
        )
        lbl = id2label[int(ex["label"])]
        _rgb_imshow(ax, spec, lbl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
