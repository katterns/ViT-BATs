"""Картинка: что именно подаётся на вход модели после предобработки."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from bat.data import load_split
from bat.data.audio import (
    CLIP_SAMPLES,
    CLIP_SEC,
    MAX_FREQ,
    MIN_FREQ,
    TARGET_SR,
    _extract_clip,
    compute_base_spec,
    load_spec,
    read_wav,
)

OUT_DEFAULT = cfg.CHECKPOINT_DIR / "preprocessed_input_preview.png"


def _spec_imshow(ax, spec, title, *, show_cbar=False):
    """spec: [1, H, W] z-score normalized."""
    img = spec[0].numpy() if hasattr(spec, "numpy") else spec[0]
    ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, CLIP_SEC * 1000, MIN_FREQ / 1000, MAX_FREQ / 1000],
    )
    ax.set_xlabel("время, мс")
    ax.set_ylabel("частота, кГц")
    ax.set_title(title, fontsize=10)
    if show_cbar:
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04, label="z-score")


def _waveform_plot(ax, clip, title):
    t_ms = np.arange(len(clip)) / TARGET_SR * 1000
    ax.plot(t_ms, clip, color="#2c7fb8", linewidth=0.8)
    ax.axvline(CLIP_SEC * 500, color="#d95f0e", ls="--", lw=0.8, alpha=0.7, label="центр импульса")
    ax.set_xlim(0, CLIP_SEC * 1000)
    ax.set_xlabel("время, мс")
    ax.set_ylabel("амплитуда")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)


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

    _, val_df, _, id2label, _ = load_split(cfg.FT_METADATA_PATH, cfg.FT_DATA_DIR)
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    row = val_df.iloc[0]
    path = row["path"]
    center = int(row["pulse_center"])
    label = id2label[int(row["label"])]

    y = read_wav(path)
    clip = _extract_clip(y, center, training=False)
    base = compute_base_spec(path, center)
    val_spec = load_spec(path, training=False, rng=rng, spec_aug=False, pulse_center=center)
    train_spec = load_spec(path, training=True, rng=rng, spec_aug=cfg.SUPERVISED_SPEC_AUG, pulse_center=center)
    examples = pick_examples(val_df, n=args.n_examples)

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f"Предобработка входа модели  •  tensor [1, {base.shape[1]}, {base.shape[2]}] float32\n"
        f"clip {CLIP_SEC * 1000:.0f} ms @ {TARGET_SR // 1000} kHz  •  "
        f"STFT n_fft=2048 hop=512  •  частоты {MIN_FREQ // 1000}–{MAX_FREQ // 1000} kHz  •  z-score",
        fontsize=11,
        y=0.995,
    )
    outer = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.1, 1.6], hspace=0.45)

    ax_w = fig.add_subplot(outer[0])
    _waveform_plot(
        ax_w,
        clip,
        f"Шаг 1: клип {CLIP_SAMPLES} сэмплов ({CLIP_SEC * 1000:.0f} ms) вокруг pulse_center  •  "
        f"{Path(path).name}  •  {label}",
    )

    mid = outer[1].subgridspec(1, 3, wspace=0.28)
    ax_base = fig.add_subplot(mid[0, 0])
    _spec_imshow(ax_base, base, "Шаг 2a: log1p|STFT| → resize → z-score\n(кэшируется на диске)")
    ax_val = fig.add_subplot(mid[0, 1])
    _spec_imshow(ax_val, val_spec, "Шаг 2b: validation / inference\n(без аугментаций)")
    ax_train = fig.add_subplot(mid[0, 2])
    _spec_imshow(ax_train, train_spec, "Шаг 2c: supervised train\nSpecAugment + gain ±6 dB", show_cbar=True)

    bottom = outer[2].subgridspec(2, 3, hspace=0.35, wspace=0.25)
    for i, ex in enumerate(examples):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(bottom[r, c])
        spec = load_spec(ex["path"], training=False, rng=rng, pulse_center=int(ex["pulse_center"]))
        lbl = id2label[int(ex["label"])]
        _spec_imshow(ax, spec, lbl)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
