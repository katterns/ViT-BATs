import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from bat.data import load_paper_trainval
from bat.data.audio import (
    SPEC_CHANNELS,
    SPEC_H,
    SPEC_W,
    load_spec,
    split_spec_overlap,
)

LOG_ROOT = cfg.CHECKPOINT_DIR / "lightning_logs"
HISTOGRAM_OUT = cfg.CHECKPOINT_DIR / "nabat_v2_results_histogram.png"
SSL_INPUT_OUT = LOG_ROOT / "ssl_pretrain" / "ssl_contrastive_input_preview.png"

EPOCH_LINE = re.compile(
    r"epoch=\s*(\d+)\s+train_loss=([\d.]+|nan)\s+val_loss=([\d.]+|nan)\s+macro_f1=([\d.]+|nan)\s+acc=([\d.]+|nan)"
)


def pick_examples(val_df, n=6, seed=42):
    rng = np.random.default_rng(seed)
    picks = []
    for label in sorted(val_df["label"].unique()):
        sub = val_df[val_df["label"] == label]
        picks.append(sub.iloc[int(rng.integers(len(sub)))])
        if len(picks) >= n:
            break
    return picks


def parse_epoch_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        m = EPOCH_LINE.match(line.strip())
        if not m:
            continue
        epoch, train_loss, val_loss, macro_f1, acc = m.groups()
        rows.append({
            "epoch": int(epoch),
            "train_loss": float(train_loss) if train_loss != "nan" else float("nan"),
            "val_loss": float(val_loss) if val_loss != "nan" else float("nan"),
            "macro_f1": float(macro_f1) if macro_f1 != "nan" else float("nan"),
            "acc": float(acc) if acc != "nan" else float("nan"),
        })
    return rows


def best_by_metric(rows: list[dict], key: str = "macro_f1") -> dict | None:
    valid = [r for r in rows if not np.isnan(r[key])]
    if not valid:
        return None
    return max(valid, key=lambda r: r[key])


def latest_run_best(path: Path) -> dict | None:
    rows = parse_epoch_log(path)
    if not rows:
        return None
    last_restart = 0
    for i, r in enumerate(rows):
        if r["epoch"] == 0:
            last_restart = i
    return best_by_metric(rows[last_restart:])


def plot_histogram(results: list[tuple[str, dict, str]], out_path: Path):
    labels, values, epochs = [], [], []
    for label, best, _ in results:
        labels.append(label)
        values.append(best["macro_f1"])
        epochs.append(best["epoch"])

    colors = ["#8e5bb5", "#7e4fb0", "#bfbfbf", "#a879cf"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="none", width=0.72, alpha=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=17, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Macro-F1", fontsize=13)
    ax.set_ylim(0, 0.9)
    fig.suptitle(
        "Supervised ablations: NABat v2 preprocessing",
        fontsize=18,
        y=0.98,
    )
    ax.set_title(
        f"nabat_paper_31 • val split • seed={cfg.RANDOM_SEED} • metrics from lightning_logs",
        fontsize=11,
        color="0.35",
        pad=10,
    )
    ax.grid(axis="both", alpha=0.22)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    for bar, val, ep in zip(bars, values, epochs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.045,
            f"ep.{ep}",
            ha="center",
            va="top",
            fontsize=9,
            color="white" if val > 0.2 else "0.2",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _show_rgb(ax, spec, title):
    if hasattr(spec, "permute"):
        img = spec.permute(1, 2, 0).numpy()
    elif spec.ndim == 3 and spec.shape[0] in (1, 3):
        img = np.transpose(spec, (1, 2, 0))
    else:
        img = spec
    ax.imshow(np.clip(img, 0, 1), origin="lower", aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_ssl_input(val_df, id2label, out_path: Path, n_examples: int = 3):
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    picks = pick_examples(val_df, n=n_examples)

    fig, axes = plt.subplots(n_examples, 3, figsize=(9.5, 3.2 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    view_frac = cfg.CONTRASTIVE_VIEW_FRAC
    fig.suptitle(
        f"SSL / contrastive input (NABat v2)  •  tensor [{SPEC_CHANNELS}, {SPEC_H}, {SPEC_W}]\n"
        f"positive pair in training: left/right overlapping views ({view_frac:.0%} width), resized to {SPEC_H}x{SPEC_W}",
        fontsize=11,
        y=0.995,
    )

    for i, row in enumerate(picks):
        center = int(row["pulse_center"])
        label = id2label[int(row["label"])]
        fname = Path(row["path"]).name
        full = load_spec(
            row["path"], training=False, rng=rng, pulse_center=center,
            cache_dir=cfg.NABAT_PAPER_SPEC_CACHE,
        )
        left, right = split_spec_overlap(full)
        _show_rgb(axes[i, 0], full, f"{label} / {fname}\nfull spec")
        _show_rgb(axes[i, 1], left, f"left view ({view_frac:.0%})")
        _show_rgb(axes[i, 2], right, f"right view ({view_frac:.0%})")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-histogram", action="store_true")
    p.add_argument("--only-histogram", action="store_true")
    args = p.parse_args()

    if not args.skip_histogram:
        runs = [
            ("CNN", LOG_ROOT / "cnn_baseline" / "epoch_log.txt", "#2ca02c"),
            ("ResNet18", LOG_ROOT / "resnet_baseline" / "epoch_log.txt", "#1f77b4"),
            ("ViT no SSL", LOG_ROOT / "finetune1" / "epoch_log.txt", "#ff7f0e"),
            ("ViT + contrastive SSL", LOG_ROOT / "finetune" / "epoch_log.txt", "#d62728"),
        ]
        results = []
        for label, log_path, color in runs:
            if not log_path.is_file():
                raise FileNotFoundError(f"Нет лога: {log_path}")
            best = latest_run_best(log_path)
            if best is None:
                raise ValueError(f"В {log_path} нет macro-F1")
            print(f"{label}: macro_f1={best['macro_f1']:.4f} epoch={best['epoch']}", flush=True)
            results.append((label, best, color))

        plot_histogram(results, HISTOGRAM_OUT)
        print(f"saved: {HISTOGRAM_OUT}", flush=True)

    if args.only_histogram:
        return

    _, val_df, _, id2label, _ = load_paper_trainval()

    plot_ssl_input(val_df, id2label, SSL_INPUT_OUT)
    print(f"saved: {SSL_INPUT_OUT}", flush=True)


if __name__ == "__main__":
    main()
