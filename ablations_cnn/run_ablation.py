import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
import torch
from ablations_cnn.presets import ALL_PRESETS, REPORT_PATH, ft_ckpt_path, parse_preset, ssl_ckpt_path


def _run(script, preset, *, mixup=False, ssl_version=1):
    cmd = [
        sys.executable, str(ROOT / "ablations_cnn" / script),
        "--preset", preset.id,
        "--ssl-version", str(ssl_version),
    ]
    if mixup and script == "finetune.py":
        cmd.append("--mixup")
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def collect_results(ssl_version=1):
    rows = []
    for preset in ALL_PRESETS:
        ckpt = ft_ckpt_path(preset, ssl_version=ssl_version)
        row = {
            "preset": preset.id,
            "ssl_version": str(ssl_version),
            "ssl_ckpt": str(ssl_ckpt_path(preset, ssl_version)),
            "ft_ckpt": str(ckpt),
            "val_macro_f1": "",
            "epoch": "",
        }
        if ckpt.is_file():
            meta = torch.load(ckpt, map_location="cpu", weights_only=False)
            row["val_macro_f1"] = f"{float(meta['val_macro_f1']):.4f}"
            row["epoch"] = str(meta.get("epoch", ""))
        rows.append(row)
    return rows


def write_report(rows):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["preset", "ssl_version", "ssl_ckpt", "ft_ckpt", "val_macro_f1", "epoch"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"report: {REPORT_PATH}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("pretrain", "finetune", "all"), default="all")
    p.add_argument("--preset", default=None)
    p.add_argument(
        "--ssl-version", type=int, choices=(1, 2, 3), default=1,
        help="1: UNet; 2: bottleneck MAE (mae_v2); 3: bottleneck + waveform sep",
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mixup", action="store_true", help="MixUp при finetune (отдельные *_mixup_best.pt)")
    args = p.parse_args()

    if args.list:
        for i, preset in enumerate(ALL_PRESETS, 1):
            active = ", ".join(name for name in ("mae", "con", "sep", "jig") if preset[name])
            print(f"{i:2d}. {preset.id:20s}  [{active}]")
        return

    if args.collect:
        write_report(collect_results(args.ssl_version))
        return

    presets = [parse_preset(args.preset)] if args.preset else list(ALL_PRESETS)

    if args.dry_run:
        for preset in presets:
            ver = f" --ssl-version {args.ssl_version}" if args.ssl_version != 1 else ""
            if args.stage in ("pretrain", "all"):
                print(f"pretrain  --preset {preset.id}{ver}")
            if args.stage in ("finetune", "all"):
                suffix = " --mixup" if args.mixup else ""
                print(f"finetune  --preset {preset.id}{ver}{suffix}")
        return

    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for preset in presets:
        if args.stage in ("pretrain", "all"):
            _run("pretrain.py", preset, ssl_version=args.ssl_version)
        if args.stage in ("finetune", "all"):
            if args.stage == "finetune" and not ssl_ckpt_path(preset, args.ssl_version).is_file():
                raise FileNotFoundError(f"SSL checkpoint missing: {ssl_ckpt_path(preset, args.ssl_version)}")
            _run("finetune.py", preset, mixup=args.mixup, ssl_version=args.ssl_version)

    if args.stage in ("finetune", "all"):
        write_report(collect_results(args.ssl_version))

if __name__ == "__main__":
    main()
