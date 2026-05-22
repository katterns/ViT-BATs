import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal
from sklearn.model_selection import train_test_split

TARGET_SR = 192_000
CLIP_SEC = 0.5
SPEC_H, SPEC_W = 128, 256
N_FFT = 2048
HOP_LENGTH = 512
MAX_FREQ = 96_000
RANDOM_SEED = 42


def resample_audio(y: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return y.astype(np.float32)
    t_old = np.linspace(0.0, len(y) / orig_sr, num=len(y), endpoint=False)
    t_new = np.linspace(0.0, len(y) / orig_sr, num=int(len(y) * target_sr / orig_sr), endpoint=False)
    return np.interp(t_new, t_old, y).astype(np.float32)


def center_crop_or_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        i0 = (len(y) - target_len) // 2
        return y[i0 : i0 + target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[(target_len - len(y)) // 2 : (target_len - len(y)) // 2 + len(y)] = y
    return out


def make_log_stft(audio: np.ndarray, sr: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio[np.isfinite(audio)]
    if len(audio) < 8:
        raise ValueError("short")
    nperseg = int(min(N_FFT, len(audio)))
    hop = int(min(HOP_LENGTH, max(1, nperseg // 2)))
    noverlap = max(0, min(nperseg - 1, nperseg - hop))
    freqs, _, zxx = signal.stft(
        audio, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary="zeros", padded=True
    )
    return np.log1p(np.abs(zxx))[freqs <= min(MAX_FREQ, sr / 2)]


def tensor_from_wav(y: np.ndarray, sr: float):
    if y.ndim > 1:
        y = y.mean(axis=-1)
    y = center_crop_or_pad(resample_audio(y, float(sr), TARGET_SR), int(CLIP_SEC * TARGET_SR))
    x = torch.tensor(make_log_stft(y, TARGET_SR), dtype=torch.float32)[None, None]
    x = F.interpolate(x, size=(SPEC_H, SPEC_W), mode="bilinear", align_corners=False)
    return x, float(np.sqrt(np.mean(y**2)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "cleaned_subset_200")
    p.add_argument("--max-files", type=int, default=0)
    args = p.parse_args()

    meta = args.data_dir / "audio_metadata_cleaned.csv"
    if not meta.is_file():
        print("missing:", meta, file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(meta)
    df["path"] = df.apply(lambda r: args.data_dir / r["species"] / r["filename"], axis=1)
    df["label"] = df["species"].astype("category").cat.codes
    missing = df[~df["path"].map(Path.is_file)]
    work = df.sample(min(args.max_files, len(df)), random_state=42) if args.max_files else df

    bad_read, bad_pipe, rms = [], [], []
    for _, row in work.iterrows():
        if not row["path"].is_file():
            continue
        try:
            y, sr = sf.read(str(row["path"]), always_2d=False)
            if not np.all(np.isfinite(y)):
                raise ValueError("non-finite")
            ten, r = tensor_from_wav(y, sr)
            if not torch.isfinite(ten).all():
                raise ValueError("non-finite tensor")
            rms.append(r)
        except (OSError, RuntimeError) as e:
            bad_read.append((row["path"].name, str(e)))
        except Exception as e:
            bad_pipe.append((row["path"].name, str(e)))

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=RANDOM_SEED, stratify=df["label"])
    vc = val_df.groupby("species").size()

    print(f"csv={len(df)} missing={len(missing)} ok={len(rms)} read_err={len(bad_read)} pipe_err={len(bad_pipe)}")
    if rms:
        rms = np.array(rms)
        print(f"rms median={np.median(rms):.2e} silent={np.mean(rms < 1e-5):.1%}")
    print(f"split train={len(train_df)} val={len(val_df)} val_min_per_class={int(vc.min())}")

    if bad_read or bad_pipe:
        for name, err in (bad_read + bad_pipe)[:5]:
            print(f"  {name}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
