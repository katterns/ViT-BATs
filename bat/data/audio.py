import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal

import config as cfg
from bat.data.nabat import (
    CLIP_MS,
    IMG_CHANNELS,
    IMG_SIZE,
    extract_clip as nabat_extract_clip,
    load_wav_mono,
    passes_quality_filter,
    process_window,
)
from bat.data.spec_cache import load_base_spec, save_base_spec

# legacy pulse centers в splits.py считались при 192 kHz
LEGACY_PULSE_SR = 192_000

PULSE_ENERGY_WIN_MS = 2.0
PULSE_THRESHOLD_RATIO = 0.15
PULSE_MIN_GAP_MS = 20.0

SPEC_H = IMG_SIZE
SPEC_W = IMG_SIZE
SPEC_CHANNELS = IMG_CHANNELS
CLIP_SEC = CLIP_MS / 1000.0
TARGET_SR = LEGACY_PULSE_SR  # для совместимости с pulse_center в CSV/кэше
CLIP_SAMPLES = int(CLIP_SEC * LEGACY_PULSE_SR)

MIN_FREQ = 5_000
MAX_FREQ = 100_000

WAV_GAIN_JITTER_DB = 3.0
SPEC_TIME_MASK_MAX = 10
SPEC_FREQ_MASK_MAX = 10
SPEC_GAIN_JITTER_DB = 6.0


def read_wav(path):
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != LEGACY_PULSE_SR:
        old_t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
        new_t = np.linspace(0, len(y) / sr, int(len(y) * LEGACY_PULSE_SR / sr), endpoint=False)
        y = np.interp(new_t, old_t, y)
    return y.astype(np.float32)


def find_pulses(y):
    win = int(PULSE_ENERGY_WIN_MS * LEGACY_PULSE_SR / 1000)
    win = max(win, 1)
    energy = y ** 2
    smooth = np.convolve(energy, np.ones(win) / win, mode="same")
    threshold = smooth.max() * PULSE_THRESHOLD_RATIO
    min_distance = max(1, int(PULSE_MIN_GAP_MS * LEGACY_PULSE_SR / 1000))
    peaks, _ = signal.find_peaks(smooth, height=threshold, distance=min_distance)
    return peaks.tolist()


def _pulse_center_native(pulse_center: int, sr: int) -> int:
    t_sec = float(pulse_center) / LEGACY_PULSE_SR
    return int(round(t_sec * sr))


def filter_nabat_pulses(path, pulse_centers):
    """Оставляет только 50-ms окна, которые прошли NABat edge/SNR/amplitude checks."""
    if not cfg.NABAT_QUALITY_FILTER:
        return list(pulse_centers), 0

    sig, sr = load_wav_mono(path)
    kept = []
    for pulse_center in pulse_centers:
        center = _pulse_center_native(int(pulse_center), sr)
        clip = nabat_extract_clip(sig, sr, center)
        if passes_quality_filter(clip, sr):
            kept.append(int(pulse_center))
    return kept, len(pulse_centers) - len(kept)


def compute_base_spec(path, pulse_center):
    sig, sr = load_wav_mono(path)
    center = _pulse_center_native(int(pulse_center), sr)
    clip = nabat_extract_clip(sig, sr, center)
    img = process_window(clip, sr, quality_filter=cfg.NABAT_QUALITY_FILTER)
    if img is None:
        raise ValueError(f"NABat quality filter rejected {path} pulse_center={pulse_center}")
    return img.astype(np.float32)


def get_or_build_base_spec(path, pulse_center):
    cached = load_base_spec(path, pulse_center)
    if cached is not None:
        return cached, True
    spec = compute_base_spec(path, pulse_center)
    save_base_spec(path, pulse_center, spec)
    return spec, False


def precompute_specs(df, desc="specs"):
    n = len(df)
    built = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        if i == 1 or i % 500 == 0 or i == n:
            print(f"{desc}: {i}/{n}", flush=True)
        _, hit = get_or_build_base_spec(row.path, int(row.pulse_center))
        if not hit:
            built += 1
    return built


def _apply_spec_aug(x, rng):
    h, w = x.shape[1], x.shape[2]
    if SPEC_FREQ_MASK_MAX > 0:
        band = rng.integers(0, SPEC_FREQ_MASK_MAX + 1)
        if band > 0:
            f0 = rng.integers(0, max(1, h - band))
            x[:, f0 : f0 + band, :] = 0
    if SPEC_TIME_MASK_MAX > 0:
        width = rng.integers(0, SPEC_TIME_MASK_MAX + 1)
        if width > 0:
            t0 = rng.integers(0, max(1, w - width))
            x[:, :, t0 : t0 + width] = 0
    if SPEC_GAIN_JITTER_DB > 0:
        db = rng.uniform(-SPEC_GAIN_JITTER_DB, SPEC_GAIN_JITTER_DB)
        x = x * (10 ** (db / 20))
    return x


def load_spec(path, training, rng, spec_aug=False, pulse_center=None):
    if pulse_center is not None and cfg.USE_SPEC_CACHE:
        base, _ = get_or_build_base_spec(path, int(pulse_center))
        x = torch.from_numpy(base.copy())
        if training and spec_aug:
            x = _apply_spec_aug(x, rng)
        return x

    sig, sr = load_wav_mono(path)
    if pulse_center is not None:
        center = _pulse_center_native(int(pulse_center), sr)
    else:
        y = read_wav(path)
        pulses = find_pulses(y)
        if len(pulses) == 0:
            center = _pulse_center_native(int(np.argmax(y ** 2)), sr)
        elif training:
            center = _pulse_center_native(pulses[rng.integers(len(pulses))], sr)
        else:
            center = _pulse_center_native(pulses[len(pulses) // 2], sr)

    clip = nabat_extract_clip(sig, sr, center)

    img = process_window(clip, sr, quality_filter=cfg.NABAT_QUALITY_FILTER)
    if img is None:
        raise ValueError(f"NABat quality filter rejected {path} pulse_center={pulse_center}")
    x = torch.from_numpy(img.copy())
    if training and spec_aug:
        x = _apply_spec_aug(x.clone(), rng)
    return x


def split_spec_overlap(spec):
    """Левая и правая части одной spec с перекрытием по времени."""
    _, _, w = spec.shape
    view_w = max(1, int(w * cfg.CONTRASTIVE_VIEW_FRAC))

    left = spec[:, :, :view_w]
    right = spec[:, :, w - view_w :]

    size = (SPEC_H, SPEC_W)
    left = F.interpolate(left.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    right = F.interpolate(right.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return left, right
