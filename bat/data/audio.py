import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy import signal

import config as cfg

TARGET_SR = 192_000
CLIP_SEC = 0.05
CLIP_SAMPLES = int(CLIP_SEC * TARGET_SR)

MIN_FREQ = 5_000
MAX_FREQ = 96_000
N_FFT = 2048
HOP_LENGTH = 512
SPEC_H = 128
SPEC_W = 256

PULSE_ENERGY_WIN_MS = 2.0
PULSE_THRESHOLD_RATIO = 0.15
PULSE_MIN_GAP_MS = 20.0

WAV_GAIN_JITTER_DB = 3.0
SPEC_TIME_MASK_MAX = 24
SPEC_FREQ_MASK_MAX = 16
SPEC_GAIN_JITTER_DB = 6.0


def read_wav(path):
    y, sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        old_t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
        new_t = np.linspace(0, len(y) / sr, int(len(y) * TARGET_SR / sr), endpoint=False)
        y = np.interp(new_t, old_t, y)
    return y.astype(np.float32)


def find_pulses(y):
    win = int(PULSE_ENERGY_WIN_MS * TARGET_SR / 1000)
    win = max(win, 1)
    energy = y ** 2
    smooth = np.convolve(energy, np.ones(win) / win, mode="same")
    threshold = smooth.max() * PULSE_THRESHOLD_RATIO
    min_distance = max(1, int(PULSE_MIN_GAP_MS * TARGET_SR / 1000))
    peaks, _ = signal.find_peaks(smooth, height=threshold, distance=min_distance)
    return peaks.tolist()


def load_spec(path, training, rng, spec_aug=False, pulse_center=None):
    y = read_wav(path)

    if pulse_center is not None:
        center = int(pulse_center)
    else:
        pulses = find_pulses(y)
        if len(pulses) == 0:
            center = int(np.argmax(y ** 2))
        elif training:
            center = pulses[rng.integers(len(pulses))]
        else:
            center = pulses[len(pulses) // 2]

    start = center - CLIP_SAMPLES // 2
    clip = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    src_start = max(start, 0)
    src_end = min(start + CLIP_SAMPLES, len(y))
    dst_start = src_start - start
    clip[dst_start : dst_start + (src_end - src_start)] = y[src_start:src_end]

    if training and WAV_GAIN_JITTER_DB > 0:
        db = rng.uniform(-WAV_GAIN_JITTER_DB, WAV_GAIN_JITTER_DB)
        clip = clip * (10 ** (db / 20))

    n_fft = min(N_FFT, len(clip))
    hop = min(HOP_LENGTH, max(1, n_fft // 2))
    freqs, _, stft = signal.stft(clip, fs=TARGET_SR, nperseg=n_fft, noverlap=n_fft - hop)
    spec = np.log1p(np.abs(stft))
    max_freq = min(MAX_FREQ, TARGET_SR / 2)
    spec = spec[(freqs >= MIN_FREQ) & (freqs <= max_freq)]

    x = torch.tensor(spec, dtype=torch.float32)[None, None]
    x = F.interpolate(x, size=(SPEC_H, SPEC_W), mode="bilinear", align_corners=False)[0]
    x = (x - x.mean()) / x.std().clamp_min(1e-6)

    if training and spec_aug:
        h, w = x.shape[1], x.shape[2]
        if SPEC_FREQ_MASK_MAX > 0:
            band = rng.integers(0, SPEC_FREQ_MASK_MAX + 1)
            if band > 0:
                f0 = rng.integers(0, max(1, h - band))
                x = x.clone()
                x[:, f0 : f0 + band, :] = 0
        if SPEC_TIME_MASK_MAX > 0:
            width = rng.integers(0, SPEC_TIME_MASK_MAX + 1)
            if width > 0:
                t0 = rng.integers(0, max(1, w - width))
                x = x.clone()
                x[:, :, t0 : t0 + width] = 0
        if SPEC_GAIN_JITTER_DB > 0:
            db = rng.uniform(-SPEC_GAIN_JITTER_DB, SPEC_GAIN_JITTER_DB)
            x = x * (10 ** (db / 20))

    return x


def split_spec_overlap(spec):
    """Левая и правая части одной spec с перекрытием по времени."""
    _, _, w = spec.shape
    view_w = max(1, int(w * cfg.CONTRASTIVE_VIEW_FRAC))

    left = spec[:, :, :view_w]
    right = spec[:, :, w - view_w :]

    size = (SPEC_H, SPEC_W)
    left = F.interpolate(left, size=size, mode="bilinear", align_corners=False)
    right = F.interpolate(right, size=size, mode="bilinear", align_corners=False)
    return left, right
