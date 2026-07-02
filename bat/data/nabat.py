"""NABat ML v2 preprocessing — порт spectrogram/spectrogram_v2.py из gottbat."""

from __future__ import annotations

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CLIP_MS = 50
MIN_FREQ_HZ = 5_000
MAX_FREQ_HZ = 100_000
SN_THRESH = 7.0
AMP_THRESH = 21.0
IMG_SIZE = 100
IMG_CHANNELS = 3
BAND_REJECT_DB = -500.0

_fig = None
_ax = None


def _get_figure():
    global _fig, _ax
    if _fig is None:
        # dpi=100 → ровно 100 px; на Retina buffer_rgba может быть 200×200 — ресайз ниже
        _fig = plt.figure(figsize=(1, 1), facecolor="black", dpi=100)
        _ax = _fig.add_axes([0, 0, 1, 1], facecolor="black")
        plt.margins(0)
    return _fig, _ax


def denoise_spec(spec: np.ndarray) -> np.ndarray:
    spec = spec - np.median(spec, axis=1, keepdims=True)
    spec = spec - np.median(spec, axis=0, keepdims=True)
    spec.clip(min=0, out=spec)
    return spec


def _band_limit(spec: np.ndarray, frequency_bands: np.ndarray, sr: float) -> np.ndarray:
    hi = min(MAX_FREQ_HZ, (sr / 2) - 2000)
    out = spec.copy()
    for i, b in enumerate(frequency_bands):
        if b <= MIN_FREQ_HZ or b >= hi:
            out[i] = BAND_REJECT_DB
    return out


def stft_power_db(sig: np.ndarray, sr: float):
    root_size = int(0.001 * sr)
    hop_length = int(root_size / 4)
    stft = librosa.stft(
        sig,
        n_fft=root_size,
        hop_length=hop_length,
        win_length=root_size,
        window="hamming",
    )
    spec = librosa.power_to_db(np.abs(stft) ** 2)
    frequency_bands = librosa.fft_frequencies(sr=sr, n_fft=root_size)
    spec = _band_limit(spec, frequency_bands, sr)
    return spec, hop_length, frequency_bands


def _peak_metrics(spec: np.ndarray, frequency_bands: np.ndarray, window_ms: float = CLIP_MS):
    index = np.unravel_index(spec.argmax(), spec.shape)
    time_index = index[1]
    frequency_index = index[0]
    peak_frequency = float(frequency_bands[frequency_index])
    peak_time = time_index / 4.0
    return peak_time, peak_frequency, time_index, frequency_index


def _passes_edge_and_freq_checks(peak_time: float, peak_frequency: float, sr: float) -> bool:
    if peak_time < CLIP_MS * 0.2 or peak_time > CLIP_MS * 0.8:
        return False
    hi = min(MAX_FREQ_HZ, (sr / 2) - 2000)
    return not (peak_frequency <= MIN_FREQ_HZ or peak_frequency >= hi)


def _passes_quality_checks(
    spec: np.ndarray,
    time_index: int,
    frequency_index: int,
) -> tuple[bool, float, float]:
    freq_amp = spec[frequency_index]
    r_other = np.sum(spec) / (spec.shape[0] * spec.shape[1])
    t0 = max(0, time_index - 4)
    t1 = min(len(freq_amp), time_index + 6)
    rsig = float(np.sum(freq_amp[t0:t1]) / max(1, t1 - t0))
    snr = rsig / r_other if r_other else 0.0
    amplitude = float(freq_amp[time_index])
    ok = snr >= SN_THRESH and amplitude >= AMP_THRESH
    return ok, snr, amplitude


def quality_metrics(sig: np.ndarray, sr: float) -> dict[str, float | bool]:
    spec, _, frequency_bands = stft_power_db(sig, sr)
    peak_time, peak_frequency, time_index, frequency_index = _peak_metrics(spec, frequency_bands)
    edge_ok = _passes_edge_and_freq_checks(peak_time, peak_frequency, sr)

    spec = denoise_spec(spec)
    quality_ok, snr, amplitude = _passes_quality_checks(spec, time_index, frequency_index)
    return {
        "ok": edge_ok and quality_ok,
        "edge_ok": edge_ok,
        "quality_ok": quality_ok,
        "peak_time": peak_time,
        "peak_frequency": peak_frequency,
        "snr": snr,
        "amplitude": amplitude,
    }


def passes_quality_filter(sig: np.ndarray, sr: float) -> bool:
    return bool(quality_metrics(sig, sr)["ok"])


def render_spectrogram_image(
    spec: np.ndarray,
    sr: float,
    hop_length: int,
    low: float = MIN_FREQ_HZ,
    high: float = MAX_FREQ_HZ,
) -> np.ndarray:
    """RGB float32 [3, H, W]"""
    fig, ax = _get_figure()
    ax.clear()
    librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=hop_length,
        x_axis="s",
        y_axis="linear",
        ax=ax,
    )
    ax.set_ylim(low, high)
    ax.axis("off")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[..., :3]
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = np.asarray(
            Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        )
    img = rgb.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def extract_clip(sig: np.ndarray, sr: int, center_sample: int) -> np.ndarray:
    n = int(CLIP_MS * sr / 1000)
    start = int(center_sample) - n // 2
    clip = np.zeros(n, dtype=np.float32)
    src_start = max(start, 0)
    src_end = min(start + n, len(sig))
    dst_start = src_start - start
    clip[dst_start : dst_start + (src_end - src_start)] = sig[src_start:src_end]
    return clip


def process_window(
    sig: np.ndarray,
    sr: float,
    *,
    quality_filter: bool = False,
) -> np.ndarray | None:
    """NABat spectrogram [3, 100, 100]"""
    spec, hop_length, frequency_bands = stft_power_db(sig, sr)

    peak_time, peak_frequency, time_index, frequency_index = _peak_metrics(spec, frequency_bands)
    if quality_filter and not _passes_edge_and_freq_checks(peak_time, peak_frequency, sr):
        return None

    spec = denoise_spec(spec)
    ok, _, _ = _passes_quality_checks(spec, time_index, frequency_index)
    if quality_filter and not ok:
        return None

    return render_spectrogram_image(spec, sr, hop_length)


def load_wav_mono(path) -> tuple[np.ndarray, int]:
    sig, sr = librosa.load(str(path), sr=None, mono=True)
    return sig.astype(np.float32), int(sr)
