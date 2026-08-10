import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps

import config as cfg
from bat.data.nabat import CLIP_MS, IMG_CHANNELS, IMG_SIZE, make_spectrogram_chw, metadata_for_offset, process_file
from bat.data.spec_cache import load_base_spec, save_base_spec

SPEC_H = IMG_SIZE
SPEC_W = IMG_SIZE
SPEC_CHANNELS = IMG_CHANNELS
CLIP_SEC = CLIP_MS / 1000.0
WAVEFORM_SAMPLE_RATE = 192_000
_MAGMA_LUT = torch.from_numpy(
    colormaps["magma"](np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
)

SPEC_TIME_MASK_MAX = 10
SPEC_FREQ_MASK_MAX = 10
SPEC_GAIN_JITTER_DB = 6.0


def find_pulses(path):
    data = process_file(path)
    if data is None:
        return []
    return [m.offset for m in data.metadata]


def compute_base_spec(path, window_offset):
    data = process_file(path)
    if data is None:
        raise ValueError(f"cannot process {path}")
    meta = metadata_for_offset(data, int(window_offset))
    if meta is None:
        raise ValueError(f"no pulse offset={window_offset} in {path}")
    img = make_spectrogram_chw(meta.window, data.sample_rate)
    if img is None:
        raise ValueError(f"cannot render spectrogram for {path} offset={window_offset}")
    return img.astype(np.float32)


def precompute_specs(df, desc="specs", *, cache_dir=None):
    from bat.data.nabat import make_spectrogram_chw, metadata_for_offset, process_file
    from bat.data.spec_cache import load_base_spec, save_base_spec

    built = 0
    groups = list(df.groupby("path", sort=False))
    n_files = len(groups)
    for fi, (path, group) in enumerate(groups, 1):
        if fi == 1 or fi % 50 == 0 or fi == n_files:
            print(f"{desc}: files {fi}/{n_files} (built={built})", flush=True)

        centers = [int(c) for c in group["pulse_center"].tolist()]
        missing = [
            c for c in centers
            if load_base_spec(path, c, cache_dir=cache_dir) is None
        ]
        if not missing:
            continue

        data = process_file(path)
        if data is None:
            continue
        for offset in missing:
            meta = metadata_for_offset(data, offset)
            if meta is None:
                continue
            img = make_spectrogram_chw(meta.window, data.sample_rate)
            if img is None:
                continue
            save_base_spec(path, offset, img.astype(np.float32), cache_dir=cache_dir)
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


def load_spec(path, training, rng, spec_aug=False, pulse_center=None, cache_dir=None):
    if pulse_center is None:
        raise ValueError("pulse_center (gottbat window offset, ms) is required")

    if cfg.USE_SPEC_CACHE:
        base = load_base_spec(path, int(pulse_center), cache_dir=cache_dir)
        if base is None:
            cache_hint = cache_dir or cfg.NABAT_PAPER_SPEC_CACHE
            raise FileNotFoundError(
                f"spec cache miss: {path} pulse_center={pulse_center}. "
                f"Run: uv run python scripts/precompute_specs.py  (cache_dir={cache_hint})"
            )
        x = torch.from_numpy(base.copy())
        if training and spec_aug:
            x = _apply_spec_aug(x, rng)
        return x

    spec = compute_base_spec(path, int(pulse_center))
    x = torch.from_numpy(spec.copy())
    if training and spec_aug:
        x = _apply_spec_aug(x.clone(), rng)
    return x


def split_spec_overlap(spec):
    _, _, w = spec.shape
    view_w = max(1, int(w * cfg.CONTRASTIVE_VIEW_FRAC))

    left = spec[:, :, :view_w]
    right = spec[:, :, w - view_w :]

    size = (SPEC_H, SPEC_W)
    left = F.interpolate(left.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    right = F.interpolate(right.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return left, right


def mix_specs(s1, s2, gain_ratio=1.0):
    """v1: occlusion mix — в каждом пикселе берётся max(g·s1, s2)."""
    if not torch.is_tensor(gain_ratio):
        gain_ratio = torch.as_tensor(gain_ratio, device=s1.device, dtype=s1.dtype)
    while gain_ratio.ndim < s1.ndim:
        gain_ratio = gain_ratio.unsqueeze(-1)
    return torch.maximum((gain_ratio * s1).clamp(0.0, 1.0), s2)


def waveform_mix_to_rgb(waveforms, sample_rate=WAVEFORM_SAMPLE_RATE):
    """Render a batch of 50 ms waveform mixtures into NABat-like RGB tensors."""
    if waveforms.ndim != 2:
        raise ValueError(f"expected [batch, samples], got {tuple(waveforms.shape)}")
    if sample_rate != WAVEFORM_SAMPLE_RATE:
        raise ValueError(f"expected sample_rate={WAVEFORM_SAMPLE_RATE}, got {sample_rate}")

    n_fft = int(0.001 * sample_rate)
    hop_length = n_fft // 4
    window = torch.hamming_window(
        n_fft,
        periodic=True,
        device=waveforms.device,
        dtype=waveforms.dtype,
    )
    stft = torch.stft(
        waveforms,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power = stft.abs().square()
    db = 10.0 * torch.log10(power.clamp_min(1e-10))
    db_max = db.amax(dim=(-2, -1), keepdim=True)
    db = torch.maximum(db, db_max - 80.0)

    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(waveforms.device)
    high = min(100_000.0, sample_rate / 2.0 - 2_000.0)
    passband = (freqs > 5_000.0) & (freqs < high)
    db = db.masked_fill(~passband[None, :, None], -500.0)

    denoised = db - db.median(dim=2, keepdim=True).values
    denoised = denoised - denoised.median(dim=1, keepdim=True).values
    denoised = denoised.clamp_min(0.0)

    # specshow uses a linear 5–100 kHz axis; at 192 kHz the top 4 kHz are blank.
    display = denoised[:, freqs > 5_000.0]
    scale = display.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    normalized = display / scale
    lut = _MAGMA_LUT.to(device=waveforms.device, dtype=waveforms.dtype)
    position = normalized * (lut.shape[0] - 1)
    lower = position.floor().long()
    upper = position.ceil().long()
    weight = (position - lower).unsqueeze(-1)
    rgb = lut[lower] * (1.0 - weight) + lut[upper] * weight
    rgb = rgb.permute(0, 3, 1, 2)

    active_height = round(
        (sample_rate / 2.0 - 5_000.0) / (100_000.0 - 5_000.0) * SPEC_H
    )
    rgb = F.interpolate(
        rgb,
        size=(active_height, SPEC_W),
        mode="bilinear",
        align_corners=False,
    )
    rgb = torch.flip(rgb, dims=(2,))
    return F.pad(rgb, (0, 0, SPEC_H - active_height, 0), value=0.0)


def temporal_jigsaw(x, n_parts=4, *, disallow_identity=True):
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    b, c, h, w = x.shape
    if w % n_parts != 0:
        raise ValueError(f"width {w} not divisible by jigsaw parts {n_parts}")
    part_w = w // n_parts
    parts = x.reshape(b, c, h, n_parts, part_w)
    perms = []
    eye = torch.arange(n_parts, device=x.device)
    for _ in range(b):
        perm = torch.randperm(n_parts, device=x.device)
        if disallow_identity and n_parts > 1:
            while torch.equal(perm, eye):
                perm = torch.randperm(n_parts, device=x.device)
        perms.append(perm)
    perm = torch.stack(perms, dim=0)
    idx = perm[:, None, None, :, None].expand(b, c, h, n_parts, part_w)
    shuffled = torch.gather(parts, 3, idx).reshape(b, c, h, w)
    if single:
        return shuffled.squeeze(0), perm.squeeze(0)
    return shuffled, perm
