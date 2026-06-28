import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
BEATS_ROOT = PROJECT_ROOT / "third_party" / "beats"

FBANK_MEAN = 15.41663
FBANK_STD = 6.55582
N_FFT = 2048
HOP_LENGTH = 512
NUM_FBANK_BINS = 128


def ensure_beats_on_path():
    if not BEATS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Код BEATs не найден: {BEATS_ROOT}\n"
            "Скопируйте BEATs.py, backbone.py, modules.py из "
            "https://github.com/microsoft/unilm/tree/master/beats"
        )
    p = str(BEATS_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_beats_checkpoint(checkpoint_path, map_location="cpu"):
    ensure_beats_on_path()
    from BEATs import BEATs, BEATsConfig

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Файл чекпоинта не найден: {path}\n"
            "Скачайте BEATs_iter3.pt: python scripts/download_beats_pretrained.py"
        )
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    beats_cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(beats_cfg)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    return model, beats_cfg


def build_beats_random():
    ensure_beats_on_path()
    from BEATs import BEATs, BEATsConfig

    cfg = {
        "input_patch_size": 16,
        "embed_dim": 512,
        "encoder_layers": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "finetuned_model": False,
    }
    beats_cfg = BEATsConfig(cfg)
    return BEATs(beats_cfg), beats_cfg


def _filter_bank(magnitude, num_bins=NUM_FBANK_BINS):
    n_freq = magnitude.shape[0]
    band = max(1, n_freq // num_bins)
    chunks = []
    for i in range(num_bins):
        lo = i * band
        hi = n_freq if i == num_bins - 1 else min(n_freq, (i + 1) * band)
        if lo >= hi:
            chunks.append(torch.zeros(magnitude.shape[1], device=magnitude.device, dtype=magnitude.dtype))
        else:
            chunks.append(magnitude[lo:hi].mean(dim=0))
    return torch.stack(chunks, dim=-1)


def _pad_or_crop_time(fbank, target_frames):
    t, c = fbank.shape
    if t == target_frames:
        return fbank
    if t > target_frames:
        i0 = (t - target_frames) // 2
        return fbank[i0 : i0 + target_frames]
    out = torch.zeros(target_frames, c, dtype=fbank.dtype, device=fbank.device)
    i0 = (target_frames - t) // 2
    out[i0 : i0 + t] = fbank
    return out


def waveform_to_beats_fbank(
    waveform,
    sr,
    *,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    num_bins=NUM_FBANK_BINS,
    target_time_frames=256,
    fbank_mean=FBANK_MEAN,
    fbank_std=FBANK_STD,
):
    w = waveform.float()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    w = w - w.mean()
    n_fft = int(min(n_fft, w.shape[-1]))
    hop_length = int(min(hop_length, max(1, n_fft // 2)))
    window = torch.hann_window(n_fft, device=w.device, dtype=w.dtype)
    spec = torch.stft(
        w,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    fbank = _filter_bank(spec.abs().squeeze(0), num_bins=num_bins)
    fbank = (fbank - fbank_mean) / (2.0 * fbank_std)
    if target_time_frames is not None:
        fbank = _pad_or_crop_time(fbank, int(target_time_frames))
    return fbank.contiguous()


def _encode(model, fbank):
    if fbank.dim() == 2:
        fbank = fbank.unsqueeze(0)
    x = model.patch_embedding(fbank.unsqueeze(1))
    x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
    x = model.layer_norm(x)
    if model.post_extract_proj is not None:
        x = model.post_extract_proj(x)
    x = model.dropout_input(x)
    x, _ = model.encoder(x, padding_mask=None)
    return x.mean(dim=1)


class BEATsBatClassifier(nn.Module):
    def __init__(self, backbone, num_classes, encoder_dim=768, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes),
        )

    def forward(self, fbank):
        return self.head(_encode(self.backbone, fbank))
