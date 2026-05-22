import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
BEATS_ROOT = PROJECT_ROOT / "third_party" / "beats"

# нормализация fbank 
BEATS_Fbank_MEAN = 15.41663
BEATS_Fbank_STD = 6.55582

# STFT
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_NUM_FBANK_BINS = 128


def ensure_beats_on_path() -> Path:
    if not BEATS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Код BEATs не найден: {BEATS_ROOT}\n"
            "Скопируйте BEATs.py, backbone.py, modules.py из "
            "https://github.com/microsoft/unilm/tree/master/beats"
        )
    p = str(BEATS_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return BEATS_ROOT


def _default_beats_iter3_cfg() -> dict:
    """Конфиг BEATs_iter3"""
    return {
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


def build_beats_from_config(cfg: Optional[dict] = None):
    ensure_beats_on_path()
    from BEATs import BEATs, BEATsConfig  

    if cfg is None:
        cfg = _default_beats_iter3_cfg()
    beats_cfg = BEATsConfig(cfg)
    return BEATs(beats_cfg), beats_cfg


def load_beats_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
):
    """Загружает BEATs_iter3 (или другой) .pt: {'cfg', 'model'}."""
    ensure_beats_on_path()
    from BEATs import BEATs, BEATsConfig  

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Файл чекпоинта не найден: {path}\n"
            "Скачайте BEATs_iter3.pt: python scripts/download_beats_pretrained.py"
        )
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt.get("cfg")
    beats_cfg = BEATsConfig(cfg)
    model = BEATs(beats_cfg)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return model, beats_cfg


def uniform_filter_bank_energies(
    magnitude: torch.Tensor,
    num_bins: int = DEFAULT_NUM_FBANK_BINS,
) -> torch.Tensor:
    """
    Разбивает частотную ось magnitude-спектrogram на num_bins равных полос,
    усредняет по частоте -> [T, num_bins].
    """
    if magnitude.dim() == 3:
        # [B, F, T] 
        raise ValueError("Ожидается magnitude [F, T]")
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


def _pad_or_crop_time(fbank: torch.Tensor, target_frames: int) -> torch.Tensor:
    """fbank [T, C] -> [target_frames, C]."""
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
    waveform: torch.Tensor,
    sr: float,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    num_bins: int = DEFAULT_NUM_FBANK_BINS,
    target_time_frames: Optional[int] = 256,
    fbank_mean: float = BEATS_Fbank_MEAN,
    fbank_std: float = BEATS_Fbank_STD,
) -> torch.Tensor:
    """
    Waveform -> normalized uniform filter-bank [T, 128] для BEATs.
    """
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
    mag = spec.abs().squeeze(0)
    fbank = uniform_filter_bank_energies(mag, num_bins=num_bins)
    fbank = (fbank - fbank_mean) / (2.0 * fbank_std)
    if target_time_frames is not None:
        fbank = _pad_or_crop_time(fbank, int(target_time_frames))
    return fbank.contiguous()


def extract_features_from_fbank(
    model: nn.Module,
    fbank: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Проход BEATs encoder по готовому fbank [B, T, 128]
    return: [B, L, D] и padding_mask.
    """
    if fbank.dim() == 2:
        fbank = fbank.unsqueeze(0)
    if padding_mask is not None:
        padding_mask = model.forward_padding_mask(fbank, padding_mask)

    x = fbank.unsqueeze(1)
    features = model.patch_embedding(x)
    features = features.reshape(features.shape[0], features.shape[1], -1)
    features = features.transpose(1, 2)
    features = model.layer_norm(features)

    if padding_mask is not None:
        padding_mask = model.forward_padding_mask(features, padding_mask)

    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)

    x = model.dropout_input(features)
    x, _layer_results = model.encoder(x, padding_mask=padding_mask)
    return x, padding_mask


def beats_mean_embedding(model: nn.Module, fbank: torch.Tensor) -> torch.Tensor:
    """Mean-pool по времени/патчам -> [B, encoder_embed_dim]."""
    x, padding_mask = extract_features_from_fbank(model, fbank)
    if padding_mask is not None and padding_mask.any():
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        denom = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        return x.sum(dim=1) / denom
    return x.mean(dim=1)


class BEATsBatClassifier(nn.Module):
    """BEATs encoder + голова классификации."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        encoder_dim: int = 768,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes),
        )

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        z = beats_mean_embedding(self.backbone, fbank)
        return self.head(z)
