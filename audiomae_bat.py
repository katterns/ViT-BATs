import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIOMAE_ROOT = PROJECT_ROOT / "third_party" / "AudioMAE"

# нормализация fbank
AUDIOSET_NORM_MEAN = -4.2677393
AUDIOSET_NORM_STD = 4.5689974

IMG_SIZE_AUDIO = (1024, 128)  # (time_frames, mel_bins) = (H, W) для Conv2d


def ensure_audiomae_on_path() -> Path:
    if not AUDIOMAE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Репозиторий AudioMAE не найден: {AUDIOMAE_ROOT}\n"
            "Выполните: git clone --depth 1 https://github.com/facebookresearch/AudioMAE.git "
            "third_party/AudioMAE"
        )
    p = str(AUDIOMAE_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return AUDIOMAE_ROOT


def build_audio_mae_base(
    img_size: Tuple[int, int] = IMG_SIZE_AUDIO,
    decoder_mode: int = 0,
):
    """Собирает MaskedAutoencoderViT (ViT-B/16) в аудио-конфигурации, без загрузки весов."""
    ensure_audiomae_on_path()
    import models_mae

    model = models_mae.mae_vit_base_patch16(
        in_chans=1,
        audio_exp=True,
        img_size=img_size,
        decoder_mode=decoder_mode,
    )
    return model


def load_audio_mae_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = False,
) -> torch.nn.modules.module._IncompatibleKeys:
    """Загружает checkpoint['model'] из .pth официального AudioMAE."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Файл чекпоинта не найден: {path}\n"
            "Скачайте ViT-B pretrained (AudioSet-2M) по ссылке из third_party/AudioMAE/README.md "
            "и сохраните, например, в checkpoints/audiomae_pretrained.pth"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return model.load_state_dict(state, strict=strict)


def waveform_to_audiomae_input(
    waveform: torch.Tensor,
    sr: float,
    *,
    target_length: int = 1024,
    num_mel_bins: int = 128,
    resample_sr: Optional[float] = 16_000,
    norm_mean: float = AUDIOSET_NORM_MEAN,
    norm_std: float = AUDIOSET_NORM_STD,
) -> torch.Tensor:
    """
    Преобразует waveform в тензор для AudioMAE: [1, 1, H, W] = [1, 1, time, freq].
    """
    import torchaudio

    w = waveform.float()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    if resample_sr is not None and int(sr) != int(resample_sr):
        w = torchaudio.functional.resample(w, orig_freq=int(sr), new_freq=int(resample_sr))
        sr = float(resample_sr)
    w = w - w.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        w,
        htk_compat=True,
        sample_frequency=float(sr),
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_shift=10,
    )
    # [time, mel]
    if fbank.shape[0] < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - fbank.shape[0]))
    else:
        fbank = fbank[:target_length, :]
    fbank = (fbank - norm_mean) / (norm_std * 2)
    # [1, 1, 1024, 128]
    return fbank.unsqueeze(0).unsqueeze(0)


def encode_audiomae_encoder(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Полный проход энкодера: возвращает [B, 1+N_patches, D] после LayerNorm.
    Токен 0 — CLS, далее патчи.
    """
    x = model.patch_embed(x)
    x = x + model.pos_embed[:, 1:, :]
    cls_tok = model.cls_token + model.pos_embed[:, :1, :]
    cls_tok = cls_tok.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tok, x), dim=1)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x


def cls_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Глобальный вектор [B, D] - только CLS."""
    tok = encode_audiomae_encoder(model, x)
    return tok[:, 0, :]


class AudioMAEBatEncoder(nn.Module):
    """
    Обёртка: предобученный AudioMAE encoder + опциональная голова для N классов (fine-tune).
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.backbone = build_audio_mae_base()
        if checkpoint_path is not None:
            msg = load_audio_mae_checkpoint(self.backbone, checkpoint_path, strict=False)
            self._load_msg = msg
        else:
            self._load_msg = None
        d = self.backbone.embed_dim
        self.head = nn.Linear(d, num_classes) if num_classes is not None else None
        if freeze_encoder and checkpoint_path is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = cls_embedding(self.backbone, x)
        if self.head is not None:
            z = self.head(z)
        return z
