from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ablations_cnn.presets import TaskSet
from bat.data.audio import SPEC_CHANNELS, temporal_jigsaw
from vit_bat import block_mask, cosine_loss, nt_xent_loss, patch_energy, patchify

PATCH_SIZE = (10, 10)
ENC_DIM = 256
PROJ_DIM = 128
NOISE_PATCH_PERCENTILE = 25.0
CONTRASTIVE_TEMP = 0.07


def _gn(ch):
    g = 8
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


def _enc_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        _gn(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BatCNNEncoder(nn.Module):
    def __init__(self, in_ch=SPEC_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            _enc_block(in_ch, 32),
            _enc_block(32, 64),
            _enc_block(64, 128),
            _enc_block(128, 256),
            _enc_block(256, 256),
        )

    def forward_stages(self, x):
        stages = []
        for block in self.net:
            x = block(x)
            stages.append(x)
        return stages

    def feature_map(self, x):
        for block in self.net:
            x = block(x)
        return x

    def embed(self, x):
        return F.adaptive_avg_pool2d(self.feature_map(x), 1).flatten(1)


class UNetDecoder(nn.Module):
    """v1: upsample from 3×3 bottleneck with skip connections (50→25→12→6→3)."""

    def __init__(self):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec4 = nn.Sequential(nn.Conv2d(128 + 256, 128, 3, padding=1, bias=False), _gn(128), nn.ReLU(True))
        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec3 = nn.Sequential(nn.Conv2d(64 + 128, 64, 3, padding=1, bias=False), _gn(64), nn.ReLU(True))
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(32 + 64, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(True))
        self.up1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 32, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(True))
        self.up0 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.dec0 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(True))
        self.recon_head = nn.Conv2d(32, SPEC_CHANNELS, kernel_size=1)
        self.sep_head1 = nn.Conv2d(32, SPEC_CHANNELS, kernel_size=1)
        self.sep_head2 = nn.Conv2d(32, SPEC_CHANNELS, kernel_size=1)

    @staticmethod
    def _match(x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _decode(self, stages, out_size):
        s0, s1, s2, s3, x = stages
        x = self.dec4(torch.cat([self._match(self.up4(x), s3), s3], dim=1))
        x = self.dec3(torch.cat([self._match(self.up3(x), s2), s2], dim=1))
        x = self.dec2(torch.cat([self._match(self.up2(x), s1), s1], dim=1))
        x = self.dec1(torch.cat([self._match(self.up1(x), s0), s0], dim=1))
        x = self.dec0(self.up0(x))
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

    def recon(self, stages, out_size):
        return torch.sigmoid(self.recon_head(self._decode(stages, out_size)))

    def separate(self, stages, out_size):
        h = self._decode(stages, out_size)
        return torch.sigmoid(self.sep_head1(h)), torch.sigmoid(self.sep_head2(h))


class BottleneckDecoder(nn.Module):
    """v2: MAE-style decoder — только bottleneck, без skip-связей."""

    def __init__(self):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec4 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False), _gn(128), nn.ReLU(True))
        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), _gn(64), nn.ReLU(True))
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(True))
        self.up1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(True))
        self.up0 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.dec0 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), _gn(32), nn.ReLU(True))
        self.recon_head = nn.Conv2d(32, SPEC_CHANNELS, kernel_size=1)
        self.sep_head1 = nn.Conv2d(32, SPEC_CHANNELS, kernel_size=1)
        self.sep_head2 = nn.Conv2d(32, SPEC_CHANNELS, kernel_size=1)

    def _decode(self, feat, out_size):
        x = self.dec4(self.up4(feat))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        x = self.dec0(self.up0(x))
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

    def recon(self, feat, out_size):
        return torch.sigmoid(self.recon_head(self._decode(feat, out_size)))

    def separate(self, feat, out_size):
        h = self._decode(feat, out_size)
        return torch.sigmoid(self.sep_head1(h)), torch.sigmoid(self.sep_head2(h))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=ENC_DIM, hidden=256, out_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def apply_patch_mask(x, mask, patch_size=PATCH_SIZE):
    pf, pt = patch_size
    b, _, h, w = x.shape
    gf, gt = h // pf, w // pt
    mask_2d = mask.view(b, gf, gt)
    spatial = mask_2d.repeat_interleave(pf, dim=1).repeat_interleave(pt, dim=2)
    return x * (1.0 - spatial.unsqueeze(1))


def patch_mse(pred, target, patch_size=PATCH_SIZE):
    return ((patchify(pred, patch_size) - patchify(target, patch_size)) ** 2).mean(dim=-1)


class BatCNNSSL(nn.Module):
    def __init__(self, tasks: TaskSet, spec_h=100, spec_w=100, ssl_version=1):
        super().__init__()
        if ssl_version not in (1, 2, 3):
            raise ValueError(f"ssl_version must be 1, 2 or 3, got {ssl_version}")
        self.tasks = tasks
        self.ssl_version = int(ssl_version)
        self.spec_hw = (spec_h, spec_w)
        self.patch_size = PATCH_SIZE
        self.grid_shape = (spec_h // PATCH_SIZE[0], spec_w // PATCH_SIZE[1])
        self.encoder = BatCNNEncoder()

        if tasks.mae or tasks.sep or tasks.jig:
            self.decoder = BottleneckDecoder() if self.ssl_version >= 2 else UNetDecoder()
        if tasks.con:
            self.projection = ProjectionHead()

    def encoder_state_dict(self):
        return self.encoder.state_dict()

    def _decode_input(self, x):
        if self.ssl_version >= 2:
            return self.encoder.feature_map(x)
        return self.encoder.forward_stages(x)

    def _bottleneck(self, encoded):
        return encoded if self.ssl_version >= 2 else encoded[-1]

    def _signal_patch_mask(self, x, mask_ratio, noise_percentile):
        targets = patchify(x, self.patch_size)
        energies = patch_energy(x, self.patch_size)
        _, mask, _, _ = block_mask(
            targets, mask_ratio, self.grid_shape, energies, noise_percentile=noise_percentile,
        )
        return mask

    def mae_loss(self, x, mask_ratio, noise_percentile, utterance_weight):
        mask = self._signal_patch_mask(x, mask_ratio, noise_percentile)
        encoded = self._decode_input(apply_patch_mask(x, mask, self.patch_size))
        pred = self.decoder.recon(encoded, self.spec_hw)
        per_patch = patch_mse(pred, x, self.patch_size)
        recon = (per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        sem = {"recon": recon.detach()}
        total = recon
        if utterance_weight > 0:
            feat = self._bottleneck(encoded)
            masked_emb = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            utt = cosine_loss(masked_emb, self.encoder.embed(x).detach())
            sem["utterance"] = utt.detach()
            total = total + utterance_weight * utt
        return total, sem

    def contrastive_loss(self, x1, x2, temperature=CONTRASTIVE_TEMP):
        z = self.projection(self.encoder.embed(torch.cat([x1, x2], dim=0)))
        z1, z2 = z.chunk(2)
        return nt_xent_loss(z1, z2, temperature)

    def separation_loss(self, mix, s1, s2):
        encoded = self._decode_input(mix)
        p1, p2 = self.decoder.separate(encoded, self.spec_hw)
        loss_a = F.mse_loss(p1, s1) + F.mse_loss(p2, s2)
        loss_b = F.mse_loss(p1, s2) + F.mse_loss(p2, s1)
        return torch.minimum(loss_a, loss_b)

    def jigsaw_loss(self, x, n_parts):
        shuffled, _ = temporal_jigsaw(x, n_parts=n_parts)
        pred = self.decoder.recon(self._decode_input(shuffled), self.spec_hw)
        return F.mse_loss(pred, x)


class BatCNNClassifier(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.encoder = BatCNNEncoder()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ENC_DIM, n_classes))

    def forward(self, x):
        return self.head(self.encoder.embed(x))


def load_ssl_encoder(classifier, ckpt_path):
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"No SSL checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    enc = ckpt.get("encoder_state")
    if not enc:
        raise KeyError(f"No encoder_state in SSL checkpoint: {path}")
    classifier.encoder.load_state_dict(enc, strict=True)
    return ckpt
