from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bat.data.audio import SPEC_CHANNELS

PATCH_SIZE = (10, 10)
EMBED_DIM = 384
ENC_DEPTH = 10
DEC_DEPTH = 3
DEC_DIM = 288
N_HEADS = 6
DROP_PATH_RATE = 0.1


def make_grid(grid_shape, device=None):
    gf, gt = grid_shape
    f = torch.linspace(0.5 / gf, 1 - 0.5 / gf, gf, device=device)
    t = torch.linspace(0.5 / gt, 1 - 0.5 / gt, gt, device=device)
    ff, tt = torch.meshgrid(f, t, indexing="ij")
    return torch.stack([ff.reshape(-1), tt.reshape(-1)], dim=-1)


class LearnableFourierPosEncoding(nn.Module):
    def __init__(self, pe_dim=64, coord_dim=2, fourier_dim=64, hidden_dim=None, gamma=10.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(32, pe_dim // 2)
        self.W_r = nn.Parameter(torch.randn(fourier_dim // 2, coord_dim) / gamma)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pe_dim),
        )

    def forward(self, positions):
        proj = positions @ self.W_r.t()
        feats = torch.cat([proj.cos(), proj.sin()], dim=-1) / math.sqrt(self.W_r.shape[0] * 2)  # fourier_dim
        return self.mlp(feats)


class ConvStemPatchEmbed(nn.Module):
    def __init__(self, embed_dim=384, patch_size=(16, 16), stem_dim=48, in_channels=None):
        super().__init__()
        in_channels = in_channels or SPEC_CHANNELS
        self.patch_size = patch_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, stem_dim),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(stem_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(self.stem(x))
        return x.flatten(2).transpose(1, 2)


class ConvPosEnc1d(nn.Module):
    def __init__(self, dim, kernel_size=33, groups=16):
        super().__init__()
        g = groups
        while dim % g != 0 and g > 1:
            g -= 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=g)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.conv(x.transpose(1, 2)).transpose(1, 2))


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask.div(keep_prob)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x, gate = self.fc1(x).chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.fc2(self.drop(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.1, drop_path=0.0, layer_scale_init=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLUMLP(dim, hidden, dropout)
        self.drop_path = DropPath(drop_path)
        self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)
        x = x + self.drop_path(self.out_drop(attn_out) * self.ls1)
        return x + self.drop_path(self.out_drop(self.mlp(self.norm2(x))) * self.ls2)


def patchify(x, patch_size):
    pf, pt = patch_size
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // pf, pf, w // pt, pt)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(b, (h // pf) * (w // pt), c * pf * pt)

def patch_energy(x, patch_size):
    return patchify(x, patch_size).pow(2).mean(dim=-1)

def pack_masked_tokens(tokens, mask):
    b, n, d = tokens.shape
    device = tokens.device
    keep_counts = (1.0 - mask).sum(dim=1).long()
    max_keep = int(keep_counts.max().item())

    tokens_keep = torch.zeros(b, max_keep, d, device=device, dtype=tokens.dtype)
    pad_mask = torch.ones(b, max_keep, dtype=torch.bool, device=device)
    ids_restore = torch.zeros(b, n, dtype=torch.long, device=device)

    for i in range(b):
        keep_idx = (mask[i] == 0).nonzero(as_tuple=True)[0]
        drop_idx = (mask[i] == 1).nonzero(as_tuple=True)[0]
        nk = keep_idx.numel()
        tokens_keep[i, :nk] = tokens[i, keep_idx]
        pad_mask[i, :nk] = False
        ids_restore[i] = torch.argsort(torch.cat([keep_idx, drop_idx]))

    return tokens_keep, mask, ids_restore, pad_mask

def block_mask(tokens, mask_ratio, grid_shape, energies, block=(2, 2), noise_percentile=25.0, signal_fraction=0.7):
    b, n, _ = tokens.shape
    device = tokens.device
    gf, gt = grid_shape
    bf, bt = block
    min_signal = max(4, n // 4)
    target_mask = max(1, min(int(round(n * mask_ratio)), n - 1))
    signal_mask = max(1, min(int(round(target_mask * signal_fraction)), target_mask))

    coarse_f = math.ceil(gf / bf)
    coarse_t = math.ceil(gt / bt)
    scores = torch.rand(b, coarse_f, coarse_t, device=device)
    scores = scores.repeat_interleave(bf, dim=1).repeat_interleave(bt, dim=2)
    patch_scores = scores[:, :gf, :gt].reshape(b, n)

    mask = torch.zeros(b, n, device=device)
    thr = torch.quantile(energies, noise_percentile / 100.0, dim=1, keepdim=True)
    is_signal = energies > thr

    for i in range(b):
        sig_idx = is_signal[i].nonzero(as_tuple=True)[0]
        if sig_idx.numel() < min_signal:
            sig_idx = energies[i].topk(min_signal).indices
        n_signal = min(signal_mask, sig_idx.numel())
        pick = sig_idx[
            (patch_scores[i, sig_idx] + 1e-4 * torch.rand(sig_idx.numel(), device=device))
            .argsort(descending=True)[:n_signal]
        ]
        mask[i, pick] = 1.0

        remaining = target_mask - int(mask[i].sum().item())
        if remaining > 0:
            candidates = (mask[i] == 0).nonzero(as_tuple=True)[0]
            scores = patch_scores[i, candidates] + 1e-4 * torch.rand(candidates.numel(), device=device)
            pick = candidates[scores.argsort(descending=True)[:remaining]]
            mask[i, pick] = 1.0

    return pack_masked_tokens(tokens, mask)

def cosine_loss(pred, target):
    return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()

def nt_xent_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.shape[0]
    features = torch.cat([z1, z2], dim=0)
    logits = features @ features.T / temperature
    labels = torch.arange(2 * batch_size, device=features.device)
    labels = (labels + batch_size) % (2 * batch_size)
    logits = logits.masked_fill(torch.eye(2 * batch_size, device=features.device, dtype=torch.bool), float("-inf"))
    return F.cross_entropy(logits, labels)

def unpatchify_2d(patches, patch_size, channels=SPEC_CHANNELS):
    pf, pt = patch_size
    if patches.dim() == 6:
        b, gf, gt, c, _, _ = patches.shape
        return patches.permute(0, 3, 1, 4, 2, 5).reshape(b, c, gf * pf, gt * pt)
    if patches.dim() == 5:
        b, gf, gt, _, _ = patches.shape
        return patches.permute(0, 1, 3, 2, 4).reshape(b, 1, gf * pf, gt * pt)

    b, n, d = patches.shape
    patch_dim = channels * pf * pt
    if d != patch_dim:
        raise ValueError(f"patch dim {d} != channels*patch_h*patch_w {patch_dim}")
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"cannot infer square grid from {n} patches")
    patches = patches.view(b, side, side, channels, pf, pt)
    return unpatchify_2d(patches, patch_size, channels)


class BatViTEncoder(nn.Module):
    def __init__(
        self,
        spec_h,
        spec_w,
        embed_dim=EMBED_DIM,
        depth=ENC_DEPTH,
        n_heads=N_HEADS,
        patch_size=PATCH_SIZE,
        drop_path_rate=DROP_PATH_RATE,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = ConvStemPatchEmbed(embed_dim, patch_size)
        self.grid_shape = (spec_h // patch_size[0], spec_w // patch_size[1])
        self.num_patches = self.grid_shape[0] * self.grid_shape[1]

        self.pos_encoder = LearnableFourierPosEncoding(pe_dim=embed_dim)
        self.conv_pos = ConvPosEnc1d(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, drop_path=drop_rates[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _patch_pos(self, batch_size, device):
        pe = self.pos_encoder(make_grid(self.grid_shape, device))
        return pe.unsqueeze(0).expand(batch_size, -1, -1)

    def embed_patch_tokens(self, x):
        tokens = self.patch_embed(x)
        return tokens + self._patch_pos(tokens.shape[0], tokens.device)

    def forward_tokens(self, tokens, key_padding_mask=None):
        b = tokens.shape[0]
        cls = self.cls_token.to(tokens.dtype).expand(b, -1, -1)
        if key_padding_mask is not None:
            cls_pad = torch.zeros(b, 1, dtype=torch.bool, device=tokens.device)
            key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)
        tokens = torch.cat([cls, self.conv_pos(tokens)], dim=1)
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)
        return self.norm(tokens)

    def forward_full(self, x):
        return self.forward_tokens(self.embed_patch_tokens(x))

    def encoder_state_dict(self):
        keys = ("patch_embed", "pos_encoder", "conv_pos", "blocks", "norm", "cls_token")
        return {k: v for k, v in self.state_dict().items() if k.split(".")[0] in keys}


class BatViTPatchMAE(nn.Module):
    def __init__(self, spec_h, spec_w, embed_dim=EMBED_DIM, dec_depth=DEC_DEPTH, dec_dim=DEC_DIM, n_heads=N_HEADS, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = BatViTEncoder(spec_h, spec_w, embed_dim, ENC_DEPTH, n_heads, patch_size)
        self.grid_shape = self.encoder.grid_shape
        self.num_patches = self.encoder.num_patches
        self.patch_dim = SPEC_CHANNELS * patch_size[0] * patch_size[1]

        self.decoder_embed = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos = LearnableFourierPosEncoding(pe_dim=dec_dim)
        self.decoder_blocks = nn.ModuleList([TransformerBlock(dec_dim, n_heads, drop_path=0.0) for _ in range(dec_depth)])
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, self.patch_dim)

    def _decode(self, latent, ids_restore, pad_mask, batch_size, device):
        dec = self.decoder_embed(latent)
        n = self.num_patches
        dec_full = torch.zeros(batch_size, n, dec.shape[-1], device=device, dtype=dec.dtype)
        keep_counts = (~pad_mask).sum(dim=1)
        pe = self.decoder_pos(make_grid(self.grid_shape, device)).unsqueeze(0).expand(batch_size, -1, -1)
        for i in range(batch_size):
            nk = int(keep_counts[i].item())
            row = torch.cat([dec[i, :nk], self.mask_token.squeeze(0).expand(n - nk, -1)], dim=0)
            dec_full[i] = row[ids_restore[i]]
        dec_full = dec_full + pe
        for blk in self.decoder_blocks:
            dec_full = blk(dec_full)
        return self.decoder_norm(dec_full)

    def forward(self, x, mask_ratio=0.75, noise_percentile=25.0, utterance_weight=0.0):
        targets = patchify(x, self.patch_size)
        energies = patch_energy(x, self.patch_size)
        tokens, mask, ids_restore, pad_mask = block_mask(
            self.encoder.embed_patch_tokens(x),
            mask_ratio,
            self.grid_shape,
            energies,
            noise_percentile=noise_percentile,
        )
        encoded = self.encoder.forward_tokens(tokens, key_padding_mask=pad_mask)
        pred = self.pred(self._decode(encoded[:, 1:], ids_restore, pad_mask, x.shape[0], tokens.device))

        loss_per_patch = ((pred - targets) ** 2).mean(dim=-1)
        recon_loss = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        sem = {"recon": recon_loss.detach()}
        total_loss = recon_loss
        if utterance_weight > 0:
            cls = encoded[:, 0]
            utt = cosine_loss(cls, encoded[:, 1:].mean(dim=1).detach())
            sem["utterance"] = utt.detach()
            total_loss = total_loss + utterance_weight * utt
        return total_loss, pred, mask, targets, sem

    def contrastive_loss(self, x1, x2, temperature=0.07):
        tokens = self.encoder.forward_full(torch.cat([x1, x2], dim=0))
        z1, z2 = tokens[:, 0].chunk(2)
        return nt_xent_loss(z1, z2, temperature)

    def encoder_state_dict(self):
        return self.encoder.encoder_state_dict()


class BatViTClassifier(nn.Module):
    def __init__(self, n_classes, spec_h, spec_w, embed_dim=EMBED_DIM, depth=ENC_DEPTH, n_heads=N_HEADS, patch_size=PATCH_SIZE):
        super().__init__()
        self.encoder = BatViTEncoder(spec_h, spec_w, embed_dim, depth, n_heads, patch_size)
        self.pool_score = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.head = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Dropout(0.2), nn.Linear(embed_dim * 2, n_classes))

    def forward(self, x):
        patch_tokens = self.encoder.forward_full(x)[:, 1:]
        weights = self.pool_score(patch_tokens).softmax(dim=1)
        mean = (patch_tokens * weights).sum(dim=1)
        var = (weights * (patch_tokens - mean.unsqueeze(1)).pow(2)).sum(dim=1)
        return self.head(torch.cat([mean, var.clamp_min(1e-6).sqrt()], dim=-1))

    def encoder_parameters(self):
        return self.encoder.parameters()


def load_ssl_encoder(classifier, ckpt_path, device, spec_hw, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM):
    path = Path(ckpt_path) if not isinstance(ckpt_path, Path) else ckpt_path
    if not path.is_file():
        raise FileNotFoundError(f"Нет SSL чекпоинта: {path}. Запустите vit_lfpe_ssl_pretrain.py")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    if ckpt_cfg.get("preprocess") not in (None, "nabat_v2"):
        raise ValueError(f"SSL ckpt preprocess={ckpt_cfg.get('preprocess')!r}, need nabat_v2")
    if ckpt_cfg.get("spec_channels") not in (None, SPEC_CHANNELS):
        raise ValueError(f"SSL ckpt spec_channels={ckpt_cfg.get('spec_channels')}, need {SPEC_CHANNELS}")
    expected = {"spec_hw": spec_hw, "patch_size": patch_size, "embed_dim": embed_dim}
    for key, val in expected.items():
        actual = ckpt_cfg.get(key)
        if isinstance(actual, list):
            actual = tuple(actual)
        if actual is not None and actual != val:
            raise ValueError(f"SSL ckpt {key}: {actual} != {val}")

    enc = ckpt.get("encoder_state")
    if not enc:
        raise KeyError("No encoder_state in checkpoint")
    classifier.encoder.load_state_dict(enc, strict=True)
    print(
        f"loaded SSL encoder: {path} (epoch={ckpt.get('epoch', '?')}, "
        f"val_recon_loss={ckpt.get('val_recon_loss', float('nan')):.4f})",
        flush=True,
    )
