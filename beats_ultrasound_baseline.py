import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config as cfg
from bat.data import load_split
from bat.lightning_utils import ClassifierModule, SaveBest, final_eval, load_weights, log_dir, make_trainer, resolve_resume
from beats_bat import BEATsBatClassifier, build_beats_random, load_beats_checkpoint, waveform_to_beats_fbank

TARGET_SR = 192_000
CLIP_SAMPLES = int(2.0 * TARGET_SR)
ENERGY_CROP_PROB = 0.7
GAIN_JITTER_DB = 6.0

BEATS_PRETRAINED = cfg.CHECKPOINT_DIR / "BEATs_iter3.pt"
BEST_CKPT = cfg.CHECKPOINT_DIR / "beats_bat_finetune_best.pt"
CONFUSION_PNG = cfg.CHECKPOINT_DIR / "beats_bat_confusion_matrix.png"
BATCH_SIZE = 8
MAX_EPOCHS, PATIENCE = 40, 10
WD, LS = 1e-4, 0.1
BACKBONE_LR, HEAD_LR = 1e-5, 3e-4
BACKBONE_LR_RANDOM = 5e-4
PLATEAU_PATIENCE, LR_FACTOR, LR_MIN = 5, 0.5, 1e-7


def resample(y, orig_sr):
    if int(orig_sr) == int(TARGET_SR):
        return y.astype(np.float32)
    t_old = np.linspace(0, len(y) / orig_sr, len(y), endpoint=False)
    t_new = np.linspace(0, len(y) / orig_sr, int(len(y) * TARGET_SR / orig_sr), endpoint=False)
    return np.interp(t_new, t_old, y).astype(np.float32)


def crop_center(y, length):
    if len(y) >= length:
        i0 = (len(y) - length) // 2
        return y[i0 : i0 + length]
    out = np.zeros(length, dtype=np.float32)
    i0 = (length - len(y)) // 2
    out[i0 : i0 + len(y)] = y
    return out


def crop_train(y, length, rng):
    if len(y) < length:
        seg = crop_center(y, length)
    elif rng.random() < ENERGY_CROP_PROB:
        best_i0, best_e = 0, -1.0
        for _ in range(8):
            i0 = int(rng.integers(0, len(y) - length + 1))
            e = float(np.mean(y[i0 : i0 + length] ** 2))
            if e > best_e:
                best_e, best_i0 = e, i0
        seg = y[best_i0 : best_i0 + length]
    else:
        i0 = int(rng.integers(0, len(y) - length + 1))
        seg = y[i0 : i0 + length]
    if GAIN_JITTER_DB > 0:
        seg = (seg * 10 ** (rng.uniform(-GAIN_JITTER_DB, GAIN_JITTER_DB) / 20)).astype(np.float32)
    return seg


class BeatsDataset(Dataset):
    def __init__(self, df, training=False):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.rng = np.random.default_rng(cfg.RANDOM_SEED)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, sr = sf.read(str(row["path"]), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=-1)
        y = resample(y, sr)
        y = crop_train(y, CLIP_SAMPLES, self.rng) if self.training else crop_center(y, CLIP_SAMPLES)
        wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        return waveform_to_beats_fbank(wav, TARGET_SR, target_time_frames=None).float(), int(row["label"])


def make_loaders(train_df, val_df):
    args = {"batch_size": BATCH_SIZE, "num_workers": cfg.NUM_WORKERS}
    counts = train_df["label"].value_counts()
    weights = train_df["label"].map(lambda lbl: 1.0 / counts[lbl]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)
    return (
        DataLoader(BeatsDataset(train_df, True), sampler=sampler, **args),
        DataLoader(BeatsDataset(val_df, False), shuffle=False, **args),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", nargs="?", const="__auto__", default=None,
                   help="без пути: last.ckpt; .ckpt — полный resume; .pt — только веса")
    args = p.parse_args()

    pl.seed_everything(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, label2id, id2label, species = load_split(
        cfg.FT_METADATA_PATH, cfg.FT_DATA_DIR, expand_pulses=False,
    )
    train_loader, val_loader = make_loaders(train_df, val_df)

    if BEATS_PRETRAINED.is_file():
        backbone, beats_cfg = load_beats_checkpoint(BEATS_PRETRAINED, map_location="cpu")
        pretrained_ok, backbone_lr = True, BACKBONE_LR
    else:
        backbone, beats_cfg = build_beats_random()
        pretrained_ok, backbone_lr = False, BACKBONE_LR_RANDOM
        print(f"WARNING: random init BEATs (нет {BEATS_PRETRAINED})")

    model = BEATsBatClassifier(backbone, len(species), encoder_dim=beats_cfg.encoder_embed_dim)
    module = ClassifierModule(
        model,
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WD, plateau_patience=PLATEAU_PATIENCE, lr_factor=LR_FACTOR, lr_min=LR_MIN,
        label_smoothing=LS,
    )

    weights_ckpt, pl_ckpt, completed_epochs = resolve_resume(args.resume, "beats_baseline", BEST_CKPT)
    initial_best = -1.0
    if weights_ckpt is not None:
        meta = load_weights(model, weights_ckpt)
        initial_best = float(meta.get("val_macro_f1", -1.0))
        print(
            f"resume weights: {weights_ckpt} (saved epoch={meta.get('epoch', '?')}, "
            f"macro_f1={initial_best:.4f}, next epoch={completed_epochs})",
            flush=True,
        )
    elif pl_ckpt is not None:
        print(f"resume trainer: {pl_ckpt}", flush=True)

    extra = {"clip_sec": 2.0, "pretrained_loaded": pretrained_ok, "backbone_lr": backbone_lr, "head_lr": HEAD_LR}
    trainer = make_trainer(
        "beats_baseline", max_epochs=MAX_EPOCHS, monitor="macro_f1", mode="max", patience=PATIENCE,
        extra_callbacks=[SaveBest(BEST_CKPT, "beats_bat", label2id, id2label, extra=extra, initial_best_f1=initial_best)],
        continuing_run=args.resume is not None,
        restore_completed_epochs=completed_epochs if pl_ckpt is None else 0,
    )
    print(f"logs: {log_dir('beats_baseline')}")
    trainer.fit(module, train_loader, val_loader, ckpt_path=str(pl_ckpt) if pl_ckpt else None)

    if BEST_CKPT.is_file():
        load_weights(model, BEST_CKPT)
    final_eval(module, val_loader, species, CONFUSION_PNG)


if __name__ == "__main__":
    main()
