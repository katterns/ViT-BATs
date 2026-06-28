import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config as cfg
from bat.data.audio import load_spec, split_spec_overlap


class BatDataset(Dataset):
    def __init__(self, df, training=False, mode="ssl"):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.mode = mode
        self.rng = np.random.default_rng(cfg.RANDOM_SEED)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        center = int(row["pulse_center"])

        if self.mode == "dual":
            full = load_spec(row["path"], self.training, self.rng, pulse_center=center)
            left, right = split_spec_overlap(full)
            return full, left, right

        spec_aug = self.training and self.mode == "supervised"
        x = load_spec(row["path"], self.training, self.rng, spec_aug=spec_aug, pulse_center=center)
        if self.mode == "supervised":
            return x, row["label"]
        return x


def make_loaders(train_df, val_df, mode="ssl", balanced=False, val_mode=None):
    val_mode = val_mode or mode
    train_ds = BatDataset(train_df, training=True, mode=mode)
    val_ds = BatDataset(val_df, training=False, mode=val_mode)
    args = {"batch_size": cfg.BATCH_SIZE, "num_workers": cfg.NUM_WORKERS}

    if balanced:
        counts = train_df["label"].value_counts()
        weights = train_df["label"].map(lambda lbl: 1.0 / counts[lbl]).values
        sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **args)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **args)

    return train_loader, DataLoader(val_ds, shuffle=False, **args)
