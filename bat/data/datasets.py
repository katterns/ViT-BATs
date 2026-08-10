import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config as cfg
from bat.data.audio import load_spec
from bat.data.waveform_cache import WaveformClipCache


class BatDataset(Dataset):
    def __init__(
        self,
        df,
        training=False,
        mode="ssl",
        cache_dir=None,
        waveform_cache_dir=None,
    ):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.mode = mode
        self.cache_dir = cache_dir
        self.rng = np.random.default_rng(cfg.RANDOM_SEED)
        self.path_to_idxs = None
        if mode in ("recording", "recording_waveform"):
            self.path_to_idxs = {}
            for i, path in enumerate(self.df["path"].tolist()):
                self.path_to_idxs.setdefault(path, []).append(i)
        self.waveforms = None
        if mode in ("waveform", "recording_waveform", "waveform_only"):
            if waveform_cache_dir is None:
                raise ValueError(f"waveform_cache_dir is required for mode={mode!r}")
            self.waveforms = WaveformClipCache(self.df, waveform_cache_dir)

    def __len__(self):
        return len(self.df)

    def _load_row(self, row, spec_aug=False):
        return load_spec(
            row["path"], self.training, self.rng,
            spec_aug=spec_aug, pulse_center=int(row["pulse_center"]),
            cache_dir=self.cache_dir,
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        center = int(row["pulse_center"])

        if self.mode in ("recording", "recording_waveform"):
            x = self._load_row(row)
            peers = [j for j in self.path_to_idxs.get(row["path"], []) if j != idx]
            if peers:
                j = int(self.rng.choice(peers))
                x2 = self._load_row(self.df.iloc[j])
            else:
                x2 = x.clone()
            if self.mode == "recording_waveform":
                waveform = torch.from_numpy(
                    self.waveforms.load(row["path"], center).copy()
                )
                return x, x2, waveform
            return x, x2

        if self.mode == "waveform":
            x = self._load_row(row)
            waveform = torch.from_numpy(
                self.waveforms.load(row["path"], center).copy()
            )
            return x, waveform

        if self.mode == "waveform_only":
            waveform = torch.from_numpy(
                self.waveforms.load(row["path"], center).copy()
            )
            return (waveform,)

        spec_aug = self.training and self.mode == "supervised" and cfg.SUPERVISED_SPEC_AUG
        x = self._load_row(row, spec_aug=spec_aug)
        if self.mode == "supervised":
            return x, row["label"]
        return x


def _worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    dataset = worker_info.dataset
    if hasattr(dataset, "rng"):
        dataset.rng = np.random.default_rng(cfg.RANDOM_SEED + worker_info.id + 1)


def make_loaders(
    train_df,
    val_df,
    mode="ssl",
    balanced=False,
    val_mode=None,
    cache_dir=None,
    waveform_cache_dir=None,
):
    val_mode = val_mode or mode
    if cfg.USE_SPEC_CACHE and cache_dir is not None:
        from bat.data.spec_cache import cache_stats

        for name, df in (("train", train_df), ("val", val_df)):
            hits, total = cache_stats(df, cache_dir=cache_dir)
            if hits < total:
                raise RuntimeError(
                    f"spec cache incomplete for {name}: {hits}/{total} in {cache_dir}. "
                    "Training with cache misses runs gottbat per sample (~100x slower). "
                    "Run: uv run python scripts/precompute_specs.py"
                )

    train_ds = BatDataset(
        train_df,
        training=True,
        mode=mode,
        cache_dir=cache_dir,
        waveform_cache_dir=waveform_cache_dir,
    )
    val_ds = BatDataset(
        val_df,
        training=False,
        mode=val_mode,
        cache_dir=cache_dir,
        waveform_cache_dir=waveform_cache_dir,
    )

    num_workers = cfg.NUM_WORKERS
    loader_kw: dict = {
        "batch_size": cfg.BATCH_SIZE,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
        loader_kw["worker_init_fn"] = _worker_init_fn

    if balanced:
        counts = train_df["label"].value_counts()
        label_arr = train_df["label"].to_numpy()
        weights = np.array([1.0 / counts[lbl] for lbl in label_arr], dtype=np.float32)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(train_df), replacement=True
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)

    return train_loader, DataLoader(val_ds, shuffle=False, **loader_kw)
