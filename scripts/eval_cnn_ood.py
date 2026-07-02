"""Inference CNN на файлах full cleaned, которых нет в subset_200 (OOD). Без обучения."""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config as cfg
from bat.data.datasets import BatDataset
from bat.data.splits import expand_by_pulses
from bat.lightning_utils import load_weights
from supervised_cnn_baseline import BatCNNA0, BEST_CKPT

CONF_THRESH = 0.57


def ood_files():
    sub = pd.read_csv(cfg.FT_METADATA_PATH)
    full = pd.read_csv(cfg.METADATA_PATH)
    sub["key"] = sub["species"] + "/" + sub["filename"]
    full["key"] = full["species"] + "/" + full["filename"]
    ood = full[~full["key"].isin(set(sub["key"]))].copy()
    ood["path"] = ood.apply(lambda r: cfg.DATA_DIR / r["species"] / r["filename"], axis=1)
    return ood[ood["path"].apply(lambda p: p.exists())].copy()


def label_maps():
    sub = pd.read_csv(cfg.FT_METADATA_PATH)
    species = sorted(sub["species"].unique())
    label2id = {name: i for i, name in enumerate(species)}
    return species, label2id


@torch.no_grad()
def infer(model, loader):
    model.eval()
    y_true, y_pred, y_conf, probs = [], [], [], []
    for x, y in loader:
        p = F.softmax(model(x), dim=-1)
        conf, pred = p.max(dim=-1)
        y_pred.extend(pred.tolist())
        y_conf.extend(conf.tolist())
        probs.append(p.numpy())
        y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_conf), np.vstack(probs)


def file_aggregate(paths, y_true, y_pred, probs, conf_thresh=None):
    df = pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred})
    file_true, file_pred, n_noid = [], [], 0
    for path, g in df.groupby("path", sort=False):
        file_true.append(int(g["y_true"].iloc[0]))
        idxs = g.index.to_numpy()
        mean_p = probs[idxs].mean(axis=0)
        pred = int(mean_p.argmax())
        if conf_thresh is not None and float(mean_p.max()) < conf_thresh:
            n_noid += 1
            file_pred.append(-1)
        else:
            file_pred.append(pred)
    file_true = np.array(file_true)
    file_pred = np.array(file_pred)
    mask = file_pred >= 0
    return file_true[mask], file_pred[mask], len(file_true), n_noid


def metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "w_prec": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "w_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="если >0 — stratified sample N файлов из OOD")
    p.add_argument("--seed", type=int, default=43)
    args = p.parse_args()

    species, label2id = label_maps()
    df = ood_files()
    df["label"] = df["species"].map(label2id)
    missing = sorted(set(species) - set(df["species"].unique()))
    print(f"OOD pool: {len(df)} files, {df['species'].nunique()} species", flush=True)
    if missing:
        print(f"species без OOD-файлов (все в subset_200): {missing}", flush=True)

    if args.sample > 0:
        # stratify только по классам, которые есть в OOD
        counts = df["label"].value_counts()
        usable = counts[counts >= 2].index
        df = df[df["label"].isin(usable)]
        _, df = train_test_split(
            df, test_size=min(args.sample, len(df)),
            random_state=args.seed, stratify=df["label"],
        )
        print(f"stratified sample: {len(df)} files (seed={args.seed})", flush=True)

    pulse_df = expand_by_pulses(df.reset_index(drop=True), desc="OOD pulses")
    print(f"pulses after quality filter: {len(pulse_df)}", flush=True)

    loader = DataLoader(
        BatDataset(pulse_df, training=False, mode="supervised"),
        batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0,
    )
    model = BatCNNA0(len(species))
    meta = load_weights(model, BEST_CKPT)
    print(f"checkpoint: epoch={meta.get('epoch')} saved_f1={meta.get('val_macro_f1'):.4f}", flush=True)

    paths = pulse_df["path"].astype(str).tolist()
    y_true, y_pred, y_conf, probs = infer(model, loader)

    print("\n=== Pulse-level ===", flush=True)
    m = metrics(y_true, y_pred)
    print(f"acc={m['acc']:.4f} w_prec={m['w_prec']:.4f} macro_f1={m['macro_f1']:.4f}", flush=True)

    print("\n=== File-level (mean prob) ===", flush=True)
    ft, fp, nfiles, _ = file_aggregate(paths, y_true, y_pred, probs)
    m = metrics(ft, fp)
    print(f"files={nfiles} acc={m['acc']:.4f} w_prec={m['w_prec']:.4f} macro_f1={m['macro_f1']:.4f}", flush=True)

    print(f"\n=== File-level + conf>={CONF_THRESH} ===", flush=True)
    ft, fp, nfiles, n_noid = file_aggregate(paths, y_true, y_pred, probs, conf_thresh=CONF_THRESH)
    m = metrics(ft, fp)
    print(f"scored={len(ft)}/{nfiles} NoID={n_noid} acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
