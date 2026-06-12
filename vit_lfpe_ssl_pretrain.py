import argparse
import logging
import math
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from clearml import OutputModel, Task
from dotenv import load_dotenv
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import config as cfg
from vit_bat import BatViTPatchMAE, unpatchify_2d


def parse_args():
    parser = argparse.ArgumentParser(description="ViT LF-PE SSL pretrain (MAE)")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=str(cfg.BEST_CKPT),
        default=None,
        help="Продолжить с чекпоинта (без аргумента — vit_lfpe_ssl_pretrain_best.pt)",
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    if cfg.DEVICE:
        return torch.device(cfg.DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def milestone_epochs(max_epochs: int) -> list[int]:
    epochs = []
    e = cfg.MILESTONE_START_EPOCH
    while e <= max_epochs:
        epochs.append(e)
        e += cfg.MILESTONE_EVERY
    return epochs


# --- audio / dataset ---

def resample_audio(y: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return y.astype(np.float32)
    t_old = np.linspace(0.0, len(y) / orig_sr, num=len(y), endpoint=False)
    t_new = np.linspace(
        0.0, len(y) / orig_sr, num=int(len(y) * target_sr / orig_sr), endpoint=False
    )
    return np.interp(t_new, t_old, y).astype(np.float32)


def center_crop_or_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        i0 = (len(y) - target_len) // 2
        return y[i0 : i0 + target_len]
    out = np.zeros(target_len, dtype=np.float32)
    i0 = (target_len - len(y)) // 2
    out[i0 : i0 + len(y)] = y
    return out


def random_crop_or_pad(y: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    if len(y) >= target_len:
        i0 = int(rng.integers(0, len(y) - target_len + 1))
        return y[i0 : i0 + target_len]
    return center_crop_or_pad(y, target_len)


def energy_biased_crop(
    y: np.ndarray, target_len: int, rng: np.random.Generator, n_candidates: int
) -> np.ndarray:
    if len(y) < target_len:
        return center_crop_or_pad(y, target_len)
    best_i0, best_e = 0, -1.0
    for _ in range(n_candidates):
        i0 = int(rng.integers(0, len(y) - target_len + 1))
        e = float(np.mean(y[i0 : i0 + target_len] ** 2))
        if e > best_e:
            best_e, best_i0 = e, i0
    return y[best_i0 : best_i0 + target_len]


def pick_train_crop(y: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < cfg.ENERGY_CROP_PROB:
        return energy_biased_crop(y, target_len, rng, cfg.ENERGY_CROP_CANDIDATES)
    return random_crop_or_pad(y, target_len, rng)


def maybe_gain_jitter(y: np.ndarray, rng: np.random.Generator, max_db: float) -> np.ndarray:
    if max_db <= 0:
        return y
    gain = 10.0 ** (float(rng.uniform(-max_db, max_db)) / 20.0)
    return (y * gain).astype(np.float32)


def make_log_stft(audio: np.ndarray, sr: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio[np.isfinite(audio)]
    if len(audio) < 8:
        raise ValueError("Audio too short")
    nperseg = int(min(cfg.N_FFT, len(audio)))
    hop = int(min(cfg.HOP_LENGTH, max(1, nperseg // 2)))
    noverlap = max(0, nperseg - hop)
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    freqs, _, zxx = signal.stft(
        audio, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary="zeros", padded=True
    )
    spec = np.log1p(np.abs(zxx))
    keep = (freqs >= cfg.MIN_FREQ) & (freqs <= min(cfg.MAX_FREQ, sr / 2))
    return spec[keep]


def spec_to_tensor(spec: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(spec, dtype=np.float32))[None, None]
    x = F.interpolate(x, size=(cfg.SPEC_H, cfg.SPEC_W), mode="bilinear", align_corners=False)[0]
    return (x - x.mean()) / (x.std().clamp_min(1e-6))


class BatWavSpectrogramSSLDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, training: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.rng = np.random.default_rng(cfg.RANDOM_SEED)
        self.aug_strength = 0.0

    def set_aug_strength(self, strength: float) -> None:
        self.aug_strength = float(max(0.0, min(1.0, strength)))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        y, sr = sf.read(str(row["path"]), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=-1)
        y = resample_audio(y, float(sr), float(cfg.TARGET_SR))
        tgt = int(cfg.CLIP_SEC * cfg.TARGET_SR)
        if self.training:
            y = pick_train_crop(y, tgt, self.rng)
            if self.aug_strength > 0:
                y = maybe_gain_jitter(
                    y, self.rng, max_db=cfg.WAV_GAIN_JITTER_DB * self.aug_strength
                )
        else:
            y = center_crop_or_pad(y, tgt)
        return spec_to_tensor(make_log_stft(y, float(cfg.TARGET_SR)))


def wav_row_to_spec(row) -> torch.Tensor:
    y, sr = sf.read(str(row["path"]), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=-1)
    y = resample_audio(y, float(sr), float(cfg.TARGET_SR))
    y = center_crop_or_pad(y, int(cfg.CLIP_SEC * cfg.TARGET_SR))
    return spec_to_tensor(make_log_stft(y, float(cfg.TARGET_SR))).unsqueeze(0)


# --- training helpers ---

class TrainLogger:
    HEADER = "epoch,lr,aug_strength,train_loss,val_loss,is_best"

    def __init__(self, path: Path, resume: bool = False):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.is_file():
            with path.open("a", encoding="utf-8") as f:
                f.write(f"\n# resumed {datetime.now().isoformat(timespec='seconds')}\n")
        else:
            path.write_text(self.HEADER + "\n", encoding="utf-8")

    def log_epoch(
        self, epoch: int, lr: float, aug: float, train_loss: float, val_loss: float, is_best: bool
    ) -> None:
        line = (
            f"{epoch},{lr:.6e},{aug:.4f},{train_loss:.6f},{val_loss:.6f},{int(is_best)}\n"
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)


def parse_train_log(path: Path) -> dict[str, list]:
    epochs, lrs, augs, trains, vals, bests = [], [], [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("epoch,"):
            continue
        parts = line.split(",")
        if len(parts) != 6:
            continue
        epochs.append(int(parts[0]))
        lrs.append(float(parts[1]))
        augs.append(float(parts[2]))
        trains.append(float(parts[3]))
        vals.append(float(parts[4]))
        bests.append(bool(int(parts[5])))
    return {
        "epoch": epochs,
        "lr": lrs,
        "aug": augs,
        "train_loss": trains,
        "val_loss": vals,
        "is_best": bests,
    }


def aug_strength_for_epoch(epoch: int) -> float:
    if epoch < cfg.AUGMENT_START_EPOCH:
        return 0.0
    step = (epoch - cfg.AUGMENT_START_EPOCH) // cfg.AUGMENT_RAMP_EVERY
    return min(1.0, (step + 1) / cfg.AUGMENT_RAMP_STEPS)


def set_epoch_lr(optimizer, epoch: int) -> float:
    if epoch <= cfg.WARMUP_EPOCHS:
        lr = cfg.BASE_LR * epoch / max(cfg.WARMUP_EPOCHS, 1)
    else:
        progress = (epoch - cfg.WARMUP_EPOCHS) / max(cfg.MAX_EPOCHS - cfg.WARMUP_EPOCHS, 1)
        lr = cfg.MIN_LR + 0.5 * (cfg.BASE_LR - cfg.MIN_LR) * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(cfg.METADATA_PATH)
    assert not df.empty, f"Пустой CSV: {cfg.METADATA_PATH}"
    species_sorted = sorted(df["species"].unique())
    df["label"] = df["species"].map({s: i for i, s in enumerate(species_sorted)})
    df["path"] = df.apply(
        lambda r: cfg.DATA_DIR / str(r["species"]) / str(r["filename"]), axis=1
    )
    df = df[df["path"].apply(lambda p: p.is_file())].reset_index(drop=True)
    train_df, val_df = train_test_split(
        df, test_size=cfg.VAL_TEST_SIZE, random_state=cfg.RANDOM_SEED, stratify=df["label"]
    )
    return train_df, val_df


def checkpoint_payload(model: BatViTPatchMAE, epoch: int, val_loss: float) -> dict:
    return {
        "model_state": model.state_dict(),
        "encoder_state": model.encoder_state_dict(),
        "model_name": "vit_lfpe_mae",
        "config": {
            "target_sr": cfg.TARGET_SR,
            "clip_sec": cfg.CLIP_SEC,
            "min_freq": cfg.MIN_FREQ,
            "max_freq": cfg.MAX_FREQ,
            "spec_hw": (cfg.SPEC_H, cfg.SPEC_W),
            "patch_size": cfg.PATCH_SIZE,
            "embed_dim": cfg.EMBED_DIM,
            "enc_depth": cfg.ENC_DEPTH,
            "dec_depth": cfg.DEC_DEPTH,
            "dec_dim": cfg.DEC_DIM,
            "n_heads": cfg.N_HEADS,
            "conv_pos_kernel": cfg.CONV_POS_KERNEL,
            "mask_ratio": cfg.MASK_RATIO,
            "noise_patch_percentile": cfg.NOISE_PATCH_PERCENTILE,
            "norm_pix_loss": cfg.NORM_PIX_LOSS,
        },
        "val_recon_loss": val_loss,
        "epoch": epoch,
    }


def save_checkpoint(model: BatViTPatchMAE, path: Path, epoch: int, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, epoch, val_loss), path)


def load_checkpoint(model: BatViTPatchMAE, path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Resume: нет файла {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    last_epoch = int(ckpt.get("epoch", 0))
    best_val_loss = float(ckpt.get("val_recon_loss", float("inf")))
    return last_epoch + 1, best_val_loss, last_epoch


def make_model(device: torch.device) -> BatViTPatchMAE:
    return BatViTPatchMAE(
        spec_h=cfg.SPEC_H,
        spec_w=cfg.SPEC_W,
        embed_dim=cfg.EMBED_DIM,
        enc_depth=cfg.ENC_DEPTH,
        dec_depth=cfg.DEC_DEPTH,
        dec_dim=cfg.DEC_DIM,
        n_heads=cfg.N_HEADS,
        patch_size=cfg.PATCH_SIZE,
        mlp_ratio=cfg.MLP_RATIO,
        dropout=cfg.DROPOUT,
        norm_pix_loss=cfg.NORM_PIX_LOSS,
        conv_pos_kernel=cfg.CONV_POS_KERNEL,
    ).to(device)


def setup_clearml_console(quiet: bool = True) -> None:
    if not quiet:
        return
    for name in ("clearml", "clearml.storage", "clearml.model", "clearml.Task"):
        logging.getLogger(name).setLevel(logging.WARNING)


def init_clearml(device: torch.device, resume: bool):
    load_dotenv(cfg.PROJECT_DIR / ".env", override=False)
    clearml_quiet = os.environ.get("CLEARML_QUIET", "1").strip().lower() in ("1", "true", "yes")
    clearml_upload_each_best = os.environ.get("CLEARML_UPLOAD_EACH_BEST", "0").strip().lower() in (
        "1", "true", "yes",
    )
    setup_clearml_console(clearml_quiet)

    clearml_output_uri = os.environ.get("CLEARML_OUTPUT_URI", True)
    if isinstance(clearml_output_uri, str) and clearml_output_uri.lower() in ("false", "0", "no", "none"):
        clearml_output_uri = False

    task_init_kw = dict(
        project_name=cfg.CLEARML_PROJECT,
        task_name=cfg.CLEARML_TASK_NAME,
        output_uri=clearml_output_uri,
        auto_resource_monitoring=False,
    )
    if cfg.CLEARML_TASK_ID:
        task_init_kw["continue_last_task"] = cfg.CLEARML_TASK_ID
        task_init_kw["reuse_last_task_id"] = cfg.CLEARML_TASK_ID
    elif cfg.CLEARML_CONTINUE_LAST and resume:
        task_init_kw["continue_last_task"] = 0
        task_init_kw["reuse_last_task_id"] = True
    else:
        task_init_kw["reuse_last_task_id"] = False

    clearml_task = Task.init(**task_init_kw)
    output_dest = clearml_task.get_output_destination() or ""
    output_model = OutputModel(task=clearml_task, name="vit_lfpe_ssl_pretrain", framework="pytorch")
    if output_dest:
        output_model.set_upload_destination(output_dest)

    clearml_task.connect({
        "model": "vit_lfpe_mae_v3_signal_aware",
        "clip_sec": cfg.CLIP_SEC,
        "batch_size": cfg.BATCH_SIZE,
        "base_lr": cfg.BASE_LR,
        "mask_ratio": cfg.MASK_RATIO,
        "device": str(device),
        "resume": resume,
    })
    print(f"device: {device} | clearml task: {clearml_task.id}")
    return clearml_task, output_model, output_dest, clearml_quiet, clearml_upload_each_best


@torch.no_grad()
def evaluate(model, val_loader, device, epoch: int | None = None):
    model.eval()
    total, n = 0.0, 0
    desc = f"val e{epoch}" if epoch is not None else "val"
    for batch_x in tqdm(val_loader, desc=desc, leave=False, unit="batch"):
        loss, _, _, _ = model(
            batch_x.to(device),
            mask_ratio=cfg.MASK_RATIO,
            noise_percentile=cfg.NOISE_PATCH_PERCENTILE,
        )
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train(
    model,
    train_loader,
    val_loader,
    train_ds,
    optimizer,
    device,
    logger: TrainLogger,
    clearml_task,
    output_model,
    output_dest,
    clearml_quiet,
    clearml_upload_each_best,
    resume_path: Path | None,
):
    milestones = set(milestone_epochs(cfg.MAX_EPOCHS))
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    start_epoch = 1

    if resume_path:
        start_epoch, best_val_loss, best_epoch = load_checkpoint(model, resume_path, device)
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        tqdm.write(
            f"Resume: {resume_path.name}  last_epoch={best_epoch}  "
            f"best_val={best_val_loss:.6f}  epochs {start_epoch}..{cfg.MAX_EPOCHS}"
        )
        if start_epoch > cfg.MAX_EPOCHS:
            raise ValueError(f"Resume: start_epoch={start_epoch} > MAX_EPOCHS={cfg.MAX_EPOCHS}")

    def upload_clearml(epoch: int):
        if output_dest and clearml_upload_each_best:
            output_model.update_weights(
                weights_filename=str(cfg.BEST_CKPT),
                upload_uri=output_dest,
                auto_delete_file=False,
                iteration=epoch,
            )

    cl_logger = clearml_task.get_logger()
    epoch_bar = tqdm(
        range(start_epoch, cfg.MAX_EPOCHS + 1),
        desc="epochs",
        unit="epoch",
        initial=start_epoch - 1,
        total=cfg.MAX_EPOCHS,
    )

    for epoch in epoch_bar:
        aug_s = aug_strength_for_epoch(epoch)
        train_ds.set_aug_strength(aug_s)
        lr_now = set_epoch_lr(optimizer, epoch)
        model.train()
        train_loss, n_batches = 0.0, 0

        for batch_x in tqdm(train_loader, desc=f"train e{epoch}", leave=False, unit="batch"):
            batch_x = batch_x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = model(
                batch_x,
                mask_ratio=cfg.MASK_RATIO,
                noise_percentile=cfg.NOISE_PATCH_PERCENTILE,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        tr_loss = train_loss / max(n_batches, 1)
        val_loss = evaluate(model, val_loader, device, epoch=epoch)
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(model, cfg.BEST_CKPT, epoch, val_loss)
            upload_clearml(epoch)
            tqdm.write(f"saved {cfg.BEST_CKPT.name}  val_loss={val_loss:.6f}")
        else:
            epochs_no_improve += 1

        logger.log_epoch(epoch, lr_now, aug_s, tr_loss, val_loss, is_best)

        cl_logger.report_scalar("loss", "train", tr_loss, iteration=epoch)
        cl_logger.report_scalar("loss", "val", val_loss, iteration=epoch)
        cl_logger.report_scalar("lr", "lr", lr_now, iteration=epoch)
        cl_logger.report_scalar("aug", "strength", aug_s, iteration=epoch)

        epoch_bar.set_postfix(val_loss=f"{val_loss:.6f}", aug=f"{aug_s:.2f}")
        tqdm.write(
            f"Epoch {epoch:02d}  lr={lr_now:.2e}  aug={aug_s:.2f}  "
            f"train={tr_loss:.6f}  val={val_loss:.6f}"
        )

        if epoch in milestones and best_state is not None:
            milestone_path = Path(str(cfg.MILESTONE_CKPT).format(epoch=epoch))
            current = model.state_dict()
            model.load_state_dict(best_state)
            save_checkpoint(model, milestone_path, best_epoch, best_val_loss)
            model.load_state_dict(current)
            tqdm.write(
                f"milestone ep{epoch}: saved best (ep{best_epoch}, val={best_val_loss:.6f}) "
                f"-> {milestone_path.name}"
            )

        if epochs_no_improve >= cfg.PATIENCE:
            tqdm.write("Early stopping.")
            break

    epoch_bar.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    tqdm.write(f"best val recon loss: {best_val_loss:.6f} (epoch {best_epoch})")
    return best_epoch, best_val_loss


# --- post-training artifacts ---

def save_loss_plot(log_path: Path, out_path: Path) -> None:
    data = parse_train_log(log_path)
    if not data["epoch"]:
        print(f"Нет данных для графика: {log_path}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(data["epoch"], data["train_loss"], label="train", color="#2563eb")
    axes[0].plot(data["epoch"], data["val_loss"], label="val", color="#dc2626")
    best_eps = [e for e, b in zip(data["epoch"], data["is_best"]) if b]
    best_vals = [v for v, b in zip(data["val_loss"], data["is_best"]) if b]
    if best_eps:
        axes[0].scatter(best_eps, best_vals, s=24, c="#16a34a", zorder=3, label="new best")
    for ms in milestone_epochs(max(data["epoch"])):
        if ms in data["epoch"]:
            axes[0].axvline(ms, color="#94a3b8", ls="--", lw=0.8, alpha=0.7)
    axes[0].set_ylabel("reconstruction loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(data["epoch"], data["aug"], label="aug strength", color="#7c3aed")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("aug strength")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    fig.suptitle("ViT LF-PE SSL pretrain", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved loss plot: {out_path}")


@torch.no_grad()
def save_recon_demo(model, val_df: pd.DataFrame, device, out_path: Path) -> None:
    model.eval()
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    row = val_df.iloc[int(rng.integers(len(val_df)))]
    x = wav_row_to_spec(row).to(device)

    _, pred, mask, targets = model(
        x,
        mask_ratio=cfg.MASK_RATIO,
        noise_percentile=cfg.NOISE_PATCH_PERCENTILE,
    )
    pf, pt = cfg.PATCH_SIZE
    gf, gt = model.grid_shape
    pred_p = pred.view(1, gf, gt, pf, pt)
    tgt_p = targets.view(1, gf, gt, pf, pt)
    m = mask.view(1, gf, gt).bool()

    recon_p = tgt_p.clone()
    recon_p[m] = pred_p[m]
    target_img = x[0, 0].cpu().numpy()
    recon_img = unpatchify_2d(recon_p[0], cfg.PATCH_SIZE).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    title = f"{row['species']}/{row['filename']}"
    fig.suptitle(title, fontsize=10)
    axes[0].imshow(target_img, origin="lower", cmap="magma", aspect="auto")
    axes[0].set_title("target")
    axes[0].axis("off")
    axes[1].imshow(recon_img, origin="lower", cmap="magma", aspect="auto")
    axes[1].set_title("reconstruction")
    axes[1].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved recon demo: {out_path}")


def main():
    args = parse_args()
    resume_path = Path(args.resume) if args.resume else None

    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    clearml_task, output_model, output_dest, clearml_quiet, clearml_upload_each_best = init_clearml(
        device, resume=resume_path is not None
    )

    train_df, val_df = load_metadata()
    print(f"SSL train: {len(train_df)}  SSL val monitor: {len(val_df)}")

    train_ds = BatWavSpectrogramSSLDataset(train_df, training=True)
    val_ds = BatWavSpectrogramSSLDataset(val_df, training=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = make_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MAE params: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.BASE_LR, weight_decay=cfg.WEIGHT_DECAY, betas=cfg.ADAMW_BETAS
    )
    train_logger = TrainLogger(cfg.TRAIN_LOG_TXT, resume=resume_path is not None)

    train(
        model, train_loader, val_loader, train_ds, optimizer, device,
        train_logger, clearml_task, output_model, output_dest,
        clearml_quiet, clearml_upload_each_best, resume_path,
    )

    save_loss_plot(cfg.TRAIN_LOG_TXT, cfg.TRAIN_LOG_PLOT)
    save_recon_demo(model, val_df, device, cfg.RECON_DEMO_PNG)


if __name__ == "__main__":
    main()
