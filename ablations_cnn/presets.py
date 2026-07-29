from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import config as cfg

TASK_ORDER = ("mae", "con", "sep", "jig")

ABLATION_ROOT = cfg.CHECKPOINT_DIR / "ablations_cnn"
SSL_CKPT_DIR = ABLATION_ROOT / "ssl"
FT_CKPT_DIR = ABLATION_ROOT / "finetune"
REPORT_PATH = ABLATION_ROOT / "ablation_results.csv"


@dataclass(frozen=True)
class TaskSet:
    mae: bool = False
    con: bool = False
    sep: bool = False
    jig: bool = False

    @property
    def id(self):
        parts = [name for name in TASK_ORDER if getattr(self, name)]
        return "+".join(parts)

    def __getitem__(self, task):
        if task not in TASK_ORDER:
            raise KeyError(task)
        return getattr(self, task)

    @classmethod
    def from_names(cls, names):
        unknown = set(names) - set(TASK_ORDER)
        if unknown:
            raise ValueError(f"unknown tasks: {sorted(unknown)}")
        return cls(**{name: name in names for name in TASK_ORDER})


def _build_all_presets():
    presets = []
    for size in range(1, len(TASK_ORDER) + 1):
        for combo in combinations(TASK_ORDER, size):
            presets.append(TaskSet.from_names(list(combo)))
    return tuple(presets)


ALL_PRESETS = _build_all_presets()


def parse_preset(value):
    key = value.strip().lower().replace(",", "+")
    for preset in ALL_PRESETS:
        if preset.id == key:
            return preset
    valid = ", ".join(p.id for p in ALL_PRESETS)
    raise ValueError(f"unknown preset {value!r}; expected one of: {valid}")


def ssl_ckpt_path(preset):
    return SSL_CKPT_DIR / f"{preset.id}_best.pt"


def _ft_stem(preset, tag=""):
    stem = preset.id
    return f"{stem}_{tag}" if tag else stem


def ft_ckpt_path(preset, tag=""):
    return FT_CKPT_DIR / f"{_ft_stem(preset, tag)}_best.pt"


def ssl_run_name(preset):
    return f"ablations_cnn/ssl_{preset.id}"


def ft_run_name(preset, tag=""):
    return f"ablations_cnn/ft_{_ft_stem(preset, tag)}"


def confusion_path(preset, tag=""):
    return ABLATION_ROOT / "confusion" / f"{_ft_stem(preset, tag)}_confusion_matrix.png"


def ssl_monitor(preset):
    """Метрика early stopping: одна задача → её val_*; комбинации → val_loss."""
    if sum(preset[t] for t in TASK_ORDER) == 1:
        if preset.mae:
            return "val_recon"
        if preset.con:
            return "val_con"
        if preset.sep:
            return "val_sep"
        if preset.jig:
            return "val_jig"
    return "val_loss"


def use_recording_loader(preset):
    """Same-recording pairs нужны только если con вместе с другими задачами."""
    return preset.con and (preset.mae or preset.sep or preset.jig)
