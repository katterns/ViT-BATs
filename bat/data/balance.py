import numpy as np
import pandas as pd

FILE_KEY = ("species", "filename")


def cap_files_per_species(
    files_df: pd.DataFrame,
    cap: int,
    *,
    seed: int = 42,
    per_split: bool = True,
) -> pd.DataFrame:
    if cap <= 0:
        raise ValueError("file cap must be > 0")

    rng = np.random.default_rng(seed)
    group_cols = ["split", "species"] if per_split and "split" in files_df.columns else ["species"]
    parts: list[pd.DataFrame] = []

    for _, group in files_df.groupby(group_cols, sort=False):
        if len(group) <= cap:
            parts.append(group)
            continue
        idx = rng.choice(len(group), size=cap, replace=False)
        parts.append(group.iloc[idx])

    out = pd.concat(parts, ignore_index=True)
    sort_cols = [c for c in ("split", "species", "filename") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _resolve_pulse_cap(counts: pd.Series, *, cap: int | None, equal: bool) -> int:
    if equal:
        return int(counts.min())
    if cap is None or cap <= 0:
        raise ValueError("pulse cap must be > 0 when equal=False")
    return int(cap)


def balance_pulses_per_species(
    pulses_df: pd.DataFrame,
    *,
    cap: int | None = None,
    equal: bool = False,
    seed: int = 42,
    per_split: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if not equal and cap is None:
        return pulses_df.copy(), {"enabled": False}

    rng = np.random.default_rng(seed)
    group_cols = ["split", "species"] if per_split and "split" in pulses_df.columns else ["species"]
    caps: dict[tuple, int] = {}
    parts: list[pd.DataFrame] = []
    before: dict[str, int] = {}
    after: dict[str, int] = {}

    if per_split and "split" in pulses_df.columns:
        for split, split_df in pulses_df.groupby("split", sort=False):
            counts = split_df.groupby("species").size()
            n_cap = _resolve_pulse_cap(counts, cap=cap, equal=equal)
            for species, count in counts.items():
                caps[(split, species)] = n_cap
                before[f"{split}:{species}"] = int(count)
    else:
        counts = pulses_df.groupby("species").size()
        n_cap = _resolve_pulse_cap(counts, cap=cap, equal=equal)
        for species, count in counts.items():
            caps[("all", species)] = n_cap
            before[str(species)] = int(count)

    for key, group in pulses_df.groupby(group_cols, sort=False):
        if per_split and "split" in pulses_df.columns:
            split, species = key
            n = caps[(split, species)]
            label = f"{split}:{species}"
        else:
            species = key[0] if isinstance(key, tuple) else key
            n = caps[("all", species)]
            label = str(species)

        if len(group) <= n:
            parts.append(group)
            after[label] = len(group)
        else:
            idx = rng.choice(len(group), size=n, replace=False)
            sampled = group.iloc[idx]
            parts.append(sampled)
            after[label] = n

    out = pd.concat(parts, ignore_index=True)
    sort_cols = [c for c in ("split", "species", "filename", "pulse_center") if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    meta = {
        "enabled": True,
        "mode": "min_per_split" if equal else "cap",
        "cap": cap,
        "per_split": per_split,
        "seed": seed,
        "n_before": int(len(pulses_df)),
        "n_after": int(len(out)),
        "pulses_per_species_before": before,
        "pulses_per_species_after": after,
    }
    return out, meta


def files_from_pulses(pulses_df: pd.DataFrame, files_df: pd.DataFrame) -> pd.DataFrame:
    keys = pulses_df[list(FILE_KEY)].drop_duplicates()
    return files_df.merge(keys, on=list(FILE_KEY), how="inner").reset_index(drop=True)
