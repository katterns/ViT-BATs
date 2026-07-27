import hashlib
from pathlib import Path

import numpy as np

import config as cfg

SPEC_CACHE_DIR = cfg.CHECKPOINT_DIR / "spec_cache"


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    return cache_dir if cache_dir is not None else SPEC_CACHE_DIR


def _disk_dtype() -> np.dtype:
    name = getattr(cfg, "SPEC_CACHE_DTYPE", "float16")
    return np.dtype(name)


def spec_cache_path(wav_path, pulse_center, *, cache_dir: Path | None = None) -> Path:
    p = Path(wav_path).resolve()
    st = p.stat()
    key = f"{cfg.SPEC_CACHE_VERSION}|{p}|{int(pulse_center)}|{st.st_mtime_ns}|{st.st_size}"
    digest = hashlib.sha1(key.encode()).hexdigest()
    return _resolve_cache_dir(cache_dir) / f"{digest}.npy"


def load_base_spec(wav_path, pulse_center, *, cache_dir: Path | None = None):
    path = spec_cache_path(wav_path, pulse_center, cache_dir=cache_dir)
    if not path.is_file():
        return None
    arr = np.load(path)
    return np.ascontiguousarray(arr, dtype=np.float32)


def save_base_spec(wav_path, pulse_center, spec, *, cache_dir: Path | None = None):
    path = spec_cache_path(wav_path, pulse_center, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(spec, dtype=_disk_dtype()))


def cache_stats(df, *, cache_dir: Path | None = None):
    hits = sum(
        spec_cache_path(r.path, int(r.pulse_center), cache_dir=cache_dir).is_file()
        for r in df.itertuples()
    )
    return hits, len(df)


def iter_required_paths(df, *, cache_dir: Path | None = None):
    for r in df.itertuples():
        wav = Path(r.path)
        if not wav.is_file():
            continue
        yield spec_cache_path(r.path, int(r.pulse_center), cache_dir=cache_dir)


def prune_spec_cache(df, *, cache_dir: Path | None = None) -> int:
    root = _resolve_cache_dir(cache_dir)
    if not root.is_dir():
        return 0
    keep = set(iter_required_paths(df, cache_dir=cache_dir))
    removed = 0
    for npy in root.glob("*.npy"):
        if npy not in keep:
            npy.unlink(missing_ok=True)
            removed += 1
    return removed


def sync_spec_cache(df, *, cache_dir: Path | None = None, desc: str = "specs") -> dict:
    from bat.data.audio import precompute_specs

    root = _resolve_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    removed = prune_spec_cache(df, cache_dir=cache_dir)
    built = precompute_specs(df, desc=desc, cache_dir=cache_dir)
    hits, total = cache_stats(df, cache_dir=cache_dir)
    if hits < total:
        raise RuntimeError(f"spec cache incomplete after sync: {hits}/{total} in {root}")
    return {
        "removed": removed,
        "built": built,
        "cached": hits,
        "total": total,
        "dir": str(root),
    }
