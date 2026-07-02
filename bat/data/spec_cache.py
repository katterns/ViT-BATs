import hashlib
from pathlib import Path

import numpy as np

import config as cfg

SPEC_CACHE_DIR = cfg.CHECKPOINT_DIR / "spec_cache"


def spec_cache_path(wav_path, pulse_center) -> Path:
    p = Path(wav_path).resolve()
    st = p.stat()
    key = f"{cfg.SPEC_CACHE_VERSION}|{p}|{int(pulse_center)}|{st.st_mtime_ns}|{st.st_size}"
    digest = hashlib.sha1(key.encode()).hexdigest()
    return SPEC_CACHE_DIR / f"{digest}.npy"


def load_base_spec(wav_path, pulse_center):
    path = spec_cache_path(wav_path, pulse_center)
    if not path.is_file():
        return None
    return np.load(path)


def save_base_spec(wav_path, pulse_center, spec):
    path = spec_cache_path(wav_path, pulse_center)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(spec, dtype=np.float32))


def cache_stats(df):
    hits = sum(spec_cache_path(r.path, int(r.pulse_center)).is_file() for r in df.itertuples())
    return hits, len(df)
