import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 192_000
CLIP_MS = 50
CLIP_SAMPLES = SAMPLE_RATE * CLIP_MS // 1000
DATA_FILE = "clips.f16"
KEYS_FILE = "keys.npy"
META_FILE = "meta.json"


def waveform_key(path, pulse_center):
    value = f"{Path(path).resolve()}\0{int(pulse_center)}"
    return hashlib.sha1(value.encode()).hexdigest()


def _extract_clip_from_file(wav, path, pulse_center):
    start_sec = int(pulse_center) / 1000.0 - CLIP_MS / 1000.0
    if start_sec < 0:
        raise ValueError(f"pulse_center={pulse_center} is before a full {CLIP_MS} ms clip")

    source_rate = int(wav.samplerate)
    start_frame = int(np.floor(start_sec * source_rate))
    end_sec = int(pulse_center) / 1000.0
    end_frame = int(np.ceil(end_sec * source_rate)) + 1
    if start_frame < 0 or end_frame > len(wav):
        raise ValueError(
            f"clip [{start_sec:.6f}, {end_sec:.6f}] is outside {path} "
            f"(duration={len(wav) / source_rate:.6f})"
        )
    wav.seek(start_frame)
    source = wav.read(end_frame - start_frame, dtype="float32", always_2d=True).mean(axis=1)

    source_positions = (start_frame + np.arange(source.size, dtype=np.float64)) / source_rate
    target_positions = start_sec + np.arange(CLIP_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    return np.interp(target_positions, source_positions, source).astype(np.float32)


def build_waveform_cache(df, cache_dir):
    frame = df.reset_index(drop=True)
    keys = np.asarray(
        [waveform_key(row.path, row.pulse_center) for row in frame.itertuples()],
        dtype="S40",
    )
    if np.unique(keys).size != keys.size:
        raise ValueError("waveform cache input contains duplicate path/pulse_center pairs")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_tmp = cache_dir / f"{DATA_FILE}.tmp"
    keys_tmp = cache_dir / f"{KEYS_FILE}.tmp"
    meta_tmp = cache_dir / f"{META_FILE}.tmp"

    clips = np.memmap(
        data_tmp,
        mode="w+",
        dtype=np.float16,
        shape=(len(frame), CLIP_SAMPLES),
    )
    completed = 0
    for path, group in frame.groupby("path", sort=False):
        with sf.SoundFile(path) as wav:
            for row_idx, row in group.iterrows():
                clips[row_idx] = _extract_clip_from_file(wav, path, row["pulse_center"])
                completed += 1
                if completed == 1 or completed % 1000 == 0 or completed == len(frame):
                    clips.flush()
                    print(f"waveform cache: {completed}/{len(frame)}", flush=True)
    clips.flush()
    del clips

    with keys_tmp.open("wb") as f:
        np.save(f, keys)
    meta_tmp.write_text(
        json.dumps(
            {
                "count": len(frame),
                "sample_rate": SAMPLE_RATE,
                "clip_ms": CLIP_MS,
                "clip_samples": CLIP_SAMPLES,
                "dtype": "float16",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    data_tmp.replace(cache_dir / DATA_FILE)
    keys_tmp.replace(cache_dir / KEYS_FILE)
    meta_tmp.replace(cache_dir / META_FILE)


class WaveformClipCache:
    def __init__(self, df, cache_dir):
        cache_dir = Path(cache_dir)
        meta_path = cache_dir / META_FILE
        data_path = cache_dir / DATA_FILE
        keys_path = cache_dir / KEYS_FILE
        for path in (meta_path, data_path, keys_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"waveform cache is incomplete: missing {path}. "
                    "Run: uv run python scripts/precompute_waveforms.py"
                )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {
            "sample_rate": SAMPLE_RATE,
            "clip_ms": CLIP_MS,
            "clip_samples": CLIP_SAMPLES,
            "dtype": "float16",
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"waveform cache {key}={meta.get(key)!r}, expected {value!r}")

        keys = np.load(keys_path)
        count = int(meta["count"])
        if len(keys) != count:
            raise ValueError(f"waveform cache has {len(keys)} keys for {count} clips")
        expected_bytes = count * CLIP_SAMPLES * np.dtype(np.float16).itemsize
        if data_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"waveform cache data size is {data_path.stat().st_size}, expected {expected_bytes}"
            )

        self._rows = {key.decode(): i for i, key in enumerate(keys)}
        if len(self._rows) != count:
            raise ValueError("waveform cache contains duplicate keys")
        self._clips = np.memmap(
            data_path,
            mode="r",
            dtype=np.float16,
            shape=(count, CLIP_SAMPLES),
        )

        missing = [
            waveform_key(row.path, row.pulse_center)
            for row in df.itertuples()
            if waveform_key(row.path, row.pulse_center) not in self._rows
        ]
        if missing:
            raise KeyError(f"waveform cache misses {len(missing)} requested clips")

    def load(self, path, pulse_center):
        key = waveform_key(path, pulse_center)
        return np.asarray(self._clips[self._rows[key]], dtype=np.float32)
