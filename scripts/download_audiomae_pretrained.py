import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "checkpoints" / "audiomae_pretrained.pth"
URL = "https://drive.google.com/uc?id=1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu"
MIN_BYTES = 1_000_000_000


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file() and OUT.stat().st_size >= MIN_BYTES:
        print(OUT)
        return 0
    if OUT.is_file():
        OUT.unlink()

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    import gdown

    try:
        gdown.download(URL, str(OUT), quiet=True)
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        OUT.unlink(missing_ok=True)
        return 1

    if not OUT.is_file() or OUT.stat().st_size < MIN_BYTES:
        OUT.unlink(missing_ok=True)
        print("incomplete download", file=sys.stderr)
        return 1

    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
