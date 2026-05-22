import sys
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "checkpoints" / "BEATs_iter3.pt"
URL = (
    "https://huggingface.co/lpepino/beats_ckpts/resolve/"
    "a2ddb6b0411c39942ae144a6414872e14e5a4329/BEATs_iter3.pt"
)
MIN_BYTES = 300_000_000


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file() and OUT.stat().st_size >= MIN_BYTES:
        print(OUT)
        return 0
    if OUT.is_file():
        OUT.unlink()

    try:
        urllib.request.urlretrieve(URL, OUT)
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
