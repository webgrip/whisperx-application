from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _get_whisperx_vad_url() -> str:
    import whisperx.vad as vad

    url = getattr(vad, "VAD_SEGMENTATION_URL", None)
    if not isinstance(url, str) or not url:
        raise RuntimeError("Could not locate whisperx.vad.VAD_SEGMENTATION_URL")
    return url


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Prefetch WhisperX VAD model into the mounted models directory. "
            "This prevents whisperx from downloading VAD at runtime."
        )
    )
    p.add_argument("--models-dir", default="models", help="Base models directory on host.")
    p.add_argument(
        "--out",
        default="vad/whisperx_vad.bin",
        help="Path under --models-dir to write the VAD blob.",
    )

    args = p.parse_args()

    models_dir = Path(args.models_dir)
    dest = models_dir / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = _get_whisperx_vad_url()

    # Use curl for robust redirect handling.
    # -f: fail on HTTP errors
    # -sS: quiet but show errors
    # -L: follow redirects
    cmd = ["curl", "-f", "-sS", "-L", "-o", str(dest), url]
    subprocess.run(cmd, check=True)

    print(f"Downloaded WhisperX VAD -> {dest}")
    print(f"Set VAD_FILE=/app/models/{args.out} (or use .env.example)")


if __name__ == "__main__":
    main()
