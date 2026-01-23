from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

def cmd_whisper(args) -> None:
    out = Path(args.models_dir) / args.repo_id.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded {args.repo_id} -> {out}")

def cmd_align(args) -> None:
    out = Path(args.models_dir) / args.repo_id.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Downloaded {args.repo_id} -> {out}")

def main() -> None:
    p = argparse.ArgumentParser(description="Prefetch (download) model repos into ./models.")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("whisper", help="Download a faster-whisper CTranslate2 model repo.")
    w.add_argument("--models-dir", default="models", help="Base models directory.")
    w.add_argument("--repo-id", default="Systran/faster-whisper-base", help="HF repo id (e.g. Systran/faster-whisper-base).")
    w.set_defaults(func=cmd_whisper)

    a = sub.add_parser("align", help="Download a wav2vec2 alignment model repo.")
    a.add_argument("--models-dir", default="models", help="Base models directory.")
    a.add_argument("--repo-id", default="facebook/wav2vec2-base-960h", help="HF repo id (e.g. facebook/wav2vec2-base-960h).")
    a.set_defaults(func=cmd_align)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
