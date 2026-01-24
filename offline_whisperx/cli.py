from __future__ import annotations

import argparse
import os
from pathlib import Path

from .whisperx_cli import WhisperXCliConfig, run_whisperx

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="offline-whisperx",
        description=(
            "Thin wrapper around the upstream `whisperx` CLI. "
            "For the recommended Docker workflow, see README.md."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Run `whisperx` on a local file.")
    t.add_argument("--audio", required=True, type=Path, help="Path to an audio/video file.")
    t.add_argument("--outdir", default=Path("output"), type=Path, help="Output directory.")
    t.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "large-v2"), help="Whisper model name.")
    t.add_argument("--align-model", default=os.environ.get("ALIGN_MODEL", ""), help="Alignment model name (optional).")
    t.add_argument("--no-align", action="store_true", help="Disable alignment.")
    t.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")
    t.add_argument("--diarize-model", default=os.environ.get("DIARIZE_MODEL", ""), help="Diarization pipeline (optional).")
    t.add_argument("--language", default=None, help="Force language code (e.g. en, de). Default: auto.")
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use.")
    t.add_argument("--compute-type", default="auto", choices=["auto", "int8", "float16", "float32"], help="Compute type.")
    t.add_argument("--model-dir", default=os.environ.get("WHISPERX_MODEL_DIR", ""), help="WhisperX --model_dir (download_root).")

    return p

def cmd_transcribe(args: argparse.Namespace) -> int:
    cfg = WhisperXCliConfig(
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        model=args.model,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        model_cache_only=None,
        align_model=(args.align_model or None),
        no_align=args.no_align,
        diarize=args.diarize,
        diarize_model=(args.diarize_model or None),
        output_format="json",
    )

    run_whisperx(audio_path=args.audio, outdir=args.outdir, cfg=cfg)
    print(f"Done. Wrote outputs to: {args.outdir.resolve()}")
    return 0

def main() -> None:
    p = build_parser()
    args = p.parse_args()

    if args.cmd == "transcribe":
        raise SystemExit(cmd_transcribe(args))

if __name__ == "__main__":
    main()
