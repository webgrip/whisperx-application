from __future__ import annotations

import argparse
from pathlib import Path
import os

from rich.console import Console
from rich.traceback import install as rich_traceback

from .config import TranscriptionConfig
from .offline import enforce_offline
from .transcriber import Transcriber

console = Console()

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="offline-whisperx", description="WhisperX offline starter CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="Transcribe an audio file (optionally align + diarize).")
    t.add_argument("--audio", required=True, type=Path, help="Path to an audio/video file.")
    t.add_argument("--outdir", default=Path("output"), type=Path, help="Output directory.")
    t.add_argument("--whisper-model", default=Path("models/faster-whisper-base"), type=Path, help="Local faster-whisper model directory.")
    t.add_argument("--align-model", default=Path("models/wav2vec2-base-960h"), type=Path, help="Local alignment model directory.")
    t.add_argument("--no-align", action="store_true", help="Disable word-level alignment.")
    t.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")
    t.add_argument("--pyannote-config", default=Path("config/pyannote_diarization_config.yaml"), type=Path, help="Local pyannote diarization config YAML.")
    t.add_argument("--language", default=None, help="Force language code (e.g. en, de). Default: auto.")
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to use.")
    t.add_argument("--compute-type", default="auto", choices=["auto", "int8", "float16", "float32"], help="Compute type.")
    t.add_argument("--hf-home", default=Path("models/hf_cache"), type=Path, help="HF cache directory.")
    t.add_argument("--online", action="store_true", help="Allow network access (disables offline enforcement).")
    t.add_argument("--json", action="store_true", help="Write transcript.json.")
    t.add_argument("--srt", action="store_true", help="Write transcript.srt.")
    t.add_argument("--vtt", action="store_true", help="Write transcript.vtt.")

    return p

def cmd_transcribe(args: argparse.Namespace) -> int:
    enforce = (not args.online) and (os.environ.get("OFFLINE_WHISPERX_ENFORCE_OFFLINE", "1") != "0")
    if enforce:
        enforce_offline(args.hf_home)

    cfg = TranscriptionConfig(
        whisper_model=args.whisper_model,
        align_model=None if args.no_align else args.align_model,
        pyannote_config=args.pyannote_config,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        do_align=not args.no_align,
        do_diarize=args.diarize,
        hf_home=args.hf_home,
        enforce_offline=enforce,
        outdir=args.outdir,
        write_json=(args.json or (not args.srt and not args.vtt)),
        write_srt=args.srt,
        write_vtt=args.vtt,
    )

    transcriber = Transcriber(cfg)
    result = transcriber.transcribe(args.audio)
    console.print(f"[bold green]Done.[/bold green] Wrote outputs to: {cfg.outdir.resolve()}")
    if isinstance(result, dict) and "_align_error" in result:
        console.print(f"[yellow]Alignment warning:[/yellow] {result['_align_error']}")
    if isinstance(result, dict) and "_diarize_error" in result:
        console.print(f"[yellow]Diarization warning:[/yellow] {result['_diarize_error']}")
    return 0

def main() -> None:
    rich_traceback(show_locals=False)
    p = build_parser()
    args = p.parse_args()

    if args.cmd == "transcribe":
        raise SystemExit(cmd_transcribe(args))

if __name__ == "__main__":
    main()
