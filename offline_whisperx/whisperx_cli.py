from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WhisperXCliConfig:
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    model: str = "large-v2"
    model_dir: Path | None = None
    model_cache_only: bool | None = None
    align_model: str | None = None
    no_align: bool = False
    diarize: bool = False
    min_speakers: int | None = None
    max_speakers: int | None = None
    hf_token: str | None = None
    diarize_model: str | None = None
    output_format: str = "json"


def build_whisperx_args(audio_path: Path, outdir: Path, cfg: WhisperXCliConfig) -> list[str]:
    args: list[str] = []

    # WhisperX supports either `whisperx AUDIO --model ...` or `whisperx --model ... AUDIO`.
    # We put AUDIO last to match common Docker examples and to keep arg building simple.
    args.extend(["--model", cfg.model])
    if cfg.model_dir is not None:
        args.extend(["--model_dir", str(cfg.model_dir)])
    if cfg.model_cache_only is not None:
        args.extend(["--model_cache_only", "True" if cfg.model_cache_only else "False"])

    args.extend(["--device", cfg.device])
    args.extend(["--compute_type", cfg.compute_type])

    if cfg.language:
        args.extend(["--language", cfg.language])

    # Output
    args.extend(["--output_dir", str(outdir)])
    args.extend(["--output_format", cfg.output_format])

    # Alignment
    if cfg.no_align:
        args.append("--no_align")
    elif cfg.align_model:
        args.extend(["--align_model", cfg.align_model])

    # Diarization
    if cfg.diarize:
        args.append("--diarize")
        if cfg.hf_token:
            args.extend(["--hf_token", cfg.hf_token])
        if cfg.diarize_model:
            args.extend(["--diarize_model", cfg.diarize_model])
        if cfg.min_speakers is not None:
            args.extend(["--min_speakers", str(cfg.min_speakers)])
        if cfg.max_speakers is not None:
            args.extend(["--max_speakers", str(cfg.max_speakers)])

    args.append(str(audio_path))
    return args


def run_whisperx(
    *,
    audio_path: Path,
    outdir: Path,
    cfg: WhisperXCliConfig,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    audio_path = Path(audio_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.diarize and not (cfg.hf_token or os.environ.get("HF_TOKEN")):
        # WhisperX typically needs a token argument for diarization even if caches exist.
        raise RuntimeError("Diarization requested but no HF token provided (set HF_TOKEN)")

    merged_env = os.environ.copy()
    if extra_env:
        merged_env.update(extra_env)

    cmd: list[str] = ["whisperx", *build_whisperx_args(audio_path, outdir, cfg)]

    return subprocess.run(
        cmd,
        text=True,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
