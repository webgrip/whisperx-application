from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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


def redact_cmd(argv: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            redacted.append("***")
            skip_next = False
            continue
        redacted.append(item)
        if item == "--hf_token":
            skip_next = True
    return redacted


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
    log_paths: list[Path] | None = None,
    line_callback: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    audio_path = Path(audio_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged_env = os.environ.copy()
    if extra_env:
        merged_env.update(extra_env)

    if cfg.diarize and not cfg.hf_token:
        # Upstream WhisperX CLI expects --hf_token for diarization. It does not reliably
        # auto-read tokens from the environment for the diarization pipeline.
        token = merged_env.get("HF_TOKEN") or merged_env.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(
                "Diarization requested but no HF token provided (set HF_TOKEN or HUGGINGFACE_HUB_TOKEN)"
            )
        cfg = replace(cfg, hf_token=token)

    cmd: list[str] = ["whisperx", *build_whisperx_args(audio_path, outdir, cfg)]

    if log_paths:
        log_files = []
        for p in log_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            log_files.append(p.open("a", encoding="utf-8"))

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                text=True,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                for f in log_files:
                    f.write(line)
                    f.flush()
                # Also surface in container logs for `docker compose logs -f whisperx-worker`.
                sys.stdout.write(line)
                sys.stdout.flush()
                if line_callback:
                    line_callback(line.rstrip("\n"))

            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, redact_cmd(cmd))
            return subprocess.CompletedProcess(cmd, rc, stdout=f"(streamed) elapsed={time.monotonic() - started:.1f}s")
        finally:
            for f in log_files:
                try:
                    f.close()
                except Exception:
                    pass

    try:
        return subprocess.run(
            cmd,
            text=True,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # Avoid leaking tokens into logs/tracebacks.
        raise subprocess.CalledProcessError(
            e.returncode,
            redact_cmd(list(e.cmd) if isinstance(e.cmd, (list, tuple)) else cmd),
            output=e.stdout,
            stderr=e.stderr,
        ) from None
