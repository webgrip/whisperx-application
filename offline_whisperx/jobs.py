from __future__ import annotations

import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from rq import get_current_job

from .chunking import iter_audio_chunks, offset_transcript_timestamps
from .util import ensure_dir, write_json
from .whisperx_cli import WhisperXCliConfig, run_whisperx


def _job_meta_update(**meta: Any) -> None:
    job = get_current_job()
    if job is None:
        return
    job.meta.update(meta)
    job.save_meta()


def transcribe_job(
    *,
    job_id: str,
    input_path: str,
    outdir: str,
    whisper_model: str,
    align_model: str,
    diarize_model: str,
    device: str,
    compute_type: str,
    language: str | None,
    do_align: bool,
    do_diarize: bool,
    model_dir: str,
    pyannote_cache: str,
    chunk_seconds: float | None,
    overlap_seconds: float,
) -> dict[str, Any]:
    """Run a transcription job.

    This function is executed by the RQ worker.
    It writes outputs under `outdir` and returns a small JSON-serializable summary.
    """

    started_at = datetime.utcnow().isoformat() + "Z"

    outdir_path = Path(outdir)
    ensure_dir(outdir_path)

    # WhisperX CLI runner config
    cfg = WhisperXCliConfig(
        device=device,
        compute_type=compute_type,
        language=language,
        model=whisper_model,
        model_dir=Path(model_dir) if model_dir else None,
        model_cache_only=None,
        align_model=(align_model or None),
        no_align=(not do_align),
        diarize=do_diarize,
        hf_token=None,  # read from env (HF_TOKEN) unless explicitly provided
        diarize_model=(diarize_model or None),
        output_format="json",
    )

    input_path_p = Path(input_path)
    _job_meta_update(status="running", progress=0.0, detail="Starting", started_at=started_at)

    global_log = outdir_path / "whisperx.log"
    ensure_dir(global_log.parent)

    if chunk_seconds is None:
        out_wx = outdir_path / "whisperx"
        ensure_dir(out_wx)

        last_meta = 0.0

        def on_line(line: str) -> None:
            nonlocal last_meta
            now = time.monotonic()
            if now - last_meta >= 2.0 and line.strip():
                _job_meta_update(detail=f"WhisperX: {line.strip()[:240]}")
                last_meta = now

        _job_meta_update(detail="Running WhisperX")
        run_whisperx(
            audio_path=input_path_p,
            outdir=out_wx,
            cfg=cfg,
            extra_env={
                "PYANNOTE_CACHE": pyannote_cache,
            },
            log_paths=[global_log],
            line_callback=on_line,
        )

        # WhisperX writer names output files based on input audio basename.
        expected = out_wx / f"{input_path_p.stem}.json"
        if expected.exists():
            result = json.loads(expected.read_text(encoding="utf-8"))
        else:
            candidates = sorted(out_wx.glob("*.json"))
            if not candidates:
                raise RuntimeError("WhisperX did not produce a JSON output file")
            result = json.loads(candidates[0].read_text(encoding="utf-8"))

        write_json(outdir_path / "transcript.json", result)
        _job_meta_update(progress=1.0, detail="Complete")
        return {"job_id": job_id, "result_path": str(outdir_path / "transcript.json")}

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and < chunk_seconds")

    chunk_work = outdir_path / "chunks"
    ensure_dir(chunk_work)

    combined: dict[str, Any] = {
        "segments": [],
        "_chunk_seconds": chunk_seconds,
        "_overlap_seconds": overlap_seconds,
        "_note": "Chunked transcription. Overlap is used to reduce boundary dropouts; early-overlap segments are trimmed to reduce duplicates.",
    }

    chunks = list(
        iter_audio_chunks(
            input_path=input_path_p,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            work_dir=chunk_work,
        )
    )

    for idx, chunk in enumerate(chunks):
        _job_meta_update(
            progress=min(0.99, idx / max(1, len(chunks))),
            detail=f"Transcribing chunk {idx + 1}/{len(chunks)} ({chunk.start_s:.1f}s–{chunk.end_s:.1f}s)",
        )

        # Per-chunk WhisperX run. This is slower than an in-process pipeline, but it
        # keeps behavior aligned with upstream WhisperX and the cache-based offline flow.
        chunk_out = chunk_work / f"out_{idx:04d}"
        ensure_dir(chunk_out)

        chunk_log = chunk_out / "whisperx.log"
        last_meta = 0.0

        def on_line(line: str) -> None:
            nonlocal last_meta
            now = time.monotonic()
            if now - last_meta >= 2.0 and line.strip():
                _job_meta_update(detail=f"WhisperX: {line.strip()[:240]}")
                last_meta = now

        _job_meta_update(detail=f"Running WhisperX chunk {idx + 1}/{len(chunks)}")
        run_whisperx(
            audio_path=chunk.path,
            outdir=chunk_out,
            cfg=cfg,
            extra_env={
                "PYANNOTE_CACHE": pyannote_cache,
            },
            log_paths=[chunk_log, global_log],
            line_callback=on_line,
        )

        expected = chunk_out / f"{chunk.path.stem}.json"
        if expected.exists():
            res = json.loads(expected.read_text(encoding="utf-8"))
        else:
            candidates = sorted(chunk_out.glob("*.json"))
            if not candidates:
                raise RuntimeError("WhisperX did not produce a JSON output file for a chunk")
            res = json.loads(candidates[0].read_text(encoding="utf-8"))

        trim = 0.0
        if overlap_seconds > 0 and idx > 0:
            trim = overlap_seconds / 2.0

        res2 = offset_transcript_timestamps(res, offset_s=chunk.start_s, trim_start_s=trim)
        segs = res2.get("segments") if isinstance(res2, dict) else None
        if isinstance(segs, list):
            combined["segments"].extend(segs)

    write_json(outdir_path / "transcript.json", combined)
    _job_meta_update(progress=1.0, detail="Complete")
    return {"job_id": job_id, "result_path": str(outdir_path / "transcript.json")}
