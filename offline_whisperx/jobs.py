from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rq import get_current_job

from .chunking import iter_audio_chunks, offset_transcript_timestamps
from .config import TranscriptionConfig
from .offline import enforce_offline
from .transcriber import Transcriber
from .util import ensure_dir, write_json


_TRANSCRIBER_CACHE: dict[tuple[str, str, str, str, str, str | None], Transcriber] = {}


def _job_meta_update(**meta: Any) -> None:
    job = get_current_job()
    if job is None:
        return
    job.meta.update(meta)
    job.save_meta()


def _get_transcriber(cfg: TranscriptionConfig) -> Transcriber:
    key = (
        str(cfg.whisper_model),
        str(cfg.align_model) if cfg.align_model is not None else "",
        str(cfg.pyannote_config),
        cfg.device,
        cfg.compute_type,
        cfg.language,
    )
    t = _TRANSCRIBER_CACHE.get(key)
    if t is None:
        t = Transcriber(cfg)
        _TRANSCRIBER_CACHE[key] = t
    return t


def transcribe_job(
    *,
    job_id: str,
    input_path: str,
    outdir: str,
    whisper_model: str,
    align_model: str,
    pyannote_config: str,
    device: str,
    compute_type: str,
    language: str | None,
    do_align: bool,
    do_diarize: bool,
    hf_home: str,
    offline_enforce: bool,
    chunk_seconds: float | None,
    overlap_seconds: float,
) -> dict[str, Any]:
    """Run a transcription job.

    This function is executed by the RQ worker.
    It writes outputs under `outdir` and returns a small JSON-serializable summary.
    """

    started_at = datetime.utcnow().isoformat() + "Z"

    if offline_enforce:
        enforce_offline(hf_home)

    outdir_path = Path(outdir)
    ensure_dir(outdir_path)

    cfg = TranscriptionConfig(
        whisper_model=Path(whisper_model),
        align_model=Path(align_model) if align_model else None,
        pyannote_config=Path(pyannote_config),
        device=device,
        compute_type=compute_type,
        language=language,
        do_align=do_align,
        do_diarize=do_diarize,
        hf_home=Path(hf_home),
        enforce_offline=offline_enforce,
        outdir=outdir_path,
        write_json=True,
        write_srt=False,
        write_vtt=False,
    )

    transcriber = _get_transcriber(cfg)

    input_path_p = Path(input_path)
    _job_meta_update(status="running", progress=0.0, detail="Starting", started_at=started_at)

    if chunk_seconds is None:
        result = transcriber.transcribe(input_path_p)
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

        res = transcriber.transcribe(chunk.path)

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
