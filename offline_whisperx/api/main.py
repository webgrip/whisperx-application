from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from rq.job import Job

from ..jobs import transcribe_job
from ..settings import ServiceSettings
from ..util import ensure_dir
from ..queue import get_redis


class HealthResponse(BaseModel):
    status: str = "ok"
    redis: str = "unknown"


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    enqueued_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    detail: str | None = None


class SyncTranscribeResponse(BaseModel):
    job_id: str
    result: dict[str, Any]


SETTINGS = ServiceSettings()

app = FastAPI(title="offline-whisperx", version="0.1.0")


def _uploads_dir() -> Path:
    ensure_dir(SETTINGS.uploads_dir)
    return SETTINGS.uploads_dir


def _jobs_dir() -> Path:
    ensure_dir(SETTINGS.jobs_dir)
    return SETTINGS.jobs_dir


def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _queue(r: Redis) -> Queue:
    return Queue(name=SETTINGS.queue_name, connection=r)


def _enqueue_transcription(
    *,
    file: UploadFile,
    chunk_seconds: float | None,
    overlap_seconds: float,
    do_align: bool,
    do_diarize: bool,
) -> tuple[str, Job, Path]:
    job_id = str(uuid4())

    uploads = _uploads_dir()
    suffix = Path(file.filename or "upload").suffix
    input_path = uploads / f"{job_id}{suffix}"
    _save_upload(file, input_path)

    outdir = _jobs_dir() / job_id
    ensure_dir(outdir)

    r = get_redis(SETTINGS)
    q = _queue(r)

    job = q.enqueue(
        transcribe_job,
        kwargs={
            "job_id": job_id,
            "input_path": str(input_path),
            "outdir": str(outdir),
            "whisper_model": SETTINGS.whisper_model,
            "align_model": SETTINGS.align_model or "",
            "diarize_model": SETTINGS.diarize_model or "",
            "device": SETTINGS.device,
            "compute_type": SETTINGS.compute_type,
            "language": SETTINGS.language,
            "do_align": do_align,
            "do_diarize": do_diarize,
            "model_dir": str(SETTINGS.model_dir),
            "pyannote_cache": str(SETTINGS.pyannote_cache),
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
        },
        job_id=job_id,
        result_ttl=7 * 24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )

    # seed initial meta
    job.meta.update({"progress": 0.0, "detail": "Queued", "enqueued_at": datetime.utcnow().isoformat() + "Z"})
    job.save_meta()

    return job_id, job, outdir


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    r = get_redis(SETTINGS)
    try:
        r.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "error"
    return HealthResponse(status="ok", redis=redis_status)


@app.post("/jobs/transcribe", response_model=JobCreateResponse)
def create_job(
    file: UploadFile = File(...),
    chunk_seconds: float | None = 1800.0,
    overlap_seconds: float = 10.0,
    do_align: bool = True,
    do_diarize: bool = False,
) -> JobCreateResponse:
    job_id, job, _outdir = _enqueue_transcription(
        file=file,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        do_align=do_align,
        do_diarize=do_diarize,
    )

    return JobCreateResponse(job_id=job_id, status=job.get_status())


@app.post("/transcribe", response_model=SyncTranscribeResponse)
def transcribe_sync(
    file: UploadFile = File(...),
    chunk_seconds: float | None = 1800.0,
    overlap_seconds: float = 10.0,
    do_align: bool = True,
    do_diarize: bool = False,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.5,
) -> SyncTranscribeResponse:
    """Synchronous transcription endpoint.

    Enqueues the transcription job and blocks until it completes (or `timeout_seconds`).
    This keeps GPU/WhisperX execution in the worker container but gives clients a
    synchronous HTTP workflow.
    """

    if timeout_seconds <= 0:
        raise HTTPException(status_code=400, detail="timeout_seconds must be > 0")
    if poll_interval_seconds <= 0:
        raise HTTPException(status_code=400, detail="poll_interval_seconds must be > 0")

    job_id, _job, outdir = _enqueue_transcription(
        file=file,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        do_align=do_align,
        do_diarize=do_diarize,
    )

    r = get_redis(SETTINGS)
    deadline = time.monotonic() + float(timeout_seconds)

    while True:
        try:
            job = Job.fetch(job_id, connection=r)
        except Exception:
            raise HTTPException(status_code=500, detail="Job disappeared")

        status = job.get_status()
        if status in {"queued", "started", "deferred", "scheduled"}:
            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=504,
                    detail={
                        "error": "Timed out waiting for transcription",
                        "job_id": job_id,
                        "status": status,
                    },
                )
            time.sleep(poll_interval_seconds)
            continue

        if status == "failed":
            log_path = outdir / "whisperx.log"
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Job failed",
                    "job_id": job_id,
                    "log_path": str(log_path),
                },
            )

        # finished
        result = job.return_value()
        if not isinstance(result, dict) or "result_path" not in result:
            raise HTTPException(status_code=500, detail={"error": "Job result missing", "job_id": job_id})

        result_path = Path(result["result_path"])
        if not result_path.exists():
            raise HTTPException(status_code=500, detail={"error": "Result file not found", "job_id": job_id})

        return SyncTranscribeResponse(job_id=job_id, result=json.loads(result_path.read_text(encoding="utf-8")))


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    r = get_redis(SETTINGS)
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    meta = job.meta or {}
    return JobStatusResponse(
        job_id=job.id,
        status=job.get_status(),
        enqueued_at=(job.enqueued_at.isoformat() + "Z") if job.enqueued_at else meta.get("enqueued_at"),
        started_at=(job.started_at.isoformat() + "Z") if job.started_at else meta.get("started_at"),
        ended_at=(job.ended_at.isoformat() + "Z") if job.ended_at else None,
        progress=float(meta.get("progress") or 0.0),
        detail=meta.get("detail"),
    )


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str) -> dict[str, Any]:
    r = get_redis(SETTINGS)
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()
    if status in {"queued", "started", "deferred", "scheduled"}:
        raise HTTPException(status_code=409, detail="Job not complete")
    if status == "failed":
        raise HTTPException(status_code=500, detail="Job failed")

    result = job.return_value()
    if not isinstance(result, dict) or "result_path" not in result:
        raise HTTPException(status_code=500, detail="Job result missing")

    result_path = Path(result["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=500, detail="Result file not found")

    return {"job_id": job.id, "result": json.loads(result_path.read_text(encoding="utf-8"))}
