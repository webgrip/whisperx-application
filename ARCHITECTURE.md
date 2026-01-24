# Architecture

This repository is a **cache-first** WhisperX runner in two modes:

1) CLI mode: run `whisperx` inside the provided Docker profile
2) Service mode: HTTP API (FastAPI) + Redis queue + GPU worker

The key design constraint is: **WhisperX and pyannote run from a public container image and use a shared on-disk cache** so you can warm it once and reuse it later.

## Components

### Python package: `offline_whisperx/`

- `cli.py`
  - Thin wrapper around the upstream `whisperx` CLI (for non-Docker usage).

- `whisperx_cli.py`
  - Runs WhisperX via subprocess (`whisperx ...`) and passes cache/config options.

- `chunking.py`
  - Splits long audio into overlapping WAV chunks via `ffmpeg` and merges results by offsetting timestamps.

- `api/main.py`
  - FastAPI app (stateless):
    - `GET /health` (includes Redis reachability)
    - `POST /jobs/transcribe` to enqueue a durable job
    - `GET /jobs/{job_id}` and `GET /jobs/{job_id}/result`

- `jobs.py`
  - The job function executed by the worker. Runs WhisperX CLI (per-chunk for long audio) and writes `transcript.json`.

## Repository layout

- `cache/` (gitignored)
  - Hugging Face + pyannote caches. Mounts into the worker at `/.cache`.

- `data/` (optional)
  - Place input files here for CLI/container usage.

- `output/`
  - Output artifacts. In service mode this also stores job state and results.

## Runtime data flow

### CLI

`offline-whisperx transcribe` → `whisperx ...` (subprocess) → outputs written to `outdir`.

### Service

Upload file → save to `/app/data/uploads` → enqueue job in Redis → worker consumes job →
(optional) chunk audio → transcribe each chunk → merge → write `/app/output/jobs/<job_id>/transcript.json`.

## Best-practice operational notes

- Keep `cache/` out of git.
- Warm `cache/` on an online machine, then copy it to the offline machine.
- For long audio, chunking avoids timeouts and memory spikes, but diarization speaker IDs may be less stable across chunks.
