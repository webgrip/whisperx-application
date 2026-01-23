# Architecture

This repository is an **offline-first** WhisperX runner that can be used in two modes:

1) CLI mode: `offline-whisperx transcribe ...`
2) Service mode: an internal-network HTTP API (FastAPI) + Redis queue + GPU worker

The key design constraint is: **models are prefetched elsewhere and mounted locally** so runtime never needs to download.

## Components

### Python package: `offline_whisperx/`

- `cli.py`
  - Thin CLI wrapper that builds a `TranscriptionConfig` and calls `Transcriber`.

- `config.py`
  - Pydantic model defining the core configuration: model paths, device/compute type, diarization on/off, output paths.

- `offline.py`
  - Sets environment variables (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, etc.) to fail-fast if anything tries to fetch.

- `transcriber.py`
  - The core “pipeline runner”:
    - loads WhisperX ASR model (faster-whisper)
    - transcribes audio
    - optionally aligns words
    - optionally diarizes and assigns speakers
    - writes outputs (JSON/SRT/VTT)

- `diarization.py`
  - Loads a pyannote pipeline from a local YAML and runs diarization.

- `chunking.py`
  - Splits long audio into overlapping WAV chunks via `ffmpeg` and merges results by offsetting timestamps.

- `api/main.py`
  - FastAPI app (stateless):
    - `GET /health` (includes Redis reachability)
    - `POST /jobs/transcribe` to enqueue a durable job
    - `GET /jobs/{job_id}` and `GET /jobs/{job_id}/result`

- `jobs.py`
  - The job function executed by the worker (writes outputs, updates progress meta).

## Repository layout

- `config/`
  - Committed configuration files (safe to keep in git).
  - Includes `config/pyannote_diarization_config.yaml`.

- `models/` (gitignored)
  - Local model artifacts (large binaries) and HF cache.
  - Mount this into containers at `/app/models`.

- `data/` (optional)
  - Place input files here for CLI/container usage.

- `output/`
  - Output artifacts. In service mode this also stores job state and results.

## Runtime data flow

### CLI

`offline-whisperx transcribe` → `Transcriber.transcribe(audio_path)` → outputs written to `outdir`.

### Service

Upload file → save to `/app/data/uploads` → enqueue job in Redis → worker consumes job →
(optional) chunk audio → transcribe each chunk → merge → write `/app/output/jobs/<job_id>/transcript.json`.

## Best-practice operational notes

- Keep `models/` out of git. Only configs go in git.
- For GPU containers, use a CUDA-enabled PyTorch base image and run with `gpus: all`.
- For long audio, always use job + chunking to avoid request timeouts and memory spikes.
- For diarization on multi-hour chunked jobs, expect speaker IDs to be less stable across chunks.
