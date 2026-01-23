# WhisperX (offline) — How to run

This repository runs WhisperX transcription (optionally diarization) **fully offline** once you’ve staged models under `models/`.

If you want the detailed model download + ops runbook, see: RUNBOOK.md.

## Prereqs

- Docker + Docker Compose
- NVIDIA driver + NVIDIA Container Toolkit (GPU worker)
- Models staged on the host and mounted to `/app/models`
  - Download instructions: RUNBOOK.md

## Run (service on your LAN)

Start the durable queue + API + GPU worker:

```bash
docker compose up -d --build whisperx-api whisperx-worker
```

Health check:

```bash
curl http://<host>:8000/health
```

Submit a transcription job (recommended for multi-hour audio):

```bash
curl -F "file=@/path/to/audio.wav" \
  "http://<host>:8000/jobs/transcribe?chunk_seconds=1800&overlap_seconds=10&do_align=true&do_diarize=false"
```

Poll status:

```bash
curl "http://<host>:8000/jobs/<job_id>"
```

Fetch result:

```bash
curl "http://<host>:8000/jobs/<job_id>/result"
```

## Volumes (recommended host layout)

- `/opt/whisperx/models` → `/app/models`
- `/opt/whisperx/output` → `/app/output`

## Optional: CLI container

```bash
docker compose --profile cli run --rm whisperx-cli
```

## More docs

- Detailed runbook: RUNBOOK.md
- Architecture overview: ARCHITECTURE.md
- Pyannote pipeline config: config/pyannote_diarization_config.yaml

