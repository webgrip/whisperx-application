# WhisperX (cache-based offline) — How to run

This repo is now a thin **API + durable job queue** that runs the upstream **WhisperX CLI** inside a **public Docker image**, using a **shared cache directory**.

It matches the workflow discussed in WhisperX issue #873: warm the cache once while you have internet, then reuse it offline later. If the network is available and something is missing, WhisperX will download it into `cache/`.

## What you get

- `whisperx-api`: FastAPI endpoint to upload audio and enqueue a job
- `whisperx-worker`: GPU worker built FROM `ghcr.io/jim60105/whisperx:no_model`, executes `whisperx ...` and writes JSON results
- `./cache/`: persistent Hugging Face + pyannote cache (portable to an offline machine)

## Prereqs

- Docker + Docker Compose
- NVIDIA driver + NVIDIA Container Toolkit (GPU machine)

## 1) Warm the cache (online machine)

WhisperX only downloads what it needs. The simplest cache-warm is: run a real transcription once.

```bash
mkdir -p cache data output

# If running on a VM and you see PermissionError writing /app/output or /.cache:
sudo chown -R 1001:0 cache data output || true

# Put a small sample in ./data (any supported audio format)
# D i a r i z a t i o n  is always enabled in this setup.
# You must accept the model terms on Hugging Face and provide a token:
export HF_TOKEN="hf_..."
docker compose --profile cli run --rm whisperx-cli \
  --model large-v2 \
  --output_dir /app/output/warmup \
  --output_format json \
  --diarize \
  --hf_token "$HF_TOKEN" \
  /app/data/sample.wav
```

Now copy the whole `cache/` directory to your offline machine and keep mounting it to `/.cache`.

## 2) Run the service (offline machine)

```bash
docker compose up -d --build whisperx-api whisperx-worker
curl http://localhost:8000/health
```

Submit a job:

```bash
curl -F "file=@/path/to/audio.wav" \
  "http://localhost:8000/jobs/transcribe?chunk_seconds=1800&overlap_seconds=10&do_align=true"
```

Poll and fetch results:

```bash
curl "http://localhost:8000/jobs/<job_id>"
curl "http://localhost:8000/jobs/<job_id>/result"
```

More details: RUNBOOK.md

