# Runbook (detailed)

This repo intentionally does **not** vend WhisperX/pyannote dependencies itself.

Instead, the GPU worker is built FROM a public WhisperX container image and runs the upstream `whisperx` CLI, using a shared cache directory (matching the approach discussed in WhisperX issue #873).

## Directory layout

- `cache/` → mounted to `/.cache` in the worker
  - Hugging Face hub cache + pyannote cache live here
- `data/` → uploads and local files
- `output/` → job artifacts, logs, and final `transcript.json`

## Cache warm-up (online)

You generally need to warm the cache by running WhisperX once while you have internet.

1) Put a short audio file in `data/` (e.g. `data/sample.wav`).
2) Warm cache:

```bash
mkdir -p cache data output

# Ensure host bind-mounts are writable by the container user.
# The upstream WhisperX image commonly runs as uid 1001.
sudo chown -R 1001:0 cache data output || true

# If you prefer, you can instead run containers as root by setting WHISPERX_UID=0 in .env
# (useful on VMs where bind-mounted dirs are root-owned).

# Diarization is always enabled in this setup.
# Accept the pyannote model terms on Hugging Face and set a token:
# - https://hf.co/pyannote/speaker-diarization-3.1
export HF_TOKEN="hf_..."

# Optional: verify your token has access to the gated pyannote model
make hf-check

docker compose --profile cli run --rm whisperx-cli \
  --model large-v2 \
  --output_dir /app/output/warmup \
  --output_format json \
  --diarize \
  --hf_token "$HF_TOKEN" \
  /app/data/sample.wav
```

Then move `cache/` to the offline machine. If the offline machine has no network, WhisperX will naturally fall back to the existing cache.

## Run the service

```bash
docker compose up -d --build whisperx-api whisperx-worker
curl http://localhost:8000/health
```

Submit:

```bash
curl -F "file=@/path/to/audio.wav" \
  "http://localhost:8000/jobs/transcribe?chunk_seconds=1800&overlap_seconds=10&do_align=true"
```

## Troubleshooting

- GPU not visible: verify NVIDIA Container Toolkit with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- Job failed: check `output/jobs/<job_id>/whisperx.log` (and per-chunk logs under `output/jobs/<job_id>/chunks/`).
- Job times out after ~180s: increase `JOB_TIMEOUT_SECONDS` in `.env` (default is 7200 in this repo).
- Diarization fails offline: make sure `cache/` includes pyannote models and `HF_TOKEN` is available if your WhisperX version requires it.
