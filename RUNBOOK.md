# Runbook (detailed)

This document contains the full, detailed instructions. The README is intentionally short.

## Model downloads (online machine)

You need, at minimum:

- Whisper (faster-whisper) model directory
  - Recommended for 8GB VRAM: `Systran/faster-whisper-small` (or `...-base` if you need speed)
- Alignment model directory (optional but recommended)
  - Default: `facebook/wav2vec2-base-960h`
- Diarization weights (only if using diarization)
  - `pyannote/segmentation-3.0`
  - `pyannote/wespeaker-voxceleb-resnet34-LM`

Download using the provided scripts:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Whisper model
python scripts/prefetch_models.py whisper --models-dir ./models --repo-id Systran/faster-whisper-small

# Alignment model
python scripts/prefetch_models.py align --models-dir ./models --repo-id facebook/wav2vec2-base-960h

# Diarization weights (requires HF token and accepted terms)
export HF_TOKEN="hf_..."
python scripts/prefetch_pyannote_models.py --models-dir ./models --hf-token "$HF_TOKEN"
```

Or (recommended) download via Docker so you don't need a host Python environment:

```bash
# Whisper + alignment + pyannote (requires HF_TOKEN)
export HF_TOKEN="hf_..."
make docker-models

# Override which Whisper model you want:
# make docker-models WHISPER_REPO=Systran/faster-whisper-base
```

Important: WhisperX also uses a small VAD model blob. `make docker-models` now downloads it into `models/vad/whisperx_vad.bin`. For fully-offline runtime, set `VAD_FILE=/app/models/vad/whisperx_vad.bin` (see `.env.example`).

Expected outputs:

- `models/faster-whisper-*/` (directory)
- `models/wav2vec2-*/` (directory)
- `models/pyannote/pyannote_model_segmentation-3.0.bin`
- `models/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin`

The diarization pipeline config is committed at:

- `config/pyannote_diarization_config.yaml`

## Copy models to your VM

Copy the whole `models/` directory to the VM and mount it to `/app/models`.

Example:

```bash
rsync -a models/ user@vm:/opt/whisperx/models/
```

## Run as a service (LAN)

Note: `WHISPER_MODEL` should point at the *directory* you downloaded under `models/` (e.g. `/app/models/faster-whisper-small` or `/app/models/faster-whisper-base`). If you downloaded `small` but the env is set to `base` (or vice-versa), the worker will try to auto-detect an available `faster-whisper-*` directory under `/app/models`.

This stack is:

- `redis`: durable job queue
- `whisperx-api`: stateless HTTP API
- `whisperx-worker`: GPU worker that runs jobs

Start:

```bash
docker compose up -d --build whisperx-api whisperx-worker
```

If you are bind-mounting host folders into the containers (the default in `docker-compose.yml`), ensure they are writable by the container user (UID 10001). Otherwise you may see warnings like “problem when trying to write in your cache folder (/app/models/hf_cache)”.

Example (repo-local folders):

```bash
sudo mkdir -p models/hf_cache data/uploads output/jobs
sudo chown -R 10001:10001 models data output
```

Example (recommended `/opt/whisperx` layout on a VM):

```bash
sudo mkdir -p /opt/whisperx/models/hf_cache /opt/whisperx/data/uploads /opt/whisperx/output/jobs
sudo chown -R 10001:10001 /opt/whisperx/models /opt/whisperx/data /opt/whisperx/output
```

Check:

```bash
curl http://<host>:8000/health
```

If `curl http://127.0.0.1:8000/health` works on the VM but other machines can’t reach it, open inbound TCP/8000 in your VM firewall and (if applicable) your cloud security group.

Submit long audio (chunked):

```bash
curl -F "file=@/path/to/long.wav" \
  "http://<host>:8000/jobs/transcribe?chunk_seconds=1800&overlap_seconds=10&do_align=true&do_diarize=false"
```

Poll:

```bash
curl "http://<host>:8000/jobs/<job_id>"
```

Result:

```bash
curl "http://<host>:8000/jobs/<job_id>/result"
```

## CLI mode (optional)

Inside Docker:

```bash
docker compose --profile cli run --rm whisperx-cli
```

Outside Docker:

```bash
OFFLINE_WHISPERX_ENFORCE_OFFLINE=1 \
HF_HOME="$(pwd)/models/hf_cache" \
TRANSFORMERS_CACHE="$(pwd)/models/hf_cache" \
offline-whisperx transcribe \
  --audio ./data/sample.wav \
  --outdir ./output
```

## Troubleshooting

- If the worker can’t see the GPU: install NVIDIA Container Toolkit and verify `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- If it tries to download: ensure `OFFLINE_WHISPERX_ENFORCE_OFFLINE=1` and that all model paths exist under the mounted `/app/models`.
