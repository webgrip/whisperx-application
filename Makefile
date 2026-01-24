.PHONY: venv install run api cli docker-api docker-worker docker-cli docker-build docker-down docker-perms docker-models docker-models-whisper docker-models-align docker-models-vad docker-models-pyannote

WHISPER_REPO ?= Systran/faster-whisper-small
ALIGN_REPO ?= facebook/wav2vec2-base-960h

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && offline-whisperx transcribe --help

api:
	. .venv/bin/activate && uvicorn offline_whisperx.api.main:app --host 0.0.0.0 --port 8000

cli:
	. .venv/bin/activate && offline-whisperx transcribe --audio ./data/sample.wav --outdir ./output

docker-build:
	docker compose build

docker-api:
	docker compose up --build whisperx-api whisperx-worker

docker-worker:
	docker compose up --build whisperx-worker

docker-cli:
	docker compose --profile cli run --rm whisperx-cli

docker-down:
	docker compose down

# Ensure bind-mounted folders exist and are writable by the container user (UID 10001).
docker-perms:
	@mkdir -p models/hf_cache data/uploads output/jobs
	@if [ "$$(id -u)" = "0" ]; then \
		chown -R 10001:10001 models data output; \
	else \
		echo "NOTE: run once: sudo chown -R 10001:10001 models data output"; \
	fi

# Download all models into ./models using the Docker image.
# Requires internet connectivity on the machine running Docker.
docker-models: docker-perms docker-models-whisper docker-models-align docker-models-vad docker-models-pyannote

docker-models-whisper:
	docker compose run --rm --entrypoint python whisperx-api \
		scripts/prefetch_models.py whisper --models-dir /app/models --repo-id "$(WHISPER_REPO)"

docker-models-align:
	docker compose run --rm --entrypoint python whisperx-api \
		scripts/prefetch_models.py align --models-dir /app/models --repo-id "$(ALIGN_REPO)"

docker-models-vad:
	docker compose run --rm --entrypoint python whisperx-api \
		scripts/prefetch_vad.py --models-dir /app/models

# Pyannote models are usually gated. You must export HF_TOKEN and accept the terms on HuggingFace.
docker-models-pyannote:
	@test -n "$$HF_TOKEN" || (echo "HF_TOKEN is required for pyannote models. Export HF_TOKEN=hf_..." && exit 1)
	docker compose run --rm -e HF_TOKEN="$$HF_TOKEN" --entrypoint python whisperx-api \
		scripts/prefetch_pyannote_models.py --models-dir /app/models --hf-token "$$HF_TOKEN"
