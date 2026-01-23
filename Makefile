.PHONY: venv install run api cli docker-api docker-worker docker-cli docker-build docker-down

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
