.PHONY: docker-up docker-down docker-logs cli cache-warm

WHISPER_MODEL ?= large-v2
WARMUP_AUDIO ?= /app/data/sample.wav

docker-up:
	@mkdir -p cache data/uploads output/jobs
	docker compose up -d --build whisperx-api whisperx-worker

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f --tail=200

# One-time cache warm-up (run on an online machine).
# Set HF_TOKEN if you want diarization cache warmed.
cache-warm:
	@mkdir -p cache output
	docker compose --profile cli run --rm whisperx-cli \
		--model "$(WHISPER_MODEL)" \
		--output_dir /app/output/warmup \
		--output_format json \
		--diarize \
		"$(WARMUP_AUDIO)"

cli:
	docker compose --profile cli run --rm whisperx-cli --help
