.PHONY: docker-up docker-down docker-logs cli cache-warm

WHISPER_MODEL ?= large-v2
WARMUP_AUDIO ?= /app/data/sample.wav

docker-up:
	@mkdir -p cache data/uploads output/jobs output/warmup
	@if [ "$$(id -u)" = "0" ]; then \
		chown -R 1001:0 cache data output || true; \
	else \
		echo "NOTE: if you hit permission errors, run once: sudo chown -R 1001:0 cache data output"; \
	fi
	docker compose up -d --build whisperx-api whisperx-worker

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f --tail=200

# One-time cache warm-up (run on an online machine).
# Requires HF_TOKEN + accepted pyannote terms.
cache-warm:
	@mkdir -p cache output/warmup
	@if [ "$$(id -u)" = "0" ]; then \
		chown -R 1001:0 cache output || true; \
	else \
		echo "NOTE: if you hit permission errors, run once: sudo chown -R 1001:0 cache output"; \
	fi
	docker compose --profile cli run --rm whisperx-cli \
		--model "$(WHISPER_MODEL)" \
		--output_dir /app/output/warmup \
		--output_format json \
		--diarize \
		"$(WARMUP_AUDIO)"

cli:
	docker compose --profile cli run --rm whisperx-cli --help
