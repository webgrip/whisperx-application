.PHONY: docker-up docker-down docker-logs cli cache-warm hf-check

WHISPER_MODEL ?= large-v2
WARMUP_AUDIO ?= /app/data/sample.wav
CONTAINER_UID ?= 1001

docker-up:
	@mkdir -p cache data/uploads output/jobs output/warmup
	@if [ "$$(id -u)" = "0" ]; then \
		chown -R "$(CONTAINER_UID)":0 cache data output || true; \
	else \
		echo "NOTE: if you hit permission errors, run once: sudo chown -R $(CONTAINER_UID):0 cache data output"; \
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
	@test -n "$$HF_TOKEN" || (echo "HF_TOKEN is required for diarization warmup. Export HF_TOKEN=hf_... and accept pyannote terms." && exit 1)
	@if [ "$$(id -u)" = "0" ]; then \
		chown -R "$(CONTAINER_UID)":0 cache output || true; \
	else \
		echo "NOTE: if you hit permission errors, run once: sudo chown -R $(CONTAINER_UID):0 cache output"; \
	fi
	@echo "Checking Hugging Face access to pyannote diarization model...";
	@$(MAKE) --no-print-directory hf-check || true
	docker compose --profile cli run --rm whisperx-cli \
		--model "$(WHISPER_MODEL)" \
		--output_dir /app/output/warmup \
		--output_format json \
		--diarize \
		--hf_token "$$HF_TOKEN" \
		"$(WARMUP_AUDIO)" \
	|| (echo ""; \
		echo "WhisperX diarization failed to load pyannote pipeline."; \
		echo "Most common fixes:"; \
		echo "  1) Accept terms: https://hf.co/pyannote/speaker-diarization-3.1"; \
		echo "  2) Ensure your HF_TOKEN belongs to that account and has read access"; \
		echo "     (fine-grained tokens often fail for gated repos; classic tokens work best)"; \
		exit 1)

# Quick sanity check for gated model access (no downloads, no token printing).
hf-check:
	@test -n "$$HF_TOKEN" || (echo "HF_TOKEN is not set" && exit 1)
	@code=$$(curl -sS -o /dev/null -w "%{http_code}" \
		-H "Authorization: Bearer $$HF_TOKEN" \
		"https://huggingface.co/api/models/pyannote/speaker-diarization-3.1"); \
	if [ "$$code" = "200" ]; then \
		echo "HF access: OK (200)"; \
	elif [ "$$code" = "401" ] || [ "$$code" = "403" ]; then \
		echo "HF access: DENIED ($$code)"; \
		echo "- Accept the terms: https://hf.co/pyannote/speaker-diarization-3.1"; \
		echo "- Verify the token is from the right HF account"; \
		exit 2; \
	else \
		echo "HF access: unexpected HTTP $$code"; \
		exit 3; \
	fi

cli:
	docker compose --profile cli run --rm whisperx-cli --help
