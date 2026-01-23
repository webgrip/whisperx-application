FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg needed for decoding most audio/video formats
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Non-root runtime user
RUN useradd -m -u 10001 appuser \
 && chown -R appuser:appuser /app

# Install deps (torch should be installed in your base image if you want CUDA;
# this Dockerfile uses a CUDA-enabled PyTorch base image).
RUN python -m pip install -U pip \
 && python -m pip install --no-cache-dir -r requirements.txt

USER appuser

ENTRYPOINT ["tini", "--"]
CMD ["offline-whisperx", "--help"]
