from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """Runtime settings for API/service + worker mode.

    Reads from environment variables and optional `.env` (if present).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Cache locations (mounted as a Docker volume in worker mode)
    hf_home: Path = Field(default=Path("/.cache/huggingface"), validation_alias="HF_HOME")
    model_dir: Path = Field(default=Path("/.cache"), validation_alias="WHISPERX_MODEL_DIR")
    pyannote_cache: Path = Field(default=Path("/.cache/pyannote"), validation_alias="PYANNOTE_CACHE")

    # WhisperX CLI configuration
    whisper_model: str = Field(default="large-v2", validation_alias="WHISPER_MODEL")
    align_model: str | None = Field(default=None, validation_alias="ALIGN_MODEL")
    diarize_model: str | None = Field(default=None, validation_alias="DIARIZE_MODEL")

    @field_validator("whisper_model", mode="before")
    @classmethod
    def _normalize_whisper_model(cls, v):
        if v is None:
            return "large-v2"
        if isinstance(v, Path):
            v = str(v)
        if not isinstance(v, str):
            return v

        s = v.strip()
        if not s:
            return "large-v2"

        # Back-compat: previous versions used local directory paths.
        # Convert those into WhisperX model names like `small`.
        if s.startswith("/") or s.startswith("./") or s.startswith("../"):
            name = Path(s).name
            if name.startswith("faster-whisper-"):
                return name.replace("faster-whisper-", "", 1)
            return name

        return s

    @field_validator("align_model", "diarize_model", mode="before")
    @classmethod
    def _normalize_optional_model_name(cls, v):
        if v is None:
            return None
        if isinstance(v, Path):
            v = str(v)
        if not isinstance(v, str):
            return v
        s = v.strip()
        if not s:
            return None

        # Back-compat: older configs used local paths. For WhisperX CLI,
        # prefer letting WhisperX auto-select instead of passing a path.
        if s.startswith("/") or s.startswith("./") or s.startswith("../"):
            return None

        return s

    # Runtime
    device: str = Field(default="auto", validation_alias="DEVICE")
    compute_type: str = Field(default="auto", validation_alias="COMPUTE_TYPE")
    language: str | None = Field(default=None, validation_alias="LANGUAGE")

    # Worker (WhisperX CLI execution) runtime.
    # Note: WhisperX CLI expects compute_type in {float16,float32,int8}; it does not accept "auto".
    worker_device: str = Field(default="cuda", validation_alias="WORKER_DEVICE")
    worker_compute_type: str = Field(default="float16", validation_alias="WORKER_COMPUTE_TYPE")

    @field_validator("language", mode="before")
    @classmethod
    def _empty_language_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v
    # Output/layout
    outdir: Path = Field(default=Path("/app/output"), validation_alias="OUTDIR")
    uploads_dir: Path = Field(default=Path("/app/data/uploads"), validation_alias="UPLOADS_DIR")
    jobs_dir: Path = Field(default=Path("/app/output/jobs"), validation_alias="JOBS_DIR")

    # Concurrency
    max_workers: int = Field(default=1, validation_alias="MAX_WORKERS")

    # Queue
    redis_url: str = Field(default="redis://redis:6379/0", validation_alias="REDIS_URL")
    queue_name: str = Field(default="whisperx", validation_alias="QUEUE_NAME")

    # Output formats (service may still return JSON)
    write_srt: bool = Field(default=False, validation_alias="WRITE_SRT")
    write_vtt: bool = Field(default=False, validation_alias="WRITE_VTT")
