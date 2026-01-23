from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """Runtime settings for API/service + worker mode.

    Reads from environment variables and optional `.env` (if present).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Offline enforcement
    offline_enforce: bool = Field(default=True, validation_alias="OFFLINE_WHISPERX_ENFORCE_OFFLINE")

    # Cache locations
    hf_home: Path = Field(default=Path("/app/models/hf_cache"), validation_alias="HF_HOME")

    # Model paths
    whisper_model: Path = Field(default=Path("/app/models/faster-whisper-base"), validation_alias="WHISPER_MODEL")
    align_model: Path = Field(default=Path("/app/models/wav2vec2-base-960h"), validation_alias="ALIGN_MODEL")
    pyannote_config: Path = Field(
        default=Path("/app/config/pyannote_diarization_config.yaml"), validation_alias="PYANNOTE_CONFIG"
    )

    # Runtime
    device: str = Field(default="auto", validation_alias="DEVICE")
    compute_type: str = Field(default="auto", validation_alias="COMPUTE_TYPE")
    language: str | None = Field(default=None, validation_alias="LANGUAGE")

    @field_validator("language", mode="before")
    @classmethod
    def _empty_language_to_none(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # Feature defaults
    do_align_default: bool = Field(default=True, validation_alias="DO_ALIGN")
    do_diarize_default: bool = Field(default=False, validation_alias="DO_DIARIZE")

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
