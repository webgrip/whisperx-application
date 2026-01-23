from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

class TranscriptionConfig(BaseModel):
    # Paths
    whisper_model: Path = Field(default=Path("models/faster-whisper-base"))
    align_model: Path | None = Field(default=Path("models/wav2vec2-base-960h"))
    pyannote_config: Path = Field(default=Path("config/pyannote_diarization_config.yaml"))

    # Runtime
    device: str = Field(default="auto")  # auto|cpu|cuda
    compute_type: str = Field(default="auto")  # auto|int8|float16|float32
    language: str | None = Field(default=None)

    # Features
    do_align: bool = Field(default=True)
    do_diarize: bool = Field(default=False)

    # Caches / offline
    hf_home: Path = Field(default=Path("models/hf_cache"))
    enforce_offline: bool = Field(default=True)

    # Output
    outdir: Path = Field(default=Path("output"))
    write_json: bool = Field(default=True)
    write_srt: bool = Field(default=False)
    write_vtt: bool = Field(default=False)
