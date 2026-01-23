from __future__ import annotations

import os
from pathlib import Path

OFFLINE_ENV_VARS = {
    # Hugging Face stack
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    # Reduce surprises/noise
    "TOKENIZERS_PARALLELISM": "false",
}

def enforce_offline(hf_home: str | Path | None = None) -> None:
    """Force offline behavior for HF/transformers/pyannote.

    If any dependency tries to download, it should error immediately rather than hanging.
    """
    for k, v in OFFLINE_ENV_VARS.items():
        os.environ.setdefault(k, v)

    if hf_home is not None:
        hf_home = str(Path(hf_home).resolve())
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_home)
        os.environ.setdefault("HF_HUB_CACHE", hf_home)
