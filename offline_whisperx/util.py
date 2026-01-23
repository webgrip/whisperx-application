from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def guess_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def guess_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"
