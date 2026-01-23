from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from pyannote.audio import Pipeline

def load_pyannote_pipeline_from_config(path_to_config: Path) -> Pipeline:
    """Load pyannote Pipeline from a local YAML config.

    pyannote interprets paths inside the YAML relative to the *current working directory*.
    We temporarily chdir to the repo root (the directory that contains `models/`) so the
    config can reference `models/...`.
    """
    path_to_config = Path(path_to_config)

    repo_root = None
    for parent in [path_to_config.parent, *path_to_config.parents]:
        if (parent / "models").is_dir():
            repo_root = parent.resolve()
            break
    if repo_root is None:
        # Best-effort fallback: assume cwd already makes `models/...` resolvable.
        repo_root = Path.cwd().resolve()
    cwd = Path.cwd().resolve()
    try:
        os.chdir(repo_root)
        pipeline = Pipeline.from_pretrained(str(path_to_config))
    finally:
        os.chdir(cwd)

    return pipeline

def diarize(pipeline: Pipeline, audio_path: Path, device: str) -> Any:
    pipeline.to(torch.device(device))
    diarization = pipeline(str(audio_path))
    return diarization

def pyannote_annotation_to_segments(annotation: Any) -> List[Dict[str, Any]]:
    """Convert pyannote Annotation into a simple segment list."""
    segments: List[Dict[str, Any]] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": str(speaker),
            }
        )
    return segments
