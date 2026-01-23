from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import TranscriptionConfig
from .util import ensure_dir, guess_compute_type, guess_device, write_json
from .outputs import write_srt, write_vtt
from .diarization import (
    load_pyannote_pipeline_from_config,
    diarize as run_diarize,
    pyannote_annotation_to_segments,
)

def _load_whisperx():
    import whisperx  # imported lazily to allow env vars to take effect first
    return whisperx

class Transcriber:
    def __init__(self, cfg: TranscriptionConfig):
        self.cfg = cfg
        self.device = guess_device(cfg.device)
        self.compute_type = guess_compute_type(cfg.compute_type, self.device)
        self.whisperx = _load_whisperx()

        self._asr_model = None
        self._align_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
        self._pyannote_pipeline = None

    def _get_asr_model(self):
        if self._asr_model is None:
            whisper_model = self._resolve_whisper_model_spec(self.cfg.whisper_model)
            self._asr_model = self.whisperx.load_model(
                whisper_model,
                device=self.device,
                compute_type=self.compute_type,
                language=self.cfg.language,
            )
        return self._asr_model

    def _resolve_whisper_model_spec(self, spec: Path) -> str:
        """Resolve configured Whisper model.

        WhisperX ultimately passes this to faster-whisper. If the value is intended to
        be a local directory but doesn't exist, faster-whisper falls back to treating it
        as a Hugging Face repo id and fails validation (e.g. '/app/models/...').

        We treat absolute paths (and obvious path-like values) as local paths and:
        - use them if they exist
        - otherwise auto-pick an available `faster-whisper-*` directory under the same parent
        - otherwise raise a clear error
        """

        spec_str = str(spec)
        p = Path(spec_str)

        looks_like_path = os.path.isabs(spec_str) or spec_str.startswith("./") or spec_str.startswith("../")
        if looks_like_path:
            if p.exists():
                return str(p)

            parent = p.parent
            try:
                candidates = [c for c in parent.glob("faster-whisper-*") if c.is_dir()]
            except Exception:
                candidates = []

            if candidates:
                preferred = [
                    "faster-whisper-large-v3",
                    "faster-whisper-large-v2",
                    "faster-whisper-large",
                    "faster-whisper-medium",
                    "faster-whisper-small",
                    "faster-whisper-base",
                    "faster-whisper-tiny",
                ]
                by_name = {c.name: c for c in candidates}
                for name in preferred:
                    if name in by_name:
                        return str(by_name[name])
                return str(sorted(candidates)[0])

            available = []
            if parent.exists():
                try:
                    available = sorted([c.name for c in parent.iterdir() if c.is_dir()])
                except Exception:
                    available = []

            raise FileNotFoundError(
                "Whisper model directory not found. "
                f"Configured WHISPER_MODEL={spec_str!r} does not exist. "
                f"Looked for 'faster-whisper-*' under {str(parent)!r}. "
                f"Available subdirectories: {available}"
            )

        # Not an obvious path: treat as a model id like 'Systran/faster-whisper-small'.
        return spec_str

    def _get_align_model(self, language: Optional[str], align_model_path: Path):
        language_code = language or "en"
        key = (language_code, str(Path(align_model_path).resolve()))
        if key not in self._align_cache:
            self._align_cache[key] = self._load_align_model(language, align_model_path)
        return self._align_cache[key]

    def _get_pyannote_pipeline(self):
        if self._pyannote_pipeline is None:
            self._pyannote_pipeline = load_pyannote_pipeline_from_config(self.cfg.pyannote_config)
        return self._pyannote_pipeline

    def transcribe(self, audio_path: Path) -> Dict[str, Any]:
        audio_path = Path(audio_path)
        ensure_dir(self.cfg.outdir)

        model = self._get_asr_model()

        # Transcribe
        audio = self.whisperx.load_audio(str(audio_path))
        result: Dict[str, Any] = model.transcribe(audio)

        # Optional alignment (word-level timestamps)
        if self.cfg.do_align and self.cfg.align_model is not None:
            try:
                model_a, metadata = self._get_align_model(self.cfg.language, self.cfg.align_model)
                result = self.whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
            except Exception as e:
                # Keep the base segment timestamps if alignment fails.
                result["_align_error"] = str(e)

        # Optional diarization + speaker assignment
        diarize_segments = None
        if self.cfg.do_diarize:
            pipeline = self._get_pyannote_pipeline()
            annotation = run_diarize(pipeline, audio_path, device=self.device)
            diarize_segments = pyannote_annotation_to_segments(annotation)
            try:
                # whisperx expects pyannote Annotation; but many workflows use segments.
                # We try the native helper if available; fallback to a simple segment-level assignment.
                if hasattr(self.whisperx, "assign_word_speakers"):
                    try:
                        # Some whisperx versions accept Annotation directly:
                        result = self.whisperx.assign_word_speakers(annotation, result)
                    except Exception:
                        # Some accept segments:
                        result = self.whisperx.assign_word_speakers(diarize_segments, result)
                else:
                    result = self._assign_segment_speakers(diarize_segments, result)
            except Exception as e:
                result["_diarize_error"] = str(e)

        # Write outputs
        if self.cfg.write_json:
            write_json(self.cfg.outdir / "transcript.json", result)

        # If the result is whisperx-aligned, segments might be in result["segments"] or result itself.
        segments = result.get("segments") if isinstance(result, dict) else None
        if segments and self.cfg.write_srt:
            write_srt(self.cfg.outdir / "transcript.srt", segments)
        if segments and self.cfg.write_vtt:
            write_vtt(self.cfg.outdir / "transcript.vtt", segments)

        if diarize_segments is not None:
            write_json(self.cfg.outdir / "diarization_segments.json", diarize_segments)

        return result

    def _load_align_model(self, language: Optional[str], align_model_path: Path):
        """Best-effort loader for alignment model.

        WhisperX changed this API a few times; this wrapper tries common signatures.
        """
        language_code = language or "en"
        align_model_path = Path(align_model_path)

        # Common signature:
        #   load_align_model(language_code, device, model_name=..., model_dir=...)
        for kwargs in (
            {"model_name": str(align_model_path)},
            {"model_dir": str(align_model_path)},
        ):
            try:
                return self.whisperx.load_align_model(language_code=language_code, device=self.device, **kwargs)
            except TypeError:
                continue

        # Fallback: older whisperx might not support local model path; this will use HF cache.
        return self.whisperx.load_align_model(language_code=language_code, device=self.device)

    def _assign_segment_speakers(self, diarize_segments, result: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal fallback: assign speaker label per ASR segment by maximum overlap with diarization segments.
        if not diarize_segments or "segments" not in result:
            return result

        def overlap(a0, a1, b0, b1):
            return max(0.0, min(a1, b1) - max(a0, b0))

        for seg in result["segments"]:
            best_speaker = None
            best_ov = 0.0
            for d in diarize_segments:
                ov = overlap(seg["start"], seg["end"], d["start"], d["end"])
                if ov > best_ov:
                    best_ov = ov
                    best_speaker = d["speaker"]
            if best_speaker is not None:
                seg["speaker"] = best_speaker
        return result
