from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

def _format_ts_srt(seconds: float) -> str:
    # SRT: HH:MM:SS,mmm
    ms_total = int(round(seconds * 1000))
    s, ms = divmod(ms_total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _format_ts_vtt(seconds: float) -> str:
    # VTT: HH:MM:SS.mmm
    ms_total = int(round(seconds * 1000))
    s, ms = divmod(ms_total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def write_srt(path: Path, segments: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_ts_srt(float(seg["start"]))
        end = _format_ts_srt(float(seg["end"]))
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if speaker:
            text = f"[{speaker}] {text}"
        lines.extend([str(i), f"{start} --> {end}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")

def write_vtt(path: Path, segments: List[Dict[str, Any]]) -> None:
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _format_ts_vtt(float(seg["start"]))
        end = _format_ts_vtt(float(seg["end"]))
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if speaker:
            text = f"[{speaker}] {text}"
        lines.extend([f"{start} --> {end}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
