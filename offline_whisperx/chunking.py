from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start_s: float
    end_s: float


def _ffmpeg_extract_wav_segment(
    *,
    input_path: Path,
    output_path: Path,
    start_s: float,
    duration_s: float,
    sample_rate: int = 16000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s}",
        "-t",
        f"{duration_s}",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        "-y",
        str(output_path),
    ]

    subprocess.run(cmd, check=True)


def iter_audio_chunks(
    *,
    input_path: Path,
    chunk_seconds: float,
    overlap_seconds: float,
    work_dir: Path,
    total_duration_seconds: float | None = None,
) -> Iterator[AudioChunk]:
    """Create fixed-size (optionally overlapping) audio chunks as temporary WAV files.

    If `total_duration_seconds` is None, we compute it via ffprobe.

    Notes:
    - Uses ffmpeg/ffprobe installed in the environment.
    - Chunks are 16kHz mono PCM WAV.
    """

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be < chunk_seconds")

    input_path = Path(input_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if total_duration_seconds is None:
        total_duration_seconds = probe_duration_seconds(input_path)

    step = chunk_seconds - overlap_seconds
    n = int(math.ceil(max(0.0, total_duration_seconds) / step))

    for i in range(n):
        start_s = i * step
        if start_s >= total_duration_seconds:
            break
        end_s = min(total_duration_seconds, start_s + chunk_seconds)
        duration_s = max(0.0, end_s - start_s)
        if duration_s <= 0:
            continue

        out = work_dir / f"chunk_{i:06d}_{start_s:.3f}_{end_s:.3f}.wav"
        _ffmpeg_extract_wav_segment(
            input_path=input_path,
            output_path=out,
            start_s=start_s,
            duration_s=duration_s,
        )
        yield AudioChunk(path=out, start_s=start_s, end_s=end_s)


def probe_duration_seconds(input_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(res.stdout.strip())


def offset_transcript_timestamps(result: dict, *, offset_s: float, trim_start_s: float = 0.0) -> dict:
    """Offset segment/word timestamps in a whisperx-style result dict.

    - Adds `offset_s` to all segment start/end.
    - If `trim_start_s` > 0, drops segments fully before that time and clips partial ones.
      (This is useful for overlap de-duplication.)
    """

    if not isinstance(result, dict):
        return result

    segments = result.get("segments")
    if not isinstance(segments, list):
        return result

    new_segments = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        # Overlap de-duplication window (relative to chunk)
        if trim_start_s > 0.0:
            if end <= trim_start_s:
                continue
            if start < trim_start_s:
                start = trim_start_s

        seg2 = dict(seg)
        seg2["start"] = start + offset_s
        seg2["end"] = end + offset_s

        # word-level timestamps (common whisperx format)
        words = seg2.get("words")
        if isinstance(words, list):
            new_words = []
            for w in words:
                if not isinstance(w, dict):
                    continue
                w2 = dict(w)
                if "start" in w2 and w2["start"] is not None:
                    w2["start"] = float(w2["start"]) + offset_s
                if "end" in w2 and w2["end"] is not None:
                    w2["end"] = float(w2["end"]) + offset_s
                new_words.append(w2)
            seg2["words"] = new_words

        new_segments.append(seg2)

    out = dict(result)
    out["segments"] = new_segments
    return out
