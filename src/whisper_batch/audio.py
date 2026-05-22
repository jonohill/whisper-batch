"""All ffmpeg / ffprobe interaction lives here.

The rest of the pipeline never shells out directly — it asks this module for a
duration, a list of silence intervals, or an extracted chunk WAV.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import Config
from .proc import run, run_async
from .types import Chunk

_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)"
)


def probe_duration(source: Path, cfg: Config) -> float:
    """Return the duration of *source* in seconds via ffprobe."""
    stdout, _ = run([
        cfg.ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(source),
    ])
    return float(stdout.strip())


def detect_silence(source: Path, cfg: Config) -> list[tuple[float, float]]:
    """Return (start, end) silence intervals in seconds, parsed from ffmpeg.

    ffmpeg's ``silencedetect`` filter logs ``silence_start`` / ``silence_end``
    lines to stderr; we run it against a null output purely to harvest those.
    """
    _, stderr = run([
        cfg.ffmpeg_bin,
        "-hide_banner", "-nostdin",
        "-i", str(source),
        "-af", f"silencedetect=noise={cfg.silence_noise_db}dB:d={cfg.min_silence_s}",
        "-f", "null", "-",
    ])

    intervals: list[tuple[float, float]] = []
    pending_start: float | None = None
    for line in stderr.splitlines():
        if m := _SILENCE_START_RE.search(line):
            pending_start = float(m.group(1))
        elif m := _SILENCE_END_RE.search(line):
            end = float(m.group(1))
            if pending_start is None:
                # Missed the matching start (e.g. silence at file start).
                pending_start = max(0.0, end - float(m.group(2)))
            intervals.append((pending_start, end))
            pending_start = None
    return intervals


async def extract_chunk(source: Path, chunk: Chunk, out_path: Path, cfg: Config) -> None:
    """Extract a 16 kHz mono PCM WAV for *chunk* from *source*.

    ``-ss`` before ``-i`` performs a fast input seek; output is re-encoded to the
    format whisper.cpp expects.
    """
    await run_async([
        cfg.ffmpeg_bin,
        "-hide_banner", "-nostdin", "-y",
        "-ss", f"{chunk.start:.3f}",
        "-t", f"{chunk.duration:.3f}",
        "-i", str(source),
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    ])
    chunk.path = out_path
