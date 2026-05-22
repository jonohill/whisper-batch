"""Transcribe a single chunk: extract its WAV, run whisper.cpp, parse the JSON.

Timestamps from whisper.cpp are relative to the chunk; we add the chunk's start
offset here so everything downstream works in absolute (global) time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .audio import extract_chunk
from .config import Config
from .proc import run_async
from .types import Chunk, Segment

log = logging.getLogger(__name__)


async def transcribe_chunk(
    source: Path, chunk: Chunk, workdir: Path, cfg: Config
) -> list[Segment]:
    """Extract, transcribe, and parse one chunk into globally-offset segments."""
    wav_path = workdir / f"chunk_{chunk.index:05d}.wav"
    out_prefix = workdir / f"chunk_{chunk.index:05d}"

    await extract_chunk(source, chunk, wav_path, cfg)

    cmd = [
        cfg.whisper_bin,
        "-m", str(cfg.model),
        "-f", str(wav_path),
        "-t", str(cfg.threads),
        "-oj",                 # write JSON output
        "-of", str(out_prefix),  # -> <out_prefix>.json
    ]
    if cfg.language:
        cmd += ["-l", cfg.language]
    cmd += cfg.extra_whisper_args

    await run_async(cmd)

    json_path = out_prefix.with_suffix(".json")
    segments = _parse_whisper_json(json_path, offset=chunk.start)

    if not cfg.keep_temp:
        wav_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)

    return segments


def _parse_whisper_json(path: Path, offset: float) -> list[Segment]:
    """Parse a whisper.cpp JSON file; offsets are in milliseconds."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[Segment] = []
    for item in data.get("transcription", []):
        offsets = item.get("offsets", {})
        start = offsets.get("from", 0) / 1000.0 + offset
        end = offsets.get("to", 0) / 1000.0 + offset
        text = item.get("text", "").strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))
    return segments
