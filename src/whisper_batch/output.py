"""Write a :class:`Transcript` out in the usual subtitle / text formats."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .types import Transcript


def _fmt_timestamp(seconds: float, sep: str = ",") -> str:
    """Format seconds as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


# --- Renderers: format a transcript to a string. ---------------------------
# Split out from the file writers so the HTTP server can reuse them to build
# response bodies without going through the filesystem.

def render_txt(transcript: Transcript) -> str:
    return "\n".join(s.text for s in transcript.segments) + "\n"


def render_srt(transcript: Transcript) -> str:
    lines: list[str] = []
    for i, seg in enumerate(transcript.segments, start=1):
        lines += [
            str(i),
            f"{_fmt_timestamp(seg.start)} --> {_fmt_timestamp(seg.end)}",
            seg.text,
            "",
        ]
    return "\n".join(lines)


def render_vtt(transcript: Transcript) -> str:
    lines = ["WEBVTT", ""]
    for seg in transcript.segments:
        lines += [
            f"{_fmt_timestamp(seg.start, '.')} --> {_fmt_timestamp(seg.end, '.')}",
            seg.text,
            "",
        ]
    return "\n".join(lines)


def render_json(transcript: Transcript) -> str:
    data = [
        {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
        for s in transcript.segments
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


# --- Writers: render to a string, then write it to disk. --------------------

def write_txt(transcript: Transcript, path: Path) -> None:
    path.write_text(render_txt(transcript), encoding="utf-8")


def write_srt(transcript: Transcript, path: Path) -> None:
    path.write_text(render_srt(transcript), encoding="utf-8")


def write_vtt(transcript: Transcript, path: Path) -> None:
    path.write_text(render_vtt(transcript), encoding="utf-8")


def write_json(transcript: Transcript, path: Path) -> None:
    path.write_text(render_json(transcript), encoding="utf-8")


WRITERS: dict[str, Callable[[Transcript, Path], None]] = {
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
    "json": write_json,
}


def write_outputs(transcript: Transcript, out_prefix: Path, formats: list[str]) -> list[Path]:
    """Write *transcript* in each requested format. Returns the paths written."""
    written: list[Path] = []
    for fmt in formats:
        writer = WRITERS.get(fmt)
        if writer is None:
            raise ValueError(f"unknown output format: {fmt!r}")
        path = out_prefix.with_name(f"{out_prefix.name}.{fmt}")
        writer(transcript, path)
        written.append(path)
    return written
