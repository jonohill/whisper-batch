"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Chunk:
    """A planned slice of the source audio, in seconds from the start."""

    index: int
    start: float
    end: float
    path: Path | None = None  # populated once the WAV is extracted

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class Segment:
    """A transcribed segment with timestamps in seconds (global, post-offset)."""

    start: float
    end: float
    text: str


@dataclass(slots=True)
class Transcript:
    """The assembled, time-ordered result."""

    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())
