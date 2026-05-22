"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    """A planned slice of the source audio, in seconds from the start.

    ``start``/``end`` are the *logical* span — a clean partition of the timeline
    used to assign transcript segments. ``extract_start``/``extract_end`` are the
    slightly padded window actually fed to whisper, so words straddling a cut are
    captured whole in context. The padding is dropped again at assembly.
    """

    index: int
    start: float
    end: float
    extract_start: float | None = None  # defaults to start (no pad)
    extract_end: float | None = None    # defaults to end (no pad)

    def __post_init__(self) -> None:
        if self.extract_start is None:
            self.extract_start = self.start
        if self.extract_end is None:
            self.extract_end = self.end

    @property
    def extract_duration(self) -> float:
        return self.extract_end - self.extract_start


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
