"""Stitch per-chunk segments into one coherent transcript.

Segments already carry absolute timestamps (the offset was applied in
:mod:`whisper_batch.transcribe`), so assembly is just an ordering step today.
This is the natural home for future overlap de-duplication or gap merging.
"""

from __future__ import annotations

from .types import Segment, Transcript


def assemble(segments: list[Segment]) -> Transcript:
    """Return a time-ordered :class:`Transcript` from raw segments."""
    ordered = sorted(segments, key=lambda s: s.start)
    return Transcript(segments=ordered)
