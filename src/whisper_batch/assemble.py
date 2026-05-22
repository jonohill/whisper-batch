"""Stitch per-chunk segments into one coherent, de-duplicated transcript.

Chunks are transcribed with a context pad on each side (see
:class:`whisper_batch.types.Chunk`), so neighbouring chunks overlap and a word
straddling a cut is captured whole. This module removes the resulting duplication
using the way whisper actually behaves at a clip's edges:

* whisper decodes left-to-right, so it transcribes the *start* of a clip well
  (full forward context) but tends to inflate or hallucinate the *trailing*
  segment as it runs out of audio. So at each seam the **later chunk wins** the
  overlap — it heard that region near its (reliable) start.

Concretely, for each chunk in order we keep a segment only if it:

* **starts before the chunk's logical end** — a segment starting in the right pad
  belongs to the next chunk, which owns that time; and
* **ends within the chunk's extract window** — a segment whose end runs past the
  audio the chunk was actually given is whisper bleeding past the clip; drop it
  and let the next chunk (which heard it in full) provide that content.

A final "later wins" pass drops any already-kept tail the current chunk
re-transcribed with better context. A small tolerance absorbs timestamp jitter.

Note: this works at segment granularity. Word-level timestamps (which the server
backend already returns) would let us splice mid-segment and recover the last
~1% of accuracy lost to coarse seams — a worthwhile future refinement.
"""

from __future__ import annotations

from .types import Chunk, Segment, Transcript

_TOL = 0.1  # seconds of slack for whisper's timestamp jitter


def assemble(chunks: list[Chunk], chunk_segments: list[list[Segment]]) -> Transcript:
    """Merge per-chunk segment lists (in chunk order) into one transcript."""
    result: list[Segment] = []

    for chunk, segments in zip(chunks, chunk_segments):
        segs = sorted(
            (
                s
                for s in segments
                if s.start < chunk.end - _TOL  # starts in our logical span
                and s.end <= chunk.extract_end + _TOL  # and we heard it in full
            ),
            key=lambda s: s.start,
        )
        if segs:
            # Later chunk wins the overlap: drop any already-kept segments that
            # start at/after this chunk's first segment.
            first = segs[0].start
            while result and result[-1].start >= first - _TOL:
                result.pop()
        result.extend(segs)

    return Transcript(segments=result)
