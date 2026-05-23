"""Stitch per-chunk segments into one coherent, de-duplicated transcript.

Chunks are transcribed with a context pad on each side (see
:class:`whisper_batch.types.Chunk`), so neighbouring chunks overlap and a word
straddling a cut is captured whole. This module removes the resulting
duplication using the way whisper actually behaves at a clip's edges:

* whisper decodes left-to-right, so it transcribes the *start* of a clip well
  (full forward context) but tends to inflate or hallucinate the *trailing*
  segment as it runs out of audio. So at each seam the **later chunk wins** the
  overlap — it heard that region near its (reliable) start.

Concretely, for each chunk in order we keep a segment only if it:

* **starts before the chunk's logical end** — a segment starting in the right
  pad belongs to the next chunk, which owns that time. The **last chunk's right
  edge is open** so its trailing content is never dropped for want of a
  successor; and
* **ends within the chunk's extract window** — a segment whose end runs past the
  audio the chunk was actually given is whisper bleeding past the clip; drop it
  and let the next chunk (which heard it in full) provide that content.

A final "later wins" pass drops any already-kept tail the current chunk
re-transcribed with better context. A small tolerance absorbs timestamp jitter.

This de-duplication is deliberately conservative: it works at *segment*
granularity, where one segment bundles several words. When a phrase straddles a
seam the two chunks render it with slightly different word boundaries, leaving a
small overlap (e.g. "...cause it to" / "it caused it to..."). We keep both
rather than drop a whole segment, because a segment dropped to remove its
duplicated *tail* also discards its unique *head* — empirically a far worse
trade (a 2-word overlap is cheaper than the ~13 unique words lost by dropping
the segment). Eliminating the overlap properly needs word-level timestamps (to
splice mid-segment) or cuts whose context pad stays within the silence so no
phrase crosses a seam — see the planning/extraction stages.
"""

from __future__ import annotations

from .types import Chunk, Segment, Transcript

_TOL = 0.1  # seconds of slack for whisper's timestamp jitter


def assemble(chunks: list[Chunk], chunk_segments: list[list[Segment]]) -> Transcript:
    """Merge per-chunk segment lists (in chunk order) into one transcript."""
    result: list[Segment] = []
    last = len(chunks) - 1

    for i, (chunk, segments) in enumerate(zip(chunks, chunk_segments)):
        # The last chunk's right edge is open so its trailing content survives
        # (there is no successor chunk to defer it to).
        hi = float("inf") if i == last else chunk.end - _TOL
        segs = sorted(
            (
                s
                for s in segments
                if s.start < hi  # starts in our logical span
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
