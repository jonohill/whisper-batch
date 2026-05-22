"""Pure chunk-planning logic — no I/O, easy to unit test.

Given the audio duration and a list of detected silence intervals, decide where
to cut. We greedily build chunks no longer than ``max_chunk_s``, preferring to
cut in the middle of a silence so word boundaries stay intact.
"""

from __future__ import annotations

from .config import Config
from .types import Chunk


def plan_chunks(
    duration: float,
    silences: list[tuple[float, float]],
    cfg: Config,
) -> list[Chunk]:
    """Build an ordered list of chunks covering ``[0, duration]``.

    Each chunk is at most ``max_chunk_s`` long. When a silence midpoint falls
    within the window we cut there (clean boundary); otherwise we force a cut at
    ``max_chunk_s`` and accept a possible mid-word split at that point.
    """
    midpoints = sorted((s + e) / 2 for s, e in silences)
    chunks: list[Chunk] = []
    t = 0.0
    idx = 0

    while t < duration - 1e-3:
        hard_end = min(t + cfg.max_chunk_s, duration)
        # Prefer the latest silence midpoint within (t + min_chunk_s, hard_end].
        candidate = _latest_in_range(midpoints, t + cfg.min_chunk_s, hard_end)
        cut = candidate if candidate is not None else hard_end
        if cut <= t:  # guarantee forward progress
            cut = hard_end
        chunks.append(Chunk(index=idx, start=t, end=min(cut, duration)))
        t = cut
        idx += 1

    return chunks


def _latest_in_range(sorted_values: list[float], lo: float, hi: float) -> float | None:
    """Return the largest value in ``[lo, hi]``, or None if there is none."""
    best: float | None = None
    for v in sorted_values:
        if v < lo:
            continue
        if v > hi:
            break
        best = v
    return best
