"""Pure chunk-planning logic — no I/O, easy to unit test.

Given the audio duration and a list of detected silence intervals, decide where
to cut. We greedily build chunks no longer than ``max_chunk_s``, preferring to
cut in the middle of a silence so word boundaries stay intact.

Each chunk is transcribed with a small context pad on each side. At a *silence*
cut that pad is **clamped to stay within the silence interval**: the chunk
ending at the cut extends no further than the silence's end, and the chunk
starting at the cut reaches back no further than the silence's start. So
neighbouring chunks overlap only on silence — no speech is fed to two chunks,
which is what keeps a phrase from being transcribed (and de-duplicated) twice at
the seam. At a *forced* cut (no silence in the window) there is nothing to clamp
to, so the pad falls back to a plain ``overlap_s`` either side.
"""

from __future__ import annotations

from .config import Config
from .types import Chunk

Interval = tuple[float, float]


def plan_chunks(
    duration: float,
    silences: list[Interval],
    cfg: Config,
) -> list[Chunk]:
    """Build an ordered list of chunks covering ``[0, duration]``.

    Each chunk is at most ``max_chunk_s`` long. When a silence midpoint falls
    within the window we cut there (clean boundary); otherwise we force a cut at
    ``max_chunk_s`` and accept a possible mid-word split at that point.
    """
    intervals = sorted(silences, key=lambda iv: (iv[0] + iv[1]) / 2)
    midpoints = [(s + e) / 2 for s, e in intervals]
    chunks: list[Chunk] = []
    t = 0.0
    idx = 0
    left_silence: Interval | None = None  # silence the current chunk's start cut in

    while t < duration - 1e-3:
        hard_end = min(t + cfg.max_chunk_s, duration)
        # Prefer the latest silence midpoint within (t + min_chunk_s, hard_end].
        j = _latest_index_in_range(midpoints, t + cfg.min_chunk_s, hard_end)
        if j is not None:
            cut, right_silence = midpoints[j], intervals[j]
        else:
            cut, right_silence = hard_end, None
        if cut <= t:  # pragma: no cover - defensive: candidates are always > t
            cut, right_silence = hard_end, None
        end = min(cut, duration)

        # Pad each side by overlap_s, but at a silence cut keep the pad inside
        # the silence so it never reaches into the neighbour's speech.
        extract_start = max(0.0, t - cfg.overlap_s)
        if left_silence is not None:
            extract_start = max(extract_start, left_silence[0])
        extract_end = min(duration, end + cfg.overlap_s)
        if right_silence is not None:
            extract_end = min(extract_end, right_silence[1])

        chunks.append(
            Chunk(
                index=idx,
                start=t,
                end=end,
                extract_start=extract_start,
                extract_end=extract_end,
            )
        )
        t = cut
        left_silence = right_silence
        idx += 1

    return chunks


def _latest_index_in_range(sorted_values: list[float], lo: float, hi: float) -> int | None:
    """Return the index of the largest value in ``[lo, hi]``, or None if none."""
    best: int | None = None
    for i, v in enumerate(sorted_values):
        if v < lo:
            continue
        if v > hi:
            break
        best = i
    return best


def _latest_in_range(sorted_values: list[float], lo: float, hi: float) -> float | None:
    """Return the largest value in ``[lo, hi]``, or None if there is none."""
    i = _latest_index_in_range(sorted_values, lo, hi)
    return None if i is None else sorted_values[i]
