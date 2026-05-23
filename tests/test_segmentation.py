"""Tests for the chunk-planning algorithm (pure logic)."""

from __future__ import annotations

from pathlib import Path

from whisper_batch.config import Config
from whisper_batch.segmentation import _latest_in_range, plan_chunks


def _cfg(**kw) -> Config:
    return Config(model=Path("m"), **kw)


def _durations(chunks):
    return [round(c.end - c.start, 2) for c in chunks]


def test_latest_in_range():
    vals = [1.0, 5.0, 9.0, 20.0]
    assert _latest_in_range(vals, 2.0, 10.0) == 9.0   # largest within range
    assert _latest_in_range(vals, 0.0, 4.0) == 1.0
    assert _latest_in_range(vals, 10.0, 19.0) is None  # nothing in range


def test_cuts_at_silence_midpoint():
    cfg = _cfg(max_chunk_s=28.0, min_chunk_s=1.0, overlap_s=0.5)
    chunks = plan_chunks(100.0, [(49.0, 50.0)], cfg)
    # silence midpoint is 49.5; reachable only from the 2nd chunk's window
    assert [round(c.end, 2) for c in chunks] == [28.0, 49.5, 77.5, 100.0]


def test_forced_split_no_silence():
    cfg = _cfg(max_chunk_s=28.0)
    chunks = plan_chunks(60.0, [], cfg)
    assert _durations(chunks) == [28.0, 28.0, 4.0]
    assert len(chunks) == 3


def test_chunks_are_contiguous_and_cover_duration():
    cfg = _cfg(max_chunk_s=10.0)
    chunks = plan_chunks(33.0, [(9.5, 10.5), (19.0, 21.0)], cfg)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 33.0
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == b.start          # no gaps / overlaps in the logical span
    assert all(c.end - c.start <= 10.0 + 1e-9 for c in chunks)


def test_extract_window_padding_and_clamping():
    cfg = _cfg(max_chunk_s=28.0, overlap_s=0.5)
    chunks = plan_chunks(100.0, [(49.0, 50.0)], cfg)
    # first chunk clamps the left pad at 0; interior chunks pad both sides
    assert chunks[0].extract_start == 0.0
    assert chunks[0].extract_end == chunks[0].end + 0.5
    assert chunks[1].extract_start == chunks[1].start - 0.5
    # last chunk clamps the right pad at the duration
    assert chunks[-1].extract_end == 100.0


def test_extract_pad_clamped_within_narrow_silence():
    # silence (49.6, 50.0) is narrower than 2*overlap, so the pad would spill
    # out of it; clamping keeps each side inside the silence interval.
    cfg = _cfg(max_chunk_s=28.0, overlap_s=0.5)
    chunks = plan_chunks(100.0, [(49.6, 50.0)], cfg)
    # midpoint 49.8 -> chunks [0,28] [28,49.8] [49.8,77.8] [77.8,100]
    assert round(chunks[1].end, 2) == 49.8
    # left chunk's trailing pad clamped to the silence end (not 49.8+0.5=50.3)
    assert chunks[1].extract_end == 50.0
    # right chunk's leading pad clamped to the silence start (not 49.8-0.5=49.3)
    assert chunks[2].extract_start == 49.6
    # the two clips overlap only inside the silence -> no shared speech
    assert chunks[2].extract_start >= 49.6 and chunks[1].extract_end <= 50.0


def test_forced_cut_keeps_symmetric_pad():
    # with no silence to clamp to, the pad falls back to plain overlap_s
    cfg = _cfg(max_chunk_s=28.0, overlap_s=0.5)
    chunks = plan_chunks(60.0, [], cfg)
    assert chunks[1].extract_start == chunks[1].start - 0.5
    assert chunks[0].extract_end == chunks[0].end + 0.5


def test_prefers_latest_midpoint_within_window():
    cfg = _cfg(max_chunk_s=28.0, min_chunk_s=1.0)
    # three candidate midpoints all within the first window -> pick the latest
    chunks = plan_chunks(60.0, [(9.5, 10.5), (19.5, 20.5), (26.5, 27.5)], cfg)
    assert round(chunks[0].end, 2) == 27.0


def test_min_chunk_rejects_too_early_silence():
    cfg = _cfg(max_chunk_s=28.0, min_chunk_s=1.0)
    # silence midpoint at 0.5 is closer than min_chunk_s -> ignored, forced cut
    chunks = plan_chunks(60.0, [(0.4, 0.6)], cfg)
    assert round(chunks[0].end, 2) == 28.0


def test_short_audio_single_chunk():
    cfg = _cfg(max_chunk_s=28.0)
    chunks = plan_chunks(10.0, [], cfg)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0 and chunks[0].end == 10.0


def test_zero_duration_yields_no_chunks():
    assert plan_chunks(0.0, [], _cfg()) == []
