"""Tests for overlap de-duplication at assembly."""

from __future__ import annotations

from whisper_batch.assemble import assemble
from whisper_batch.types import Chunk, Segment


def _texts(transcript):
    return [s.text for s in transcript.segments]


def test_empty():
    assert assemble([], []) == assemble([], [])  # no crash
    assert _texts(assemble([], [])) == []


def test_single_chunk_keeps_in_order():
    ch = Chunk(0, 0.0, 28.0, 0.0, 28.5)
    segs = [Segment(5, 6, "b"), Segment(0, 1, "a")]
    assert _texts(assemble([ch], [segs])) == ["a", "b"]


def test_non_final_chunk_defers_segment_starting_past_logical_end():
    # a segment starting in the right pad belongs to the next chunk, which owns
    # that time -- so a *non-final* chunk drops it and the next chunk keeps it.
    ch0 = Chunk(0, 0.0, 10.0, 0.0, 10.5)
    ch1 = Chunk(1, 10.0, 20.0, 9.5, 20.5)
    segs0 = [Segment(2, 3, "keep"), Segment(10.0, 10.4, "next-owns")]
    segs1 = [Segment(10.0, 10.4, "next-owns"), Segment(12, 13, "more")]
    out = _texts(assemble([ch0, ch1], [segs0, segs1]))
    assert out == ["keep", "next-owns", "more"]
    assert out.count("next-owns") == 1  # ch0 deferred it, ch1 kept it once


def test_last_chunk_keeps_trailing_segment():
    # the final chunk has no successor to defer to, so a segment starting in its
    # last moments (which a non-final chunk would drop) must survive.
    ch = Chunk(0, 0.0, 10.0, 0.0, 10.5)
    segs = [Segment(2, 3, "a"), Segment(9.95, 10.4, "tail")]
    assert _texts(assemble([ch], [segs])) == ["a", "tail"]


def test_drops_bleeding_tail_past_extract_window():
    # whisper inflated this segment past the audio the chunk was given -> drop
    ch = Chunk(0, 0.0, 28.0, 0.0, 28.5)
    segs = [Segment(2, 3, "keep"), Segment(25, 29.0, "hallucinated")]
    assert _texts(assemble([ch], [segs])) == ["keep"]


def test_later_chunk_wins_overlap():
    # chunk1's first segment starts at/before chunk0's kept tail -> tail dropped
    ch0 = Chunk(0, 0.0, 10.0, 0.0, 10.5)
    ch1 = Chunk(1, 10.0, 20.0, 9.5, 20.5)
    segs0 = [Segment(9.6, 9.9, "stale-tail")]
    segs1 = [Segment(9.55, 10.4, "fresh"), Segment(11.0, 12.0, "more")]
    out = _texts(assemble([ch0, ch1], [segs0, segs1]))
    assert out == ["fresh", "more"]  # stale-tail popped


def test_cyberhood_scenario_no_loss_no_dup():
    # mirrors the real failure: chunk0 hallucinates a bleeding tail; chunk1 has
    # the real content with full context. Keep real content once, no duplicate.
    ch0 = Chunk(0, 0.0, 28.0, 0.0, 28.5)
    ch1 = Chunk(1, 28.0, 56.0, 27.5, 56.5)
    segs0 = [Segment(0, 2, "a"), Segment(26, 27.5, "b"), Segment(25, 29.0, "halluc")]
    segs1 = [Segment(27.7, 28.3, "X"), Segment(28.4, 30.0, "Y")]
    out = _texts(assemble([ch0, ch1], [segs0, segs1]))
    assert out == ["a", "b", "X", "Y"]
    assert out.count("X") == 1 and "halluc" not in out


def test_result_is_time_ordered():
    ch0 = Chunk(0, 0.0, 10.0, 0.0, 10.5)
    ch1 = Chunk(1, 10.0, 20.0, 9.5, 20.5)
    out = assemble([ch0, ch1], [[Segment(1, 2, "a")], [Segment(11, 12, "b")]])
    starts = [s.start for s in out.segments]
    assert starts == sorted(starts)
