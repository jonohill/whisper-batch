"""Tests for parsing whisper-server's SRT/VTT responses."""

from __future__ import annotations

from whisper_batch.server import _parse_srt, _srt_to_seconds


def test_srt_to_seconds():
    assert _srt_to_seconds("00:00:03,960") == 3.96
    assert _srt_to_seconds("01:02:03.500") == 3723.5     # VTT uses a dot
    assert _srt_to_seconds(" 00:00:00,000 ") == 0.0
    assert _srt_to_seconds("garbage") is None


def test_parse_srt_basic():
    text = (
        "1\n00:00:00,000 --> 00:00:02,000\n Hello\n\n"
        "2\n00:00:02,000 --> 00:00:05,500\n World again\n"
    )
    segs = _parse_srt(text)
    assert len(segs) == 2
    assert (segs[0].start, segs[0].end, segs[0].text) == (0.0, 2.0, "Hello")
    assert (segs[1].end, segs[1].text) == (5.5, "World again")


def test_parse_vtt_with_header_and_dots():
    text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi there\n"
    segs = _parse_srt(text)
    assert len(segs) == 1                 # the WEBVTT header block is skipped
    assert segs[0].text == "Hi there"


def test_parse_srt_joins_multiline_text():
    text = "1\n00:00:00,000 --> 00:00:02,000\nline one\nline two\n"
    assert _parse_srt(text)[0].text == "line one line two"


def test_parse_srt_drops_textless_cue():
    assert _parse_srt("1\n00:00:00,000 --> 00:00:02,000\n") == []


def test_parse_srt_empty_input():
    assert _parse_srt("") == []
    assert _parse_srt("   \n\n  ") == []


def test_parse_srt_skips_unparseable_timestamps():
    # has a "-->" line but the stamps don't parse -> cue is skipped
    assert _parse_srt("1\nBAD --> ALSO_BAD\nsome text\n") == []
