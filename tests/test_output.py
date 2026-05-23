"""Tests for the output writers."""

from __future__ import annotations

import json

import pytest

from whisper_batch.output import (
    WRITERS,
    _fmt_timestamp,
    write_outputs,
)
from whisper_batch.types import Segment, Transcript


def _transcript():
    return Transcript([Segment(0.0, 1.5, "Olá mundo"), Segment(1.5, 3.0, "second")])


def test_fmt_timestamp():
    assert _fmt_timestamp(0.0) == "00:00:00,000"
    assert _fmt_timestamp(3661.5) == "01:01:01,500"
    assert _fmt_timestamp(3661.5, ".") == "01:01:01.500"
    assert _fmt_timestamp(-5.0) == "00:00:00,000"   # clamped


def test_write_txt(tmp_path):
    paths = write_outputs(_transcript(), tmp_path / "out", ["txt"])
    assert paths == [tmp_path / "out.txt"]
    assert paths[0].read_text(encoding="utf-8") == "Olá mundo\nsecond\n"


def test_write_srt(tmp_path):
    write_outputs(_transcript(), tmp_path / "out", ["srt"])
    body = (tmp_path / "out.srt").read_text(encoding="utf-8")
    assert body.startswith("1\n00:00:00,000 --> 00:00:01,500\nOlá mundo")
    assert "2\n00:00:01,500 --> 00:00:03,000\nsecond" in body


def test_write_vtt(tmp_path):
    write_outputs(_transcript(), tmp_path / "out", ["vtt"])
    body = (tmp_path / "out.vtt").read_text(encoding="utf-8")
    assert body.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in body


def test_write_json_roundtrip(tmp_path):
    write_outputs(_transcript(), tmp_path / "out", ["json"])
    data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert data == [
        {"start": 0.0, "end": 1.5, "text": "Olá mundo"},
        {"start": 1.5, "end": 3.0, "text": "second"},
    ]


def test_write_outputs_multiple_and_paths(tmp_path):
    paths = write_outputs(_transcript(), tmp_path / "out", ["txt", "json"])
    assert [p.name for p in paths] == ["out.txt", "out.json"]
    assert all(p.exists() for p in paths)


def test_write_outputs_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        write_outputs(_transcript(), tmp_path / "out", ["bogus"])


def test_writers_registry():
    assert set(WRITERS) == {"txt", "srt", "vtt", "json"}
