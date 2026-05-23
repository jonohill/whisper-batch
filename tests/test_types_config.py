"""Tests for the data structures and configuration."""

from __future__ import annotations

from pathlib import Path

import whisper_batch.config as config_mod
from whisper_batch.config import Config, _default_workers
from whisper_batch.types import Chunk, Segment, Transcript


def test_chunk_extract_defaults_to_logical_span():
    c = Chunk(0, 5.0, 10.0)
    assert c.extract_start == 5.0
    assert c.extract_end == 10.0
    assert c.extract_duration == 5.0


def test_chunk_extract_explicit():
    c = Chunk(0, 5.0, 10.0, 4.5, 10.5)
    assert c.extract_duration == 6.0


def test_transcript_text_joins_nonempty():
    t = Transcript([Segment(0, 1, " a "), Segment(1, 2, "  "), Segment(2, 3, "b")])
    assert t.text == "a b"


def test_transcript_empty_text():
    assert Transcript().text == ""


def test_default_workers(monkeypatch):
    monkeypatch.setattr(config_mod.os, "cpu_count", lambda: 10)
    assert _default_workers() == 5
    monkeypatch.setattr(config_mod.os, "cpu_count", lambda: 1)
    assert _default_workers() == 1
    monkeypatch.setattr(config_mod.os, "cpu_count", lambda: None)
    assert _default_workers() == 2   # falls back to 4 // 2


def test_config_defaults():
    c = Config(model=Path("m"))
    assert c.whisper_server_bin == "whisper-server"
    assert c.threads == 2
    assert c.server_port == 18080
    assert c.no_gpu is False
    assert c.overlap_s == 0.5
    assert c.max_chunk_s == 28.0


def test_config_has_no_removed_backend_fields():
    c = Config(model=Path("m"))
    assert not hasattr(c, "backend")
    assert not hasattr(c, "whisper_bin")
    assert not hasattr(c, "extra_whisper_args")
