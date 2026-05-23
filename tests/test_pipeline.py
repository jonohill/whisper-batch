"""Tests for the top-level orchestration (all I/O faked)."""

from __future__ import annotations

from pathlib import Path

import whisper_batch.audio as audio_mod
import whisper_batch.pipeline as pipeline_mod
from whisper_batch.config import Config
from whisper_batch.pipeline import transcribe_file
from whisper_batch.types import Segment


def _fake_backend_class(events):
    class FakeBackend:
        def __init__(self, cfg):
            events.append("init")

        async def __aenter__(self):
            events.append("enter")
            return self

        async def __aexit__(self, *exc):
            events.append("exit")

        async def transcribe(self, wav):
            return [Segment(0.5, 1.0, "hi")]   # chunk-local

    return FakeBackend


async def _noop_extract(source, chunk, wav, cfg):
    return None


async def test_pipeline_wires_stages(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(audio_mod, "probe_duration", lambda src, c: 60.0)
    monkeypatch.setattr(audio_mod, "detect_silence", lambda src, c: [])
    monkeypatch.setattr(audio_mod, "extract_chunk", _noop_extract)
    monkeypatch.setattr(pipeline_mod, "ServerBackend", _fake_backend_class(events))

    # keep_temp=False exercises the workdir cleanup + per-chunk unlink paths
    cfg = Config(model=Path("m"), max_chunk_s=28.0, overlap_s=0.5)
    transcript = await transcribe_file(Path("audio.mp3"), cfg, progress=False)

    # 60s / 28s => 3 chunks; each contributes one offset segment
    assert [round(s.start, 2) for s in transcript.segments] == [0.5, 28.0, 56.0]
    assert events == ["init", "enter", "exit"]   # backend lifecycle ran


async def test_pipeline_keep_temp_leaves_workdir(monkeypatch, tmp_path):
    events: list[str] = []
    monkeypatch.setattr(audio_mod, "probe_duration", lambda src, c: 30.0)
    monkeypatch.setattr(audio_mod, "detect_silence", lambda src, c: [])
    monkeypatch.setattr(audio_mod, "extract_chunk", _noop_extract)
    monkeypatch.setattr(pipeline_mod, "ServerBackend", _fake_backend_class(events))

    workdir = tmp_path / "wd"
    workdir.mkdir()
    monkeypatch.setattr(pipeline_mod.tempfile, "mkdtemp", lambda prefix="": str(workdir))

    transcript = await transcribe_file(
        Path("audio.mp3"), Config(model=Path("m"), keep_temp=True), progress=False
    )
    assert transcript.segments               # produced output
    assert workdir.exists()                  # kept, not cleaned up


async def test_pipeline_empty_when_no_chunks(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(audio_mod, "probe_duration", lambda src, c: 0.0)
    monkeypatch.setattr(audio_mod, "detect_silence", lambda src, c: [])
    monkeypatch.setattr(pipeline_mod, "ServerBackend", _fake_backend_class(events))

    transcript = await transcribe_file(Path("audio.mp3"), Config(model=Path("m")),
                                       progress=False)
    assert transcript.segments == []
    assert events == []   # backend never started
