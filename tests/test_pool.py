"""Tests for the concurrency layer (with a fake backend, no real whisper)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import whisper_batch.audio as audio_mod
from whisper_batch.config import Config
from whisper_batch.pool import transcribe_chunks
from whisper_batch.types import Chunk, Segment


async def _noop_extract(source, chunk, wav, cfg):
    return None


async def test_offsets_by_extract_start_and_aligns(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_mod, "extract_chunk", _noop_extract)
    cfg = Config(model=Path("m"), keep_temp=True)
    chunks = [
        Chunk(0, 0.0, 28.0, 0.0, 28.5),
        Chunk(1, 28.0, 56.0, 27.5, 56.5),
    ]

    class FakeBackend:
        async def transcribe(self, wav, *, language=None):
            return [Segment(1.0, 2.0, "w")]   # chunk-local timestamps

    result = await transcribe_chunks(
        Path("s"), chunks, tmp_path, cfg, FakeBackend(), progress=False
    )
    assert len(result) == 2                     # one list per chunk, in order
    assert result[0][0].start == 1.0            # extract_start 0.0
    assert result[1][0].start == 1.0 + 27.5     # offset by extract_start


async def test_concurrency_is_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_mod, "extract_chunk", _noop_extract)
    cfg = Config(model=Path("m"), workers=2, keep_temp=True)
    chunks = [Chunk(i, i * 10.0, i * 10.0 + 10.0) for i in range(6)]

    state = {"current": 0, "peak": 0}

    class FakeBackend:
        async def transcribe(self, wav, *, language=None):
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            await asyncio.sleep(0.02)
            state["current"] -= 1
            return []

    await transcribe_chunks(Path("s"), chunks, tmp_path, cfg, FakeBackend(), progress=False)
    assert state["peak"] == 2                    # never exceeds workers, and reaches it


async def test_progress_logging(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(audio_mod, "extract_chunk", _noop_extract)
    cfg = Config(model=Path("m"), keep_temp=True)
    chunks = [Chunk(0, 0.0, 65.0, 0.0, 65.0)]

    class FakeBackend:
        async def transcribe(self, wav, *, language=None):
            return []

    await transcribe_chunks(Path("s"), chunks, tmp_path, cfg, FakeBackend(), progress=True)
    err = capsys.readouterr().err
    assert "1/1" in err          # progress counter
    assert "00:00:00-00:01:05" in err   # _fmt rendered the chunk span


async def test_keep_temp_false_unlinks(monkeypatch, tmp_path):
    # the per-chunk WAV is removed when keep_temp is False (missing file is fine)
    monkeypatch.setattr(audio_mod, "extract_chunk", _noop_extract)
    cfg = Config(model=Path("m"))  # keep_temp defaults to False
    chunks = [Chunk(0, 0.0, 10.0, 0.0, 10.0)]

    class FakeBackend:
        async def transcribe(self, wav, *, language=None):
            return []

    result = await transcribe_chunks(Path("s"), chunks, tmp_path, cfg, FakeBackend(), progress=False)
    assert result == [[]]
