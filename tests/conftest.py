"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_batch.config import Config
from whisper_batch.types import Chunk, Segment


@pytest.fixture
def cfg() -> Config:
    return Config(model=Path("model.bin"))


@pytest.fixture
def mk_chunk():
    def _mk(index, start, end, extract_start=None, extract_end=None) -> Chunk:
        return Chunk(index, start, end, extract_start, extract_end)

    return _mk


@pytest.fixture
def mk_seg():
    def _mk(start, end, text="x") -> Segment:
        return Segment(start, end, text)

    return _mk
