"""Concurrency layer: run chunks through a transcription backend in parallel.

For each chunk the pool extracts its WAV, hands it to the backend (CLI or warm
server), and shifts the returned chunk-local timestamps into global time. An
:class:`asyncio.Semaphore` caps how many chunks are in flight at once.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from . import audio
from .config import Config
from .server import ServerBackend
from .types import Chunk, Segment


async def transcribe_chunks(
    source: Path,
    chunks: list[Chunk],
    workdir: Path,
    cfg: Config,
    backend: ServerBackend,
    *,
    progress: bool = True,
) -> list[list[Segment]]:
    """Transcribe all chunks with at most ``cfg.workers`` in flight.

    Returns one segment list per chunk, aligned to *chunks* order
    (``asyncio.gather`` preserves input order). Timestamps are global but not yet
    de-duplicated across the padded overlaps — :func:`assemble` does that.
    """
    sem = asyncio.Semaphore(cfg.workers)
    total = len(chunks)
    done = 0

    async def worker(chunk: Chunk) -> list[Segment]:
        nonlocal done
        async with sem:
            wav = workdir / f"chunk_{chunk.index:05d}.wav"
            await audio.extract_chunk(source, chunk, wav, cfg)
            local = await backend.transcribe(wav)
            if not cfg.keep_temp:
                wav.unlink(missing_ok=True)
        # Timestamps are relative to the (padded) extract window.
        offset = chunk.extract_start
        segments = [Segment(s.start + offset, s.end + offset, s.text) for s in local]
        done += 1
        if progress:
            _log_progress(done, total, chunk)
        return segments

    return await asyncio.gather(*(worker(c) for c in chunks))


def _log_progress(done: int, total: int, chunk: Chunk) -> None:
    pct = done / total * 100 if total else 100.0
    print(
        f"[{done:>4}/{total}] {pct:5.1f}%  chunk {chunk.index} "
        f"({_fmt(chunk.start)}-{_fmt(chunk.end)})",
        file=sys.stderr,
        flush=True,
    )


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
