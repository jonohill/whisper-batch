"""Concurrency layer: run many chunks through whisper.cpp at once.

Each chunk is an independent whisper.cpp subprocess, so there is no GIL concern;
an :class:`asyncio.Semaphore` simply caps how many run simultaneously.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from .config import Config
from .transcribe import transcribe_chunk
from .types import Chunk, Segment


async def transcribe_chunks(
    source: Path,
    chunks: list[Chunk],
    workdir: Path,
    cfg: Config,
    *,
    progress: bool = True,
) -> list[Segment]:
    """Transcribe all chunks with at most ``cfg.workers`` in flight.

    Results are returned flattened and in chunk order (``asyncio.gather``
    preserves input order regardless of completion order).
    """
    sem = asyncio.Semaphore(cfg.workers)
    total = len(chunks)
    done = 0

    async def worker(chunk: Chunk) -> list[Segment]:
        nonlocal done
        async with sem:
            segments = await transcribe_chunk(source, chunk, workdir, cfg)
        done += 1
        if progress:
            _log_progress(done, total, chunk)
        return segments

    results = await asyncio.gather(*(worker(c) for c in chunks))
    return [seg for chunk_segs in results for seg in chunk_segs]


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
