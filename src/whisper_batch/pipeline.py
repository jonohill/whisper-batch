"""Top-level orchestration tying the stages together.

    probe duration -> detect silence -> plan chunks -> transcribe (pool) -> assemble

The chosen backend is started (warm servers spawned, or nothing for the CLI
backend) for the duration of the run via its async context manager.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

from . import audio, segmentation
from .assemble import assemble
from .config import Config
from .pool import transcribe_chunks
from .server import ServerBackend
from .types import Transcript

log = logging.getLogger(__name__)


async def transcribe_file(
    source: Path,
    cfg: Config,
    *,
    progress: bool = True,
    backend: ServerBackend | None = None,
) -> Transcript:
    """Run the full pipeline over *source* and return the assembled transcript.

    *backend* lets a long-running caller (the HTTP server) supply an
    already-started, shared warm pool that outlives a single file. When omitted
    — the CLI's one-shot case — a pool is spawned for the duration of this call
    and torn down afterwards.
    """
    duration = audio.probe_duration(source, cfg)
    log.info("duration: %.1fs", duration)

    silences = audio.detect_silence(source, cfg, duration)
    log.info("detected %d silence interval(s)", len(silences))

    chunks = segmentation.plan_chunks(duration, silences, cfg)
    log.info("planned %d chunk(s) (<= %.0fs each)", len(chunks), cfg.max_chunk_s)
    if not chunks:
        return Transcript()

    workdir = Path(tempfile.mkdtemp(prefix="whisper_batch_"))
    log.debug("workdir: %s", workdir)

    # Own the backend (start/stop it) only if the caller didn't hand us one.
    owns_backend = backend is None
    if owns_backend:
        backend = ServerBackend(cfg)
    lifecycle = backend if owns_backend else nullcontext()
    try:
        async with lifecycle:
            chunk_segments = await transcribe_chunks(
                source, chunks, workdir, cfg, backend, progress=progress
            )
    finally:
        if cfg.keep_temp:
            log.info("kept intermediate files in %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    return assemble(chunks, chunk_segments)
