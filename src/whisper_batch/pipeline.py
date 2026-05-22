"""Top-level orchestration tying the stages together.

    probe duration -> detect silence -> plan chunks -> transcribe (pool) -> assemble
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from . import audio, segmentation
from .assemble import assemble
from .config import Config
from .pool import transcribe_chunks
from .types import Transcript

log = logging.getLogger(__name__)


async def transcribe_file(
    source: Path, cfg: Config, *, progress: bool = True
) -> Transcript:
    """Run the full pipeline over *source* and return the assembled transcript."""
    duration = audio.probe_duration(source, cfg)
    log.info("duration: %.1fs", duration)

    silences = audio.detect_silence(source, cfg)
    log.info("detected %d silence interval(s)", len(silences))

    chunks = segmentation.plan_chunks(duration, silences, cfg)
    log.info("planned %d chunk(s) (<= %.0fs each)", len(chunks), cfg.max_chunk_s)
    if not chunks:
        return Transcript()

    workdir = Path(tempfile.mkdtemp(prefix="whisper_batch_"))
    log.debug("workdir: %s", workdir)
    try:
        segments = await transcribe_chunks(
            source, chunks, workdir, cfg, progress=progress
        )
    finally:
        if cfg.keep_temp:
            log.info("kept intermediate files in %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    return assemble(segments)
