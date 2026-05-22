"""Runtime configuration for a transcription run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_workers() -> int:
    cpu = os.cpu_count() or 4
    # Whisper's decoder is largely sequential, so a single worker rarely fills
    # all cores. Favour many light workers over a few heavy ones; with the
    # default of 2 threads/worker this targets roughly all available cores.
    return max(1, cpu // 2)


@dataclass(slots=True)
class Config:
    """Everything a run needs. `model` is the only required field."""

    model: Path

    # External binaries (overridable for non-standard installs).
    whisper_server_bin: str = "whisper-server"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    language: str | None = None  # None => whisper auto-detects per chunk

    # Concurrency (also the number of warm servers)
    workers: int = field(default_factory=_default_workers)
    threads: int = 2  # threads per whisper-server

    # Server pool
    server_host: str = "127.0.0.1"
    server_port: int = 18080  # base port; the pool uses port .. port+workers-1
    no_gpu: bool = False      # pass -ng to whisper.cpp

    # Chunking
    max_chunk_s: float = 28.0        # stay under Whisper's 30s receptive field
    min_chunk_s: float = 1.0         # avoid wastefully tiny chunks
    overlap_s: float = 0.5           # context pad each side of a chunk (for clean cuts)
    silence_noise_db: float = -30.0  # ffmpeg silencedetect noise floor (dB)
    min_silence_s: float = 0.5       # ffmpeg silencedetect minimum silence (s)

    # Misc
    keep_temp: bool = False
