"""Transcription backends: turn a 16 kHz mono WAV into chunk-local segments.

Both implementations share one interface so the pool doesn't care which is used:

* :class:`CliBackend` spawns a fresh ``whisper-cli`` per chunk — reloads the model
  every time, but needs no setup.
* :class:`ServerBackend` keeps a pool of warm ``whisper-server`` processes that
  load the model once (see :mod:`whisper_batch.server`).

Backends return timestamps relative to the chunk; the pool shifts them into
global time.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .config import Config
from .proc import run_async
from .types import Segment


class Backend(ABC):
    """Transcribes WAV files. Use as an async context manager for lifecycle."""

    async def __aenter__(self) -> "Backend":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Acquire resources (e.g. spawn servers). Default: nothing."""

    async def stop(self) -> None:
        """Release resources. Default: nothing."""

    @abstractmethod
    async def transcribe(self, wav_path: Path) -> list[Segment]:
        """Transcribe *wav_path* into segments with chunk-local timestamps (s)."""


class CliBackend(Backend):
    """One ``whisper-cli`` invocation per chunk (model reloads each time)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def transcribe(self, wav_path: Path) -> list[Segment]:
        cfg = self.cfg
        out_prefix = wav_path.with_suffix("")  # -> <prefix>.json
        cmd = [
            cfg.whisper_bin,
            "-m", str(cfg.model),
            "-f", str(wav_path),
            "-t", str(cfg.threads),
            "-oj", "-of", str(out_prefix),
        ]
        if cfg.language:
            cmd += ["-l", cfg.language]
        if cfg.no_gpu:
            cmd += ["-ng"]
        cmd += cfg.extra_whisper_args

        await run_async(cmd)

        json_path = out_prefix.with_suffix(".json")
        segments = _parse_cli_json(json_path)
        if not cfg.keep_temp:
            json_path.unlink(missing_ok=True)
        return segments


def _parse_cli_json(path: Path) -> list[Segment]:
    """whisper-cli JSON: ``offsets`` are milliseconds, relative to the chunk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    segments: list[Segment] = []
    for item in data.get("transcription", []):
        offsets = item.get("offsets", {})
        text = (item.get("text") or "").strip()
        if text:
            segments.append(
                Segment(offsets.get("from", 0) / 1000.0, offsets.get("to", 0) / 1000.0, text)
            )
    return segments


def make_backend(cfg: Config) -> Backend:
    """Construct the backend named by ``cfg.backend``."""
    if cfg.backend == "server":
        from .server import ServerBackend  # lazy: keeps the http stack out of cli use
        return ServerBackend(cfg)
    if cfg.backend == "cli":
        return CliBackend(cfg)
    raise ValueError(f"unknown backend: {cfg.backend!r}")
