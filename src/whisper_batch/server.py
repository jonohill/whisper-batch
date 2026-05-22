"""Warm worker pool: a set of long-lived ``whisper-server`` processes.

Each server loads the model once at startup; chunks are dispatched to whichever
server is currently free (an :class:`asyncio.Queue` of idle servers). This is
what removes the per-chunk model-reload cost the CLI backend pays.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from . import http
from .backend import Backend
from .config import Config
from .types import Segment

log = logging.getLogger(__name__)


class WhisperServer:
    """Lifecycle of a single ``whisper-server`` process on its own port."""

    def __init__(self, cfg: Config, port: int, log_path: Path) -> None:
        self.cfg = cfg
        self.port = port
        self.log_path = log_path
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.server_host}:{self.port}"

    @property
    def inference_url(self) -> str:
        return f"{self.base_url}/inference"

    async def start(self, timeout: float = 120.0) -> None:
        cfg = self.cfg
        cmd = [
            cfg.whisper_server_bin,
            "-m", str(cfg.model),
            "-t", str(cfg.threads),
            "--host", cfg.server_host,
            "--port", str(self.port),
        ]
        if cfg.language:
            cmd += ["-l", cfg.language]
        if cfg.no_gpu:
            cmd += ["-ng"]

        log_file = self.log_path.open("wb")
        self._proc = await asyncio.create_subprocess_exec(
            *map(str, cmd), stdout=log_file, stderr=log_file
        )
        await self._await_ready(timeout)

    async def _await_ready(self, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            if self._proc and self._proc.returncode is not None:
                raise RuntimeError(
                    f"whisper-server on port {self.port} exited early "
                    f"(rc={self._proc.returncode}):\n{self._tail_log()}"
                )
            if await asyncio.to_thread(http.probe, self.base_url + "/", 1.0):
                return
            if loop.time() > deadline:
                raise TimeoutError(
                    f"whisper-server on port {self.port} not ready after "
                    f"{timeout:.0f}s:\n{self._tail_log()}"
                )
            await asyncio.sleep(0.25)

    async def stop(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()

    def _tail_log(self, lines: int = 15) -> str:
        try:
            content = self.log_path.read_text("utf-8", errors="replace")
        except OSError:
            return "(no log)"
        return "\n".join(content.splitlines()[-lines:])


class ServerBackend(Backend):
    """Dispatches chunks across a pool of warm whisper-server instances."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.servers: list[WhisperServer] = []
        self._free: asyncio.Queue[WhisperServer] | None = None
        self._logdir: Path | None = None

    async def start(self) -> None:
        cfg = self.cfg
        n = max(1, cfg.workers)
        self._logdir = Path(tempfile.mkdtemp(prefix="whisper_servers_"))
        self.servers = [
            WhisperServer(cfg, cfg.server_port + i, self._logdir / f"server_{i}.log")
            for i in range(n)
        ]
        log.info(
            "starting %d whisper-server(s) on ports %d-%d ...",
            n, cfg.server_port, cfg.server_port + n - 1,
        )
        await asyncio.gather(*(s.start() for s in self.servers))

        self._free = asyncio.Queue()
        for server in self.servers:
            self._free.put_nowait(server)
        log.info("%d server(s) ready", n)

    async def stop(self) -> None:
        await asyncio.gather(*(s.stop() for s in self.servers), return_exceptions=True)
        if self._logdir and not self.cfg.keep_temp:
            shutil.rmtree(self._logdir, ignore_errors=True)

    async def transcribe(self, wav_path: Path) -> list[Segment]:
        if self._free is None:
            raise RuntimeError("ServerBackend.start() was not called")
        server = await self._free.get()
        try:
            data = await asyncio.to_thread(
                http.post_file,
                server.inference_url,
                "file",
                wav_path,
                {
                    "response_format": "verbose_json",
                    "language": self.cfg.language or "auto",
                },
            )
        finally:
            self._free.put_nowait(server)
        return _parse_server_json(data)


def _parse_server_json(data: dict) -> list[Segment]:
    """whisper-server verbose_json: start/end are seconds, relative to the chunk."""
    segments: list[Segment] = []
    for seg in data.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text:
            segments.append(Segment(float(seg["start"]), float(seg["end"]), text))
    return segments
