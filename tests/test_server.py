"""Tests for the warm-server pool, with the process layer faked."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

import whisper_batch.server as server_mod
from whisper_batch.config import Config
from whisper_batch.server import ServerBackend, WhisperServer


def _cfg(**kw) -> Config:
    return Config(model=Path("m"), **kw)


def test_whisperserver_urls():
    ws = WhisperServer(_cfg(server_host="127.0.0.1"), 18080, Path("/tmp/x.log"))
    assert ws.base_url == "http://127.0.0.1:18080"
    assert ws.inference_url == "http://127.0.0.1:18080/inference"


def test_tail_log(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("\n".join(f"line{i}" for i in range(20)))
    ws = WhisperServer(_cfg(), 1, log)
    assert ws._tail_log(lines=3).splitlines() == ["line17", "line18", "line19"]
    missing = WhisperServer(_cfg(), 1, tmp_path / "missing.log")
    assert missing._tail_log() == "(no log)"


async def test_transcribe_dispatches_and_returns_server(monkeypatch):
    backend = ServerBackend(_cfg())
    backend._free = asyncio.Queue()
    backend._free.put_nowait(SimpleNamespace(inference_url="http://h/inference"))

    srt = "1\n00:00:00,000 --> 00:00:01,000\n hi\n"
    monkeypatch.setattr(server_mod.http, "post_file", lambda *a, **k: srt)

    segs = await backend.transcribe(Path("c.wav"))
    assert [s.text for s in segs] == ["hi"]
    assert backend._free.qsize() == 1   # server handed back to the pool


async def test_transcribe_without_start_raises():
    with pytest.raises(RuntimeError):
        await ServerBackend(_cfg()).transcribe(Path("c.wav"))


async def test_start_populates_pool_then_stop(monkeypatch):
    started, stopped = [], []

    class FakeWS:
        def __init__(self, cfg, port, log_path):
            self.port = port

        async def start(self, timeout=120.0):
            started.append(self.port)

        async def stop(self):
            stopped.append(self.port)

    monkeypatch.setattr(server_mod, "WhisperServer", FakeWS)
    backend = ServerBackend(_cfg(workers=3, server_port=20000))
    await backend.start()
    assert sorted(started) == [20000, 20001, 20002]
    assert backend._free.qsize() == 3
    await backend.stop()
    assert sorted(stopped) == [20000, 20001, 20002]


async def test_start_cleans_up_on_partial_failure(monkeypatch):
    started, stopped = [], []

    class FakeWS:
        def __init__(self, cfg, port, log_path):
            self.port = port

        async def start(self, timeout=120.0):
            if self.port == 20001:
                raise RuntimeError("boom")
            started.append(self.port)

        async def stop(self):
            stopped.append(self.port)

    monkeypatch.setattr(server_mod, "WhisperServer", FakeWS)
    backend = ServerBackend(_cfg(workers=3, server_port=20000))
    with pytest.raises(RuntimeError, match="boom"):
        await backend.start()
    # whatever started got stopped — nothing left orphaned
    assert set(started) <= set(stopped)


async def test_async_context_manager(monkeypatch):
    class FakeWS:
        def __init__(self, cfg, port, log_path):
            self.port = port

        async def start(self, timeout=120.0):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(server_mod, "WhisperServer", FakeWS)
    async with ServerBackend(_cfg(workers=1)) as backend:
        assert backend._free.qsize() == 1


class _FakeProc:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


async def test_whisperserver_start_and_stop(monkeypatch, tmp_path):
    async def fake_create(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(server_mod.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(server_mod.http, "probe", lambda url, timeout=1.0: True)

    ws = WhisperServer(_cfg(no_gpu=True, language="en"), 21000, tmp_path / "s.log")
    await ws.start(timeout=5.0)
    assert ws._proc is not None
    await ws.stop()
    assert ws._proc.returncode is not None


async def test_whisperserver_detects_early_exit(monkeypatch, tmp_path):
    async def fake_create(*args, **kwargs):
        return _FakeProc(returncode=1)   # already dead

    monkeypatch.setattr(server_mod.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(server_mod.http, "probe", lambda url, timeout=1.0: False)

    ws = WhisperServer(_cfg(), 21001, tmp_path / "s.log")
    with pytest.raises(RuntimeError, match="exited early"):
        await ws.start(timeout=5.0)


async def test_whisperserver_ready_timeout(monkeypatch, tmp_path):
    async def fake_create(*args, **kwargs):
        return _FakeProc()   # alive but never becomes ready

    monkeypatch.setattr(server_mod.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(server_mod.http, "probe", lambda url, timeout=1.0: False)

    ws = WhisperServer(_cfg(), 21002, tmp_path / "s.log")
    with pytest.raises(TimeoutError):
        await ws.start(timeout=0.4)


async def test_whisperserver_stop_noop_when_not_started():
    ws = WhisperServer(_cfg(), 1, Path("/tmp/never.log"))
    await ws.stop()   # _proc is None -> returns without error


async def test_whisperserver_stop_escalates_to_kill(monkeypatch, tmp_path):
    async def fake_create(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(server_mod.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(server_mod.http, "probe", lambda url, timeout=1.0: True)
    ws = WhisperServer(_cfg(), 21003, tmp_path / "s.log")
    await ws.start(timeout=5.0)

    # force the graceful wait to time out so stop() escalates to kill()
    async def fake_wait_for(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(server_mod.asyncio, "wait_for", fake_wait_for)
    await ws.stop()
    assert ws._proc.returncode == -9   # kill() was called
