"""Tests for argument parsing, exit codes, and Config construction."""

from __future__ import annotations

from pathlib import Path

import whisper_batch.cli as cli_mod
from whisper_batch.cli import main
from whisper_batch.types import Segment, Transcript


def test_missing_model_returns_2(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    inp = tmp_path / "a.wav"
    inp.write_bytes(b"x")
    assert main([str(inp)]) == 2


def test_missing_input_returns_2(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert main([str(tmp_path / "nope.wav"), "-m", "model.bin"]) == 2


def test_unknown_format_returns_2(tmp_path):
    inp = tmp_path / "a.wav"
    inp.write_bytes(b"x")
    assert main([str(inp), "-m", "model.bin", "-f", "txt,bogus"]) == 2


def test_success_builds_config_and_writes(monkeypatch, tmp_path):
    inp = tmp_path / "a.wav"
    inp.write_bytes(b"x")
    captured = {}

    async def fake_transcribe_file(source, cfg, progress=True):
        captured["source"] = source
        captured["cfg"] = cfg
        return Transcript([Segment(0.0, 1.0, "hello")])

    monkeypatch.setattr(cli_mod, "transcribe_file", fake_transcribe_file)

    out = tmp_path / "out"
    rc = main([
        str(inp), "-m", "model.bin", "-l", "en",
        "-w", "3", "-t", "4", "--server-port", "19000",
        "--overlap", "0.7", "--no-gpu", "-f", "txt,json", "-o", str(out),
    ])

    assert rc == 0
    cfg = captured["cfg"]
    assert cfg.workers == 3
    assert cfg.threads == 4
    assert cfg.no_gpu is True
    assert cfg.server_port == 19000
    assert cfg.overlap_s == 0.7
    assert cfg.language == "en"
    assert (tmp_path / "out.txt").exists()
    assert (tmp_path / "out.json").exists()


def test_command_error_returns_1(monkeypatch, tmp_path):
    from whisper_batch.proc import CommandError

    inp = tmp_path / "a.wav"
    inp.write_bytes(b"x")

    async def boom(source, cfg, progress=True):
        raise CommandError(["whisper-server"], 1, "stderr text")

    monkeypatch.setattr(cli_mod, "transcribe_file", boom)
    rc = main([str(inp), "-m", "model.bin", "-f", "txt", "-o", str(tmp_path / "o")])
    assert rc == 1


def test_model_from_env(monkeypatch, tmp_path):
    inp = tmp_path / "a.wav"
    inp.write_bytes(b"x")
    monkeypatch.setenv("WHISPER_MODEL", "from-env.bin")
    captured = {}

    async def fake_transcribe_file(source, cfg, progress=True):
        captured["cfg"] = cfg
        return Transcript([])

    monkeypatch.setattr(cli_mod, "transcribe_file", fake_transcribe_file)
    rc = main([str(inp), "-o", str(tmp_path / "out"), "-f", "txt"])
    assert rc == 0
    assert str(captured["cfg"].model) == "from-env.bin"
