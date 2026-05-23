"""Tests for the ffmpeg/ffprobe layer.

Parsing logic is unit-tested with mocked subprocess output; a single end-to-end
test runs real ffmpeg/ffprobe (skipped if they are not installed).
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

import whisper_batch.audio as audio_mod
from whisper_batch.audio import detect_silence, extract_chunk, probe_duration
from whisper_batch.types import Chunk

_HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def test_probe_duration_parses_float(monkeypatch, cfg):
    monkeypatch.setattr(audio_mod, "run", lambda cmd: ("123.45\n", ""))
    assert probe_duration(Path("a.mp3"), cfg) == 123.45


def test_detect_silence_parses_intervals(monkeypatch, cfg):
    stderr = "\n".join([
        "[silencedetect @ 0x1] silence_start: 1.0",
        "[silencedetect @ 0x1] silence_end: 2.5 | silence_duration: 1.5",
        "[silencedetect @ 0x1] silence_start: 10.0",
        "[silencedetect @ 0x1] silence_end: 11.0 | silence_duration: 1.0",
    ])
    monkeypatch.setattr(audio_mod, "run", lambda cmd: ("", stderr))
    assert detect_silence(Path("a"), cfg) == [(1.0, 2.5), (10.0, 11.0)]


def test_detect_silence_orphan_end_uses_duration(monkeypatch, cfg):
    # silence_end with no preceding start -> start derived from the duration
    stderr = "[silencedetect] silence_end: 2.0 | silence_duration: 1.0"
    monkeypatch.setattr(audio_mod, "run", lambda cmd: ("", stderr))
    assert detect_silence(Path("a"), cfg) == [(1.0, 2.0)]


def test_extract_chunk_builds_correct_ffmpeg_args(monkeypatch, cfg):
    captured = {}

    async def fake_run_async(cmd):
        captured["cmd"] = cmd
        return ("", "")

    monkeypatch.setattr(audio_mod, "run_async", fake_run_async)
    chunk = Chunk(0, 10.0, 38.0, extract_start=9.5, extract_end=38.5)
    asyncio.run(extract_chunk(Path("src.mp3"), chunk, Path("out.wav"), cfg))

    cmd = captured["cmd"]
    assert cmd[cmd.index("-ss") + 1] == "9.500"
    assert cmd[cmd.index("-t") + 1] == "29.000"      # extract_duration
    assert "16000" in cmd and "pcm_s16le" in cmd
    assert cmd[-1] == "out.wav"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_audio_integration(tmp_path, cfg):
    wav = tmp_path / "tone.wav"
    # 3s tone | 1s silence | 3s tone  (silence gap around 3-4s)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1[o]", "-map", "[o]",
            str(wav),
        ],
        check=True,
    )

    assert abs(probe_duration(wav, cfg) - 7.0) < 0.3

    silences = detect_silence(wav, cfg)
    assert any(s <= 4.0 and e >= 3.0 for s, e in silences)  # found the gap

    out = tmp_path / "clip.wav"
    chunk = Chunk(0, 2.0, 5.0, extract_start=2.0, extract_end=5.0)
    asyncio.run(extract_chunk(wav, chunk, out, cfg))
    assert out.exists()
    assert abs(probe_duration(out, cfg) - 3.0) < 0.2
