"""Tests for the external-command wrappers (using trivial real commands)."""

from __future__ import annotations

import pytest

from whisper_batch.proc import CommandError, run, run_async


def test_run_returns_stdout():
    out, err = run(["printf", "hi"])
    assert out == "hi"
    assert err == ""


def test_run_captures_stderr():
    _, err = run(["sh", "-c", "printf oops 1>&2"])
    assert "oops" in err


def test_run_raises_on_failure():
    with pytest.raises(CommandError) as ei:
        run(["sh", "-c", "printf boom 1>&2; exit 3"])
    assert ei.value.returncode == 3
    assert "boom" in str(ei.value)


async def test_run_async_returns_stdout():
    out, _ = await run_async(["printf", "hi"])
    assert out == "hi"


async def test_run_async_raises_on_failure():
    with pytest.raises(CommandError) as ei:
        await run_async(["sh", "-c", "exit 2"])
    assert ei.value.returncode == 2
