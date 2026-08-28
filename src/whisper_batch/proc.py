"""Thin wrappers around external command execution.

All ffmpeg / whisper.cpp invocations go through here so that error handling and
command logging live in one place. Both helpers return ``(stdout, stderr)`` and
raise :class:`CommandError` on a non-zero exit.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Sequence

log = logging.getLogger(__name__)


class CommandError(RuntimeError):
    """Raised when an external command exits non-zero."""

    def __init__(self, cmd: Sequence[str], returncode: int, stderr: str) -> None:
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"command failed ({returncode}): {' '.join(self.cmd)}\n{stderr.strip()}"
        )


def run(cmd: Sequence[str]) -> tuple[str, str]:
    """Run *cmd* synchronously. Returns ``(stdout, stderr)``."""
    log.debug("run: %s", " ".join(map(str, cmd)))
    # ffmpeg truncates long metadata tag values when dumping them to stderr and
    # can cut mid-UTF-8-sequence, so decode leniently rather than raising.
    proc = subprocess.run(
        list(cmd),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise CommandError(cmd, proc.returncode, proc.stderr)
    return proc.stdout, proc.stderr


async def run_async(cmd: Sequence[str]) -> tuple[str, str]:
    """Run *cmd* asynchronously. Returns ``(stdout, stderr)``."""
    log.debug("run_async: %s", " ".join(map(str, cmd)))
    proc = await asyncio.create_subprocess_exec(
        *map(str, cmd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    stdout = out.decode("utf-8", "replace")
    stderr = err.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise CommandError(cmd, proc.returncode or -1, stderr)
    return stdout, stderr
