"""Minimal stdlib HTTP helpers for talking to whisper-server (no third-party deps).

These are synchronous; backends call them via ``asyncio.to_thread`` so the event
loop stays free while a request is in flight.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def probe(url: str, timeout: float = 1.0) -> bool:
    """Return True if *url* answers with any HTTP response (i.e. server is up)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True  # an HTTP error status is still a live server
    except (urllib.error.URLError, OSError):
        return False


def post_file(
    url: str,
    file_field: str,
    file_path: Path,
    fields: dict[str, str],
    timeout: float = 1800.0,
) -> dict:
    """POST *file_path* plus form *fields* as multipart/form-data; return JSON."""
    boundary = uuid.uuid4().hex
    head = bytearray()
    for key, value in fields.items():
        head += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()
    head += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{file_path.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode()
    body = bytes(head) + file_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
