"""Tests for the stdlib HTTP helpers, against a real local server.

This exercises the hand-rolled multipart encoder for real, which is exactly the
kind of code worth a round-trip test.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from whisper_batch import http


class _Handler(BaseHTTPRequestHandler):
    last_body: bytes = b""
    last_ctype: str = ""

    def log_message(self, *args):  # silence the server
        pass

    def do_GET(self):
        code = 404 if self.path == "/missing" else 200
        self.send_response(code)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).last_body = self.rfile.read(length)
        type(self).last_ctype = self.headers.get("Content-Type", "")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"RESPONSE-BODY")


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join()


def test_probe_true_when_up(server):
    assert http.probe(server + "/", timeout=2.0) is True


def test_probe_true_on_http_error(server):
    # an HTTP error status still means a live server
    assert http.probe(server + "/missing", timeout=2.0) is True


def test_probe_false_when_down():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listening here now
    assert http.probe(f"http://127.0.0.1:{port}/", timeout=0.5) is False


def test_post_file_multipart_roundtrip(server, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"AUDIODATA\x00\x01")
    body = http.post_file(
        server + "/inference",
        "file",
        wav,
        {"response_format": "srt", "language": "en"},
        timeout=10.0,
    )
    assert body == "RESPONSE-BODY"

    sent = _Handler.last_body
    assert _Handler.last_ctype.startswith("multipart/form-data; boundary=")
    assert b'name="response_format"' in sent and b"srt" in sent
    assert b'name="language"' in sent and b"en" in sent
    assert b'name="file"; filename="a.wav"' in sent
    assert b"AUDIODATA\x00\x01" in sent   # raw file bytes preserved
