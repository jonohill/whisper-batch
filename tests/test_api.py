"""Tests for the HTTP service, with the transcription pipeline faked.

A :class:`FakeTranscriber` stands in for the warm pool so these run without
ffmpeg, whisper-server, or a model — they exercise the HTTP layer: routing,
response_format rendering, auth, and that the uploaded file and language reach
the transcriber.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import whisper_batch.api as api_mod
from whisper_batch.api import Transcriber, create_app
from whisper_batch.config import Config
from whisper_batch.proc import CommandError
from whisper_batch.types import Segment, Transcript

pytestmark = pytest.mark.filterwarnings("ignore")


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None, bytes]] = []

    async def transcribe(self, source: Path, *, language: str | None = None) -> Transcript:
        # Record what the HTTP layer handed us, including the saved upload bytes.
        self.calls.append((source, language, source.read_bytes()))
        return Transcript([Segment(0.0, 1.5, "Olá mundo"), Segment(1.5, 3.0, "second")])


def _client(transcriber: FakeTranscriber | None = None, *, api_key: str | None = None):
    transcriber = transcriber or FakeTranscriber()
    app = create_app(transcriber=transcriber, api_key=api_key)
    return TestClient(app), transcriber


def _post(client, *, fmt=None, language=None, file=b"RIFFfake", path="/v1/audio/transcriptions",
          headers=None):
    data = {}
    if fmt is not None:
        data["response_format"] = fmt
    if language is not None:
        data["language"] = language
    return client.post(
        path,
        files={"file": ("clip.wav", file, "audio/wav")},
        data=data,
        headers=headers or {},
    )


def test_health():
    client, _ = _client()
    with client:
        assert client.get("/health").json() == {"status": "ok"}


def test_default_json():
    client, _ = _client()
    with client:
        r = _post(client)
    assert r.status_code == 200
    assert r.json() == {"text": "Olá mundo second"}


def test_text_format():
    client, _ = _client()
    with client:
        r = _post(client, fmt="text")
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == "Olá mundo second"


def test_srt_format():
    client, _ = _client()
    with client:
        r = _post(client, fmt="srt")
    assert "1\n00:00:00,000 --> 00:00:01,500\nOlá mundo" in r.text
    assert "2\n00:00:01,500 --> 00:00:03,000\nsecond" in r.text


def test_vtt_format():
    client, _ = _client()
    with client:
        r = _post(client, fmt="vtt")
    assert r.text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in r.text


def test_verbose_json():
    client, _ = _client()
    with client:
        r = _post(client, fmt="verbose_json", language="pt")
    body = r.json()
    assert body["language"] == "pt"
    assert body["text"] == "Olá mundo second"
    assert body["duration"] == 3.0
    assert body["segments"][0] == {
        "id": 0, "seek": 0, "start": 0.0, "end": 1.5, "text": "Olá mundo",
    }


def test_unknown_format_is_400():
    client, _ = _client()
    with client:
        r = _post(client, fmt="bogus")
    assert r.status_code == 400


def test_missing_file_is_422():
    client, _ = _client()
    with client:
        r = client.post("/v1/audio/transcriptions", data={"response_format": "json"})
    assert r.status_code == 422


def test_language_and_upload_reach_transcriber():
    client, fake = _client()
    with client:
        _post(client, language="de", file=b"AUDIODATA")
    (saved_path, language, saved_bytes) = fake.calls[0]
    assert language == "de"
    assert saved_bytes == b"AUDIODATA"           # upload was streamed to disk
    assert saved_path.name.endswith(".wav")      # original suffix preserved


def test_upload_temp_is_cleaned_up():
    client, fake = _client()
    with client:
        _post(client)
    saved_path = fake.calls[0][0]
    assert not saved_path.exists()               # per-request temp dir removed
    assert not saved_path.parent.exists()


def test_inference_alias():
    client, _ = _client()
    with client:
        r = _post(client, fmt="srt", path="/inference")
    assert r.status_code == 200
    assert "00:00:00,000 --> 00:00:01,500" in r.text


def test_auth_required_when_key_set():
    client, _ = _client(api_key="secret")
    with client:
        denied = _post(client)
        assert denied.status_code == 401
        ok = _post(client, headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200


def test_no_auth_when_key_unset():
    client, _ = _client(api_key=None)
    with client:
        # An Authorization header is simply ignored when no key is configured.
        r = _post(client, headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 200


# --- production wiring (no real pool: transcribe_file / uvicorn are stubbed) ---


async def test_transcriber_applies_language_and_reuses_backend(monkeypatch):
    captured = {}

    async def fake_transcribe_file(source, cfg, *, progress, backend):
        captured.update(source=source, language=cfg.language, backend=backend, progress=progress)
        return Transcript([Segment(0.0, 1.0, "hi")])

    monkeypatch.setattr(api_mod, "transcribe_file", fake_transcribe_file)
    sentinel_backend = object()
    t = Transcriber(Config(model=Path("m")), sentinel_backend)

    await t.transcribe(Path("a.wav"), language="fr")
    assert captured["language"] == "fr"          # per-request override applied
    assert captured["backend"] is sentinel_backend  # shared warm pool reused
    assert captured["progress"] is False


def test_parse_args_defaults():
    args = api_mod._parse_args(["-m", "model.bin"])
    assert (args.host, args.port, args.threads) == ("0.0.0.0", 8000, 2)


def test_main_without_model_returns_2(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert api_mod.main([]) == 2


def test_create_app_requires_cfg_or_transcriber():
    app = create_app()  # neither supplied
    with pytest.raises(RuntimeError, match="cfg or transcriber"):
        with TestClient(app):  # entering runs the lifespan, which raises
            pass


def test_command_error_becomes_400():
    class Failing:
        async def transcribe(self, source, *, language=None):
            raise CommandError(["ffmpeg"], 1, "moov atom not found")

    app = create_app(transcriber=Failing())
    with TestClient(app) as client:
        r = _post(client)
    assert r.status_code == 400
    assert "moov atom" in r.json()["detail"]


def test_production_lifespan_starts_and_stops_pool(monkeypatch):
    """create_app(cfg=...) spins one shared pool up at startup, down at shutdown."""
    events = []

    class FakeBackend:
        def __init__(self, cfg):
            events.append("init")

        async def start(self):
            events.append("start")

        async def stop(self):
            events.append("stop")

    async def fake_transcribe_file(source, cfg, *, progress, backend):
        assert isinstance(backend, FakeBackend)   # request used the shared pool
        return Transcript([Segment(0.0, 1.0, "ok")])

    monkeypatch.setattr(api_mod, "ServerBackend", FakeBackend)
    monkeypatch.setattr(api_mod, "transcribe_file", fake_transcribe_file)

    app = create_app(cfg=Config(model=Path("m")))
    with TestClient(app) as client:        # entering runs lifespan startup
        assert _post(client).json() == {"text": "ok"}
    assert events == ["init", "start", "stop"]   # pool torn down on shutdown


def test_main_builds_app_and_serves(monkeypatch):
    calls = {}
    monkeypatch.setattr(api_mod, "create_app", lambda **kw: ("APP", kw))
    # main() does `import uvicorn` locally; patch its run on the real module.
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.update(app=app, **kw))
    monkeypatch.setenv("WHISPER_BATCH_API_KEY", "k")

    rc = api_mod.main(["-m", "model.bin", "--port", "9001", "-w", "4"])
    assert rc == 0
    assert calls["app"][0] == "APP"
    assert calls["app"][1]["api_key"] == "k"      # api_key threaded from env
    assert calls["app"][1]["cfg"].workers == 4    # -w applied to the pool config
    assert calls["port"] == 9001
