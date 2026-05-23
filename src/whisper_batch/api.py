"""A long-running HTTP transcription service over the warm whisper-server pool.

This wraps the same pipeline the CLI uses (silence-aware chunking -> parallel
transcription across a warm pool -> assembly) behind an HTTP endpoint, and keeps
the pool warm for the lifetime of the process so each request reuses it instead
of paying model-load startup.

The public interface mirrors OpenAI's audio-transcriptions API
(``POST /v1/audio/transcriptions``), which is the de facto standard the wider
ecosystem (openai SDKs, GUIs, faster-whisper-server, LocalAI, ...) speaks. A
``/inference`` alias is provided so existing whisper.cpp-server clients work too.

Optional dependency: this module needs the ``server`` extra
(``pip install whisper-batch[server]``); the CLI itself stays dependency-free.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

try:
    from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse, Response
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without extras
    raise ModuleNotFoundError(
        "the HTTP server needs the 'server' extra: pip install 'whisper-batch[server]'"
    ) from exc

from . import output
from .config import Config
from .pipeline import transcribe_file
from .proc import CommandError
from .server import ServerBackend
from .types import Transcript

log = logging.getLogger(__name__)

# OpenAI's response_format values; verbose_json carries segment timestamps.
_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}


class Transcriber:
    """Adapts the file pipeline to a shared, already-warm backend.

    One instance is held for the process lifetime; every request runs through
    the same pool. ``language`` is applied per request without disturbing the
    pool (the warm servers auto-detect; the override rides on the inference call).
    """

    def __init__(self, cfg: Config, backend: ServerBackend) -> None:
        self.cfg = cfg
        self.backend = backend

    async def transcribe(self, source: Path, *, language: str | None = None) -> Transcript:
        cfg = replace(self.cfg, language=language) if language else self.cfg
        return await transcribe_file(source, cfg, progress=False, backend=self.backend)


async def _require_auth(request: Request) -> None:
    """Enforce a bearer token iff one is configured (open otherwise)."""
    expected: str | None = request.app.state.api_key
    if not expected:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


async def _save_upload(file: UploadFile) -> Path:
    """Stream *file* to a fresh temp dir (ffmpeg needs a real path). Caller cleans up."""
    tmpdir = Path(tempfile.mkdtemp(prefix="whisper_api_"))
    suffix = Path(file.filename or "").suffix
    dest = tmpdir / f"upload{suffix}"
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return dest


def _render(transcript: Transcript, response_format: str, language: str | None) -> Response:
    """Serialise *transcript* in the requested OpenAI response_format."""
    if response_format == "text":
        return PlainTextResponse(transcript.text)
    if response_format == "srt":
        return PlainTextResponse(output.render_srt(transcript), media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(output.render_vtt(transcript), media_type="text/vtt")
    if response_format == "verbose_json":
        segs = transcript.segments
        return JSONResponse(
            {
                "task": "transcribe",
                "language": language or "auto",
                "duration": round(max((s.end for s in segs), default=0.0), 3),
                "text": transcript.text,
                "segments": [
                    {
                        "id": i,
                        "seek": 0,
                        "start": round(s.start, 3),
                        "end": round(s.end, 3),
                        "text": s.text,
                    }
                    for i, s in enumerate(segs)
                ],
            }
        )
    # default: "json"
    return JSONResponse({"text": transcript.text})


def _register_routes(app: FastAPI) -> None:
    async def _handle(
        request: Request,
        file: UploadFile,
        language: str | None,
        response_format: str,
    ) -> Response:
        if response_format not in _FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported response_format {response_format!r}; "
                f"expected one of {sorted(_FORMATS)}",
            )
        transcriber: Transcriber = request.app.state.transcriber
        audio_path = await _save_upload(file)
        try:
            transcript = await transcriber.transcribe(audio_path, language=language)
        except CommandError as exc:
            # A failed ffmpeg/whisper invocation usually means bad input.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(audio_path.parent, ignore_errors=True)
        return _render(transcript, response_format, language)

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(_require_auth)])
    async def transcriptions(  # noqa: D401 - FastAPI route
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),  # accepted for compatibility; pool model is fixed
        language: str | None = Form(None),
        prompt: str | None = Form(None),  # noqa: ARG001 - accepted, not yet applied
        response_format: str = Form("json"),
        temperature: float = Form(0.0),  # noqa: ARG001 - accepted, not yet applied
    ) -> Response:
        return await _handle(request, file, language, response_format)

    # Drop-in alias for clients pointed at a whisper.cpp server's /inference.
    @app.post("/inference", dependencies=[Depends(_require_auth)])
    async def inference(
        request: Request,
        file: UploadFile = File(...),
        language: str | None = Form(None),
        response_format: str = Form("json"),
    ) -> Response:
        return await _handle(request, file, language, response_format)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}


def create_app(
    *,
    transcriber: Transcriber | None = None,
    cfg: Config | None = None,
    api_key: str | None = None,
) -> FastAPI:
    """Build the app.

    In production pass *cfg*: the lifespan starts one shared warm pool and tears
    it down on shutdown. Tests pass a ready-made *transcriber* to skip spawning
    real whisper-server processes.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.api_key = api_key
        if transcriber is not None:
            app.state.transcriber = transcriber
            yield
            return
        if cfg is None:
            raise RuntimeError("create_app needs either cfg or transcriber")
        backend = ServerBackend(cfg)
        await backend.start()
        app.state.transcriber = Transcriber(cfg, backend)
        try:
            yield
        finally:
            await backend.stop()

    app = FastAPI(title="whisper-batch", version="0.1.0", lifespan=lifespan)
    _register_routes(app)
    return app


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="whisper-batch-server",
        description="OpenAI-compatible HTTP transcription service over a warm whisper.cpp pool.",
    )
    p.add_argument(
        "-m", "--model", default=os.environ.get("WHISPER_MODEL"),
        help="path to a ggml whisper model (or set WHISPER_MODEL)",
    )
    p.add_argument("--host", default="0.0.0.0", help="HTTP bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="HTTP bind port (default: 8000)")
    p.add_argument("-w", "--workers", type=int, help="warm whisper-server instances")
    p.add_argument("-t", "--threads", type=int, default=2, help="threads per worker")
    p.add_argument("-l", "--language", help="default language code (default: auto-detect)")
    p.add_argument("--whisper-server-bin", default="whisper-server",
                   help="whisper.cpp server binary")
    p.add_argument("--server-host", default="127.0.0.1", help="host for the internal pool")
    p.add_argument("--server-port", type=int, default=18080,
                   help="base port for the internal pool (uses port .. port+workers-1)")
    p.add_argument("--no-gpu", action="store_true", help="disable GPU (passes -ng)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = _parse_args(argv)
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.model:
        print("error: no model given (use -m or set WHISPER_MODEL)")
        return 2

    cfg = Config(
        model=Path(args.model),
        whisper_server_bin=args.whisper_server_bin,
        server_host=args.server_host,
        server_port=args.server_port,
        language=args.language,
        threads=args.threads,
        no_gpu=args.no_gpu,
    )
    if args.workers:
        cfg.workers = args.workers

    app = create_app(cfg=cfg, api_key=os.environ.get("WHISPER_BATCH_API_KEY"))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
