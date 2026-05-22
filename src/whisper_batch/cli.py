"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .config import Config
from .output import WRITERS, write_outputs
from .pipeline import transcribe_file
from .proc import CommandError


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="whisper-batch",
        description="Parallel whisper.cpp transcription via silence-aware chunking.",
    )
    p.add_argument("input", type=Path, help="source audio/video file")
    p.add_argument(
        "-m", "--model", default=os.environ.get("WHISPER_MODEL"),
        help="path to a ggml whisper model (or set WHISPER_MODEL)",
    )
    p.add_argument(
        "-o", "--output", type=Path,
        help="output path prefix (default: alongside the input file)",
    )
    p.add_argument(
        "-f", "--format", default="txt,srt",
        help=f"comma-separated formats from {sorted(WRITERS)} (default: txt,srt)",
    )
    p.add_argument("-l", "--language", help="force language code (default: auto-detect)")
    p.add_argument("-w", "--workers", type=int, help="concurrent whisper.cpp workers")
    p.add_argument("-t", "--threads", type=int, default=2, help="threads per worker")
    p.add_argument("--max-chunk", type=float, default=28.0, help="max chunk length, s")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="context pad each side of a chunk, s (de-duplicated at assembly)")
    p.add_argument("--silence-noise", type=float, default=-30.0, help="silence floor, dB")
    p.add_argument("--min-silence", type=float, default=0.5, help="min silence length, s")
    p.add_argument("--whisper-server-bin", default="whisper-server",
                   help="whisper.cpp server binary")
    p.add_argument("--server-host", default="127.0.0.1", help="host for the server pool")
    p.add_argument("--server-port", type=int, default=18080,
                   help="base port for the server pool (uses port .. port+workers-1)")
    p.add_argument("--no-gpu", action="store_true",
                   help="disable GPU (passes -ng to whisper.cpp); useful on CPU-bound hosts")
    p.add_argument("--keep-temp", action="store_true", help="keep intermediate files")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)],
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.model:
        print("error: no model given (use -m or set WHISPER_MODEL)", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    unknown = [f for f in formats if f not in WRITERS]
    if unknown:
        print(f"error: unknown format(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    cfg = Config(
        model=Path(args.model),
        whisper_server_bin=args.whisper_server_bin,
        server_host=args.server_host,
        server_port=args.server_port,
        language=args.language,
        threads=args.threads,
        max_chunk_s=args.max_chunk,
        overlap_s=args.overlap,
        silence_noise_db=args.silence_noise,
        min_silence_s=args.min_silence,
        keep_temp=args.keep_temp,
        no_gpu=args.no_gpu,
    )
    if args.workers:
        cfg.workers = args.workers

    out_prefix = args.output or args.input.with_suffix("")

    try:
        transcript = asyncio.run(transcribe_file(args.input, cfg))
    except CommandError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    written = write_outputs(transcript, out_prefix, formats)
    for path in written:
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
