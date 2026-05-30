"""On-demand download of ggml whisper models."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# whisper.cpp's canonical model host. Every model lives at
# {base}/ggml-{name}.bin (e.g. ggml-base.en.bin, ggml-small.bin).
GGML_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Where name-resolved models are written when WHISPER_MODELS_DIR is unset. Matches
# the image's VOLUME so a download persists across restarts on a mounted volume.
DEFAULT_MODELS_DIR = "/models"


def model_filename(name: str) -> str:
    """``base.en`` -> ``ggml-base.en.bin`` (pass-through if already a filename)."""
    name = name.strip()
    if name.startswith("ggml-"):
        name = name[len("ggml-") :]
    if name.endswith(".bin"):
        name = name[: -len(".bin")]
    return f"ggml-{name}.bin"


def model_url(name: str) -> str:
    return f"{GGML_BASE_URL}/{model_filename(name)}"


def download_model(name: str, dest: Path) -> Path:
    """Download model ``name`` to ``dest``"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = model_url(name)
    tmp = dest.with_name(dest.name + ".part")
    log.info("downloading model %s from %s", name, url)
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # noqa: S310
            # Copy in fixed-size blocks to bound memory on large models.
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    log.info("model ready at %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def resolve_model(explicit: str | None) -> Path | None:
    """Resolve the model to load, downloading by name if needed.

    Precedence:
      1. ``explicit`` path (``-m``/``WHISPER_MODEL``) that already exists.
      2. ``WHISPER_MODEL_NAME`` -> ``{WHISPER_MODELS_DIR}/ggml-<name>.bin``,
         downloaded on first use.
      3. ``explicit`` path even if missing (so the server reports it as before).

    Returns ``None`` only when neither an explicit path nor a name was given.
    """
    if explicit and Path(explicit).exists():
        return Path(explicit)

    name = os.environ.get("WHISPER_MODEL_NAME")
    if name:
        dest_dir = Path(os.environ.get("WHISPER_MODELS_DIR") or DEFAULT_MODELS_DIR)
        dest = dest_dir / model_filename(name)
        if not dest.exists():
            download_model(name, dest)
        return dest

    return Path(explicit) if explicit else None
