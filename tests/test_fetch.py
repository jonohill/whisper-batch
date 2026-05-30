"""Tests for name-based model resolution and download (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_batch import fetch


@pytest.mark.parametrize(
    "name,expected",
    [
        ("base.en", "ggml-base.en.bin"),
        ("small", "ggml-small.bin"),
        ("  base.en  ", "ggml-base.en.bin"),
        ("ggml-base.en.bin", "ggml-base.en.bin"),  # already a filename
        ("ggml-small", "ggml-small.bin"),
    ],
)
def test_model_filename(name: str, expected: str) -> None:
    assert fetch.model_filename(name) == expected


def test_model_url() -> None:
    assert fetch.model_url("base.en") == (
        f"{fetch.GGML_BASE_URL}/ggml-base.en.bin"
    )


def test_resolve_explicit_existing_wins(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "mine.bin"
    model.write_bytes(b"x")
    # Even with a name set, an existing explicit path is used as-is.
    monkeypatch.setenv("WHISPER_MODEL_NAME", "base.en")
    assert fetch.resolve_model(str(model)) == model


def test_resolve_none_without_path_or_name(monkeypatch) -> None:
    monkeypatch.delenv("WHISPER_MODEL_NAME", raising=False)
    assert fetch.resolve_model(None) is None


def test_resolve_missing_explicit_passes_through(monkeypatch) -> None:
    monkeypatch.delenv("WHISPER_MODEL_NAME", raising=False)
    # No name -> a missing explicit path is returned so the caller can report it.
    assert fetch.resolve_model("/nope/model.bin") == Path("/nope/model.bin")


def test_resolve_downloads_by_name(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(name: str, dest: Path) -> Path:
        calls.append((name, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"model")
        return dest

    monkeypatch.setattr(fetch, "download_model", fake_download)
    monkeypatch.setenv("WHISPER_MODEL_NAME", "base.en")
    monkeypatch.setenv("WHISPER_MODELS_DIR", str(tmp_path))

    got = fetch.resolve_model(None)

    assert got == tmp_path / "ggml-base.en.bin"
    assert calls == [("base.en", tmp_path / "ggml-base.en.bin")]


def test_resolve_skips_download_when_cached(tmp_path: Path, monkeypatch) -> None:
    cached = tmp_path / "ggml-base.en.bin"
    cached.write_bytes(b"already here")

    def boom(name: str, dest: Path) -> Path:  # pragma: no cover
        raise AssertionError("should not download when the file exists")

    monkeypatch.setattr(fetch, "download_model", boom)
    monkeypatch.setenv("WHISPER_MODEL_NAME", "base.en")
    monkeypatch.setenv("WHISPER_MODELS_DIR", str(tmp_path))

    assert fetch.resolve_model(None) == cached


def test_download_is_atomic(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "sub" / "ggml-base.en.bin"

    class FakeResp:
        def __init__(self) -> None:
            self._data = [b"abc", b"def", b""]

        def read(self, _n: int) -> bytes:
            return self._data.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda url: FakeResp())

    out = fetch.download_model("base.en", dest)

    assert out == dest
    assert dest.read_bytes() == b"abcdef"
    # No leftover temp file.
    assert not dest.with_name(dest.name + ".part").exists()
