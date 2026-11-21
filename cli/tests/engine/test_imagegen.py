"""Discovery of mflux, the model cache check and the PNG check (AC1, AC2, AC8)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.imagegen import COMMAND, find_mflux, model_cached, png_size

from tests.image_fakes import fake_mflux, png


def _python(folder: Path) -> Path:
    python = folder / "python"
    python.write_text("")
    return python


def test_find_mflux_returns_the_command_next_to_the_interpreter(tmp_path: Path) -> None:
    command = fake_mflux(tmp_path)
    assert find_mflux("darwin", "arm64", _python(tmp_path)) == command


def test_find_mflux_is_none_when_the_command_is_missing(tmp_path: Path) -> None:
    assert find_mflux("darwin", "arm64", _python(tmp_path)) is None


def test_find_mflux_is_none_when_the_file_is_not_executable(tmp_path: Path) -> None:
    (tmp_path / COMMAND).write_text("")
    assert find_mflux("darwin", "arm64", _python(tmp_path)) is None


def test_find_mflux_is_none_for_a_directory_of_that_name(tmp_path: Path) -> None:
    (tmp_path / COMMAND).mkdir()
    assert find_mflux("darwin", "arm64", _python(tmp_path)) is None


@pytest.mark.parametrize("system, machine", [("linux", "arm64"), ("darwin", "x86_64"), ("win32", "AMD64")])
def test_find_mflux_is_none_on_other_platforms_even_with_the_command(tmp_path: Path, system: str, machine: str) -> None:
    fake_mflux(tmp_path)
    assert find_mflux(system, machine, _python(tmp_path)) is None


def _cache(root: Path, repo: str, snapshots: list[str]) -> None:
    folder = root / f"models--{repo.replace('/', '--')}" / "snapshots"
    folder.mkdir(parents=True)
    for name in snapshots:
        (folder / name).mkdir()
        (folder / name / "weights.bin").write_text("x")


def test_model_cached_reads_hf_hub_cache_first(tmp_path: Path) -> None:
    _cache(tmp_path / "hub", "a/b", ["abc"])
    env = {"HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HOME": str(tmp_path / "other")}
    assert model_cached("a/b", env) is True


def test_model_cached_falls_back_to_hf_home_hub(tmp_path: Path) -> None:
    _cache(tmp_path / "home" / "hub", "a/b", ["abc"])
    assert model_cached("a/b", {"HF_HOME": str(tmp_path / "home")}) is True


def test_model_cached_defaults_to_the_users_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _cache(tmp_path / ".cache" / "huggingface" / "hub", "a/b", ["abc"])
    assert model_cached("a/b", {}) is True


def test_model_cached_is_false_for_another_model_or_an_empty_snapshot_folder(tmp_path: Path) -> None:
    _cache(tmp_path, "a/b", [])
    env = {"HF_HUB_CACHE": str(tmp_path)}
    assert model_cached("a/b", env) is False
    assert model_cached("a/c", env) is False


def test_model_cached_counts_an_existing_local_folder(tmp_path: Path) -> None:
    assert model_cached(str(tmp_path), {"HF_HUB_CACHE": str(tmp_path / "none")}) is True
    assert model_cached(str(tmp_path / "missing"), {"HF_HUB_CACHE": str(tmp_path / "none")}) is False


def test_png_size_reads_the_ihdr() -> None:
    assert png_size(png(512, 256)) == (512, 256)


def test_png_size_is_none_for_a_jpeg_truncated_data_and_empty_bytes() -> None:
    assert png_size(b"\xff\xd8\xff\xe0" + b"\0" * 50) is None
    assert png_size(png(8, 8)[:20]) is None
    assert png_size(b"") is None


def test_png_size_is_none_when_the_first_chunk_is_not_ihdr() -> None:
    data = bytearray(png(8, 8))
    data[12:16] = b"IDAT"
    assert png_size(bytes(data)) is None


def test_png_size_is_none_for_a_zero_size() -> None:
    assert png_size(png(0, 0)) is None
