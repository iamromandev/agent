"""Model status, formatting and the warm-up PNG of `/images` (AC1, AC2, NFR2)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from core.engine.image_pull import (
    ModelStatus,
    format_elapsed,
    format_gb,
    format_status,
    grey_png,
    model_status,
    repo_folder,
)
from core.engine.imagegen import png_size


def _env(tmp_path: Path) -> dict[str, str]:
    return {"HF_HUB_CACHE": str(tmp_path / "hub")}


def _snapshot(tmp_path: Path, model: str, blob_sizes: list[int], incomplete: bool = False) -> None:
    folder = repo_folder(model, _env(tmp_path))
    (folder / "blobs").mkdir(parents=True)
    (folder / "snapshots" / "abc").mkdir(parents=True)
    for index, size in enumerate(blob_sizes):
        blob = folder / "blobs" / f"b{index}"
        blob.write_bytes(b"x" * size)
        (folder / "snapshots" / "abc" / f"f{index}").write_text("link")
    if incomplete:
        (folder / "blobs" / "zz.incomplete").write_bytes(b"yy")


def test_repo_folder_follows_the_hub_cache(tmp_path: Path) -> None:
    assert repo_folder("a/b", _env(tmp_path)) == tmp_path / "hub" / "models--a--b"


def test_complete_snapshot_is_cached_with_its_size(tmp_path: Path) -> None:
    _snapshot(tmp_path, "a/b", [100, 50])
    assert model_status("generate", "a/b", _env(tmp_path)) == ModelStatus("generate", "a/b", "cached", 150)


def test_nothing_is_missing(tmp_path: Path) -> None:
    assert model_status("edit", "a/b", _env(tmp_path)) == ModelStatus("edit", "a/b", "missing", None)


def test_incomplete_blob_is_partial(tmp_path: Path) -> None:
    _snapshot(tmp_path, "a/b", [100], incomplete=True)
    status = model_status("edit", "a/b", _env(tmp_path))
    assert status.state == "partial"
    assert status.size == 102


def test_empty_snapshot_folder_is_partial(tmp_path: Path) -> None:
    folder = repo_folder("a/b", _env(tmp_path))
    (folder / "snapshots" / "abc").mkdir(parents=True)
    assert model_status("edit", "a/b", _env(tmp_path)).state == "partial"


def test_a_folder_is_a_local_folder(tmp_path: Path) -> None:
    assert model_status("edit", str(tmp_path), _env(tmp_path)) == ModelStatus("edit", str(tmp_path), "local folder", None)


def test_no_command_is_unavailable() -> None:
    assert model_status("edit", None).state == "unavailable"


def test_format_status_lists_rows_and_the_pull(tmp_path: Path) -> None:
    rows = [
        ModelStatus("generate", "a/b", "cached", 4_000_000_000),
        ModelStatus("edit", "c/d", "missing", None),
        ModelStatus("edit", "", "unavailable", None),
    ]
    text = format_status(rows, "edit: loading model… 0:03")
    lines = text.splitlines()
    assert lines[0] == "generate: a/b - cached, 4.0 GB"
    assert lines[1] == "edit: c/d - missing"
    assert lines[2] == "edit: unavailable (mflux command not found)"
    assert lines[3] == "pulling edit: loading model… 0:03"
    assert "pulling" not in format_status(rows, None)


def test_format_status_escapes_the_name() -> None:
    text = format_status([ModelStatus("generate", "a\x1b[2J", "missing", None)], None)
    assert "\x1b" not in text


def test_format_gb_and_elapsed() -> None:
    assert format_gb(15_000_000_000) == "15.0 GB"
    assert format_elapsed(1450) == "24m 10s"
    assert format_elapsed(45) == "45s"


def test_grey_png_is_a_valid_png_of_the_size() -> None:
    assert png_size(grey_png(256, 256)) == (256, 256)


def test_status_of_a_big_cache_only_stats(tmp_path: Path) -> None:
    folder = repo_folder("a/b", _env(tmp_path))
    (folder / "blobs").mkdir(parents=True)
    (folder / "snapshots" / "abc").mkdir(parents=True)
    for index in range(20):
        blob = folder / "blobs" / f"b{index}"
        with blob.open("wb") as handle:
            handle.truncate(1_000_000_000)
        (folder / "snapshots" / "abc" / f"f{index}").symlink_to(blob)
    started = time.monotonic()
    status = model_status("generate", "a/b", _env(tmp_path))
    assert time.monotonic() - started < 0.2
    assert status.size == 20_000_000_000


@pytest.mark.parametrize("model", ["a/b", "x/y"])
def test_status_never_raises_on_an_unreadable_cache(tmp_path: Path, model: str) -> None:
    assert model_status("generate", model, {"HF_HUB_CACHE": str(tmp_path / "nope" / "deeper")}).state == "missing"
