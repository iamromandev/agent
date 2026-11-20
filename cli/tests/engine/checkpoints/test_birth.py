"""Birth times: `/undo` tells a file or folder the turn made from one that was there before, on Linux too."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from src.engine.checkpoints import birth
from src.engine.checkpoints.birth import birth_time, settle

KEEPS_BIRTH_TIMES = pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"), reason="macOS and Linux keep birth times"
)


def born(path: Path) -> float | None:
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return birth_time(parent, path.name, os.lstat(path))
    finally:
        os.close(parent)


def pause() -> float:
    """A moment between two file operations, past the kernel's coarse clock tick."""
    time.sleep(0.05)
    mark = time.time()
    time.sleep(0.05)
    return mark


def test_no_birth_time_is_none_so_the_caller_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "f.txt").write_text("f")
    monkeypatch.setattr(birth, "_statx", None)
    info = os.stat_result(tuple(os.lstat(tmp_path / "f.txt"))[:10])  # the ten fields, no st_birthtime

    assert born_with(tmp_path, "f.txt", info) is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="statx is the Linux path")
def test_an_entry_swapped_after_the_lstat_gives_no_birth_time(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("f")
    (tmp_path / "other.txt").write_text("o")
    info = os.lstat(tmp_path / "f.txt")
    (tmp_path / "other.txt").replace(tmp_path / "f.txt")  # same name, another file

    found = born_with(tmp_path, "f.txt", info)

    assert os.lstat(tmp_path / "f.txt").st_ino != info.st_ino and found is None


def born_with(folder: Path, name: str, info: os.stat_result) -> float | None:
    parent = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return birth_time(parent, name, info)
    finally:
        os.close(parent)


@KEEPS_BIRTH_TIMES
@pytest.mark.parametrize("attempt", range(20))
def test_after_settle_a_file_made_at_once_is_born_after_the_mark(tmp_path: Path, attempt: int) -> None:
    """Linux stamps files from a coarse clock up to a tick behind `time.time()`: without `settle`, a file the turn
    writes right after the `before` snapshot can look older than the snapshot."""
    mark = time.time()
    settle(mark)
    (tmp_path / "new.txt").write_text("new")

    found = born(tmp_path / "new.txt")

    assert found is not None and found > mark


def test_settle_returns_within_its_limit_whatever_the_mark() -> None:
    started = time.monotonic()

    settle(time.time() + 3600)  # a mark the clock will not pass: it gives up

    assert time.monotonic() - started < birth.SETTLE_LIMIT + 0.5  # room for a busy CI runner
