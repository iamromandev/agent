"""The autouse fixture in conftest.py keeps every test off the real user data folder (AC19)."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest
from src.agent.session.paths import sessions_root
from src.config import get_settings

from tests.conftest import REAL_DATA_DIRS, RealDataDirWriteError


def test_settings_point_at_a_temporary_sessions_folder(tmp_path_factory: pytest.TempPathFactory) -> None:
    configured = get_settings().sessions_dir
    assert configured is not None
    assert Path(configured).is_relative_to(tmp_path_factory.getbasetemp())
    for real in REAL_DATA_DIRS:
        assert not Path(configured).resolve().is_relative_to(real)


def test_the_default_root_is_a_temporary_folder_too(tmp_path_factory: pytest.TempPathFactory) -> None:
    assert sessions_root(None).is_relative_to(tmp_path_factory.getbasetemp())


def test_the_settings_cache_is_cleared_for_each_test(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    monkeypatch.setenv("AGENT_SESSIONS_DIR", "/elsewhere")
    get_settings.cache_clear()
    assert get_settings().sessions_dir == "/elsewhere"
    assert first.sessions_dir != "/elsewhere"


def test_creating_a_folder_under_the_real_data_dir_is_refused() -> None:
    target = REAL_DATA_DIRS[0] / "agent-test-guard-folder"
    try:
        with pytest.raises(RealDataDirWriteError):
            target.mkdir(parents=True)
    finally:
        if target.exists():  # the guard failed: do not leave litter in the real folder
            target.rmdir()


def test_creating_a_file_under_the_real_data_dir_is_refused() -> None:
    target = REAL_DATA_DIRS[0] / "agent-test-guard-file"
    try:
        with pytest.raises(RealDataDirWriteError):
            os.open(target, os.O_WRONLY | os.O_CREAT, 0o600)
    finally:
        if target.exists():  # the guard failed: do not leave litter in the real folder
            target.unlink()


def test_reading_under_the_real_data_dir_is_not_refused() -> None:
    with contextlib.suppress(FileNotFoundError):
        os.close(os.open(REAL_DATA_DIRS[0], os.O_RDONLY))


def test_a_symlink_under_the_real_data_dir_is_refused(tmp_path: Path) -> None:
    target = REAL_DATA_DIRS[0] / "agent-test-guard-link"
    try:
        with pytest.raises(RealDataDirWriteError):
            os.symlink(tmp_path, target)
    finally:
        if target.is_symlink():  # the guard failed: do not leave litter in the real folder
            target.unlink()


def test_a_rename_or_replace_into_or_out_of_the_real_data_dir_is_refused(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"  # if a guard fails, the call fails harmlessly with FileNotFoundError
    inside = REAL_DATA_DIRS[0] / "agent-test-guard-missing"
    for move in (os.rename, os.replace):
        with pytest.raises(RealDataDirWriteError):
            move(missing, inside)
        with pytest.raises(RealDataDirWriteError):
            move(inside, missing)


def test_a_chmod_under_the_real_data_dir_is_refused() -> None:
    target = REAL_DATA_DIRS[0] / "agent-test-guard-missing"  # never exists, so a failed guard is harmless
    with pytest.raises(RealDataDirWriteError):
        os.chmod(target, 0o700)


@pytest.mark.parametrize("mode", ["w", "a", "x", "wb", "r+", "w+"])
def test_open_in_a_write_mode_under_the_real_data_dir_is_refused(mode: str) -> None:
    target = REAL_DATA_DIRS[0] / "agent-test-guard-open"
    try:
        with pytest.raises(RealDataDirWriteError):
            open(target, mode)  # noqa: SIM115
    finally:
        if target.exists():  # the guard failed: do not leave litter in the real folder
            target.unlink()


def test_path_write_text_under_the_real_data_dir_is_refused() -> None:
    target = REAL_DATA_DIRS[0] / "agent-test-guard-write-text"
    try:
        with pytest.raises(RealDataDirWriteError):
            target.write_text("x")
    finally:
        if target.exists():  # the guard failed: do not leave litter in the real folder
            target.unlink()


def test_open_for_reading_and_writes_elsewhere_still_work(tmp_path: Path) -> None:
    target = tmp_path / "ok.txt"
    with open(target, "w") as file:
        file.write("x")
    os.chmod(target, 0o600)
    os.symlink(target, tmp_path / "link")
    os.replace(target, tmp_path / "moved")
    os.rename(tmp_path / "moved", tmp_path / "back")
    with open(tmp_path / "back") as file:
        assert file.read() == "x"
    with contextlib.suppress(FileNotFoundError), open(REAL_DATA_DIRS[0] / "agent-test-guard-read") as file:
        file.read()
