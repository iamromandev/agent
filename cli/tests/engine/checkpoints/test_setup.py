"""prepare_checkpoints: the cases where checkpoints are off, and the pruning of stale folders."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from src.engine.checkpoints.setup import STALE_AFTER, prepare_checkpoints, prune_stale
from src.engine.guardrails.setup import Guardrails


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def data_env(tmp_path: Path) -> dict[str, str]:
    return {"XDG_DATA_HOME": str(tmp_path / "data")}


def prepare(workspace: Path, data_env: dict[str, str], version: tuple[int, int] | None = (2, 39), enabled: bool = True):
    return prepare_checkpoints(
        workspace,
        Guardrails.off(workspace),
        {"PATH": "/usr/bin"},
        enabled=enabled,
        find=lambda env, ws: "/usr/bin/git",
        version=lambda git, env: version,
        data_env=data_env,
    )


def test_git_older_than_2_20_is_one_warning(workspace: Path, data_env: dict[str, str]) -> None:
    store, warnings = prepare(workspace, data_env, version=(2, 19))

    assert store is None
    assert len(warnings) == 1
    assert "2.19" in warnings[0]
    assert warnings[0].startswith("checkpoints off: ")


def test_the_home_folder_as_workspace_is_off(
    tmp_path: Path, data_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    store, warnings = prepare(tmp_path, data_env)

    assert store is None
    assert "home folder" in warnings[0]


def test_a_workspace_that_holds_the_data_folder_is_off(tmp_path: Path) -> None:
    store, warnings = prepare(tmp_path, {"XDG_DATA_HOME": str(tmp_path / "data")})

    assert store is None
    assert "data folder" in warnings[0]


def test_an_unwritable_data_folder_is_off(workspace: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("file")

    store, warnings = prepare(workspace, {"XDG_DATA_HOME": str(blocker)})

    assert store is None
    assert "cannot write" in warnings[0]


def dead_pid() -> int:
    pid = 4_000_000
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except PermissionError:
            pass
        pid += 1


def make_folder(root: Path, name: str, age: float) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "f").write_text("x")
    old = time.time() - age
    os.utime(folder, (old, old))
    return folder


def name(pid: int, prefix: str = "0123456789abcdef") -> str:
    return f"{prefix}-{pid}"


def test_prune_removes_old_folders_of_dead_runs_only(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    stale = make_folder(root, name(dead_pid()), STALE_AFTER + 60)
    fresh = make_folder(root, name(dead_pid(), "1123456789abcdef"), 60)
    live = make_folder(root, name(os.getpid(), "2123456789abcdef"), STALE_AFTER + 60)
    other = make_folder(root, "unrelated", STALE_AFTER + 60)

    prune_stale(root)

    assert not stale.exists()
    assert fresh.exists()
    assert live.exists()
    assert other.exists()


def test_prune_survives_a_pid_too_large_for_the_os(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    kept = make_folder(root, name(10**30), STALE_AFTER + 60)
    stale = make_folder(root, name(dead_pid(), "3123456789abcdef"), STALE_AFTER + 60)

    prune_stale(root)

    assert kept.exists()
    assert not stale.exists()


def test_only_a_git_inside_the_workspace_is_git_not_found(workspace: Path, data_env: dict[str, str]) -> None:
    (workspace / "bin").mkdir()
    (workspace / "bin" / "git").write_text("#!/bin/sh\necho git version 2.39.0\n")
    (workspace / "bin" / "git").chmod(0o755)

    store, warnings = prepare_checkpoints(
        workspace, Guardrails.off(workspace), {"PATH": str(workspace / "bin")}, enabled=True, data_env=data_env
    )

    assert store is None
    assert warnings == ["checkpoints off: git not found; /undo restores write/edit changes only"]
