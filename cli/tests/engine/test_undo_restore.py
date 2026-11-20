"""Restoring a turn: bytes, deletions, conflicts, confinement, and the notice text."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.engine.undo import CHANGED, TOO_LARGE, UndoLog, UndoTurn, restore_turn


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


def write(log: UndoLog, path: Path, data: bytes) -> None:
    log.before(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    log.after(path)


def finished(log: UndoLog) -> UndoTurn:
    log.finish()
    turn = log.pop()
    assert turn is not None
    return turn


def run(workspace: Path, *steps: tuple[str, bytes], seed: dict[str, bytes] | None = None) -> UndoTurn:
    for name, data in (seed or {}).items():
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_bytes(data)
    log = UndoLog(workspace)
    log.begin(0, "prompt")
    for name, data in steps:
        write(log, workspace / name, data)
    return finished(log)


def test_a_created_folder_that_is_not_empty_is_kept(workspace: Path) -> None:
    turn = run(workspace, ("new/f.txt", b"x"))
    (workspace / "new" / "other.txt").write_text("user's")

    restore_turn(turn, workspace)

    assert not (workspace / "new" / "f.txt").exists()
    assert (workspace / "new" / "other.txt").read_text() == "user's"


def test_a_file_deleted_afterwards_is_skipped_when_the_turn_edited_it(workspace: Path) -> None:
    turn = run(workspace, ("a.py", b"new"), seed={"a.py": b"old"})
    (workspace / "a.py").unlink()

    result = restore_turn(turn, workspace)

    assert not (workspace / "a.py").exists()
    assert result.entries == [("skipped", "a.py", CHANGED)]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_a_file_that_cannot_be_restored_is_skipped_with_the_reason_and_the_rest_go_on(workspace: Path) -> None:
    turn = run(workspace, ("locked/a.py", b"A2"), ("b.py", b"B2"), seed={"locked/a.py": b"A", "b.py": b"B"})
    (workspace / "locked").chmod(0o500)
    try:
        result = restore_turn(turn, workspace)
    finally:
        (workspace / "locked").chmod(0o700)

    assert (workspace / "locked" / "a.py").read_bytes() == b"A2"
    assert (workspace / "b.py").read_bytes() == b"B"
    assert result.entries[0][:2] == ("skipped", "locked/a.py")
    assert result.entries[0][2]
    assert result.entries[1] == ("restored", "b.py", None)
    assert sorted(p.name for p in (workspace / "locked").iterdir()) == ["a.py"]


def test_a_path_outside_the_workspace_is_refused(tmp_path: Path, workspace: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"old")
    log = UndoLog(workspace)
    log.begin(0, "p")
    write(log, outside, b"new")
    turn = finished(log)

    result = restore_turn(turn, workspace)

    assert outside.read_bytes() == b"new"
    assert result.entries[0][0] == "skipped"


def test_a_too_large_file_is_skipped_and_the_small_one_restored(workspace: Path) -> None:
    (workspace / "big.txt").write_bytes(b"x" * 20)
    (workspace / "small.txt").write_bytes(b"s")
    log = UndoLog(workspace, file_limit=10)
    log.begin(0, "p")
    write(log, workspace / "big.txt", b"y" * 20)
    write(log, workspace / "small.txt", b"t")
    turn = finished(log)

    result = restore_turn(turn, workspace)

    assert (workspace / "big.txt").read_bytes() == b"y" * 20
    assert (workspace / "small.txt").read_bytes() == b"s"
    assert result.entries == [("skipped", "big.txt", TOO_LARGE), ("restored", "small.txt", None)]


# --- notices -----------------------------------------------------------------------------------


def test_a_created_folder_swapped_for_a_symlink_never_removes_a_folder_outside(
    tmp_path: Path, workspace: Path
) -> None:
    outside = tmp_path / "outside"
    (outside / "b").mkdir(parents=True)
    turn = run(workspace, ("a/b/f.txt", b"x"))
    (workspace / "a" / "b" / "f.txt").unlink()
    (workspace / "a" / "b").rmdir()
    (workspace / "a").rmdir()
    (workspace / "a").symlink_to(outside)

    restore_turn(turn, workspace)

    assert (outside / "b").is_dir()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_a_read_only_file_is_skipped_as_read_only_and_the_rest_go_on(workspace: Path) -> None:
    turn = run(workspace, ("a.py", b"A2"), ("b.py", b"B2"), seed={"a.py": b"A", "b.py": b"B"})
    (workspace / "a.py").chmod(0o444)

    result = restore_turn(turn, workspace)

    assert (workspace / "a.py").read_bytes() == b"A2"
    assert (workspace / "b.py").read_bytes() == b"B"
    assert result.entries == [("skipped", "a.py", "read-only"), ("restored", "b.py", None)]


def test_restoring_a_hard_linked_file_keeps_the_link(workspace: Path) -> None:
    turn = run(workspace, ("a.py", b"new"), seed={"a.py": b"old"})
    link = workspace / "link.py"
    os.link(workspace / "a.py", link)

    result = restore_turn(turn, workspace)

    assert result.entries == [("restored", "a.py", None)]
    assert link.read_bytes() == b"old"
    assert os.path.samefile(workspace / "a.py", link)
