"""Restoring a turn: bytes, deletions, conflicts, confinement, and the notice text."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.agent.events import Notice
from src.agent.undo import CHANGED, TOO_LARGE, UndoLog, UndoResult, UndoTurn, restore_turn, undo_notices


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


def test_an_edited_file_gets_its_bytes_back_and_a_created_file_is_deleted(workspace: Path) -> None:
    turn = run(workspace, ("a.py", b"new"), ("n.txt", b"x"), seed={"a.py": b"old"})

    result = restore_turn(turn, workspace)

    assert (workspace / "a.py").read_bytes() == b"old"
    assert not (workspace / "n.txt").exists()
    assert result.entries == [("restored", "a.py", None), ("deleted", "n.txt", None)]


def test_three_edits_of_one_file_go_back_to_the_state_before_the_first(workspace: Path) -> None:
    turn = run(workspace, ("a.py", b"1"), ("a.py", b"2"), ("a.py", b"3"), seed={"a.py": b"0"})

    restore_turn(turn, workspace)

    assert (workspace / "a.py").read_bytes() == b"0"


def test_created_folders_are_removed_but_existing_ones_are_kept(workspace: Path) -> None:
    (workspace / "src").mkdir()
    turn = run(workspace, ("new/dir/f.txt", b"x"), ("src/g.txt", b"y"))

    restore_turn(turn, workspace)

    assert not (workspace / "new").exists()
    assert (workspace / "src").is_dir()


def test_a_created_folder_that_is_not_empty_is_kept(workspace: Path) -> None:
    turn = run(workspace, ("new/f.txt", b"x"))
    (workspace / "new" / "other.txt").write_text("user's")

    restore_turn(turn, workspace)

    assert not (workspace / "new" / "f.txt").exists()
    assert (workspace / "new" / "other.txt").read_text() == "user's"


def test_a_file_changed_afterwards_is_skipped_and_the_others_are_restored(workspace: Path) -> None:
    turn = run(workspace, ("a.py", b"A2"), ("b.py", b"B2"), seed={"a.py": b"A", "b.py": b"B"})
    (workspace / "b.py").write_bytes(b"by hand")

    result = restore_turn(turn, workspace)

    assert (workspace / "a.py").read_bytes() == b"A"
    assert (workspace / "b.py").read_bytes() == b"by hand"
    assert result.entries == [("restored", "a.py", None), ("skipped", "b.py", CHANGED)]


def test_a_created_file_the_user_edited_is_kept_and_skipped(workspace: Path) -> None:
    turn = run(workspace, ("n.txt", b"made"))
    (workspace / "n.txt").write_bytes(b"edited")

    result = restore_turn(turn, workspace)

    assert (workspace / "n.txt").read_bytes() == b"edited"
    assert result.entries == [("skipped", "n.txt", CHANGED)]


def test_a_file_deleted_afterwards_is_skipped_when_the_turn_edited_it(workspace: Path) -> None:
    turn = run(workspace, ("a.py", b"new"), seed={"a.py": b"old"})
    (workspace / "a.py").unlink()

    result = restore_turn(turn, workspace)

    assert not (workspace / "a.py").exists()
    assert result.entries == [("skipped", "a.py", CHANGED)]


def test_a_created_file_deleted_afterwards_is_skipped(workspace: Path) -> None:
    turn = run(workspace, ("n.txt", b"made"))
    (workspace / "n.txt").unlink()

    result = restore_turn(turn, workspace)

    assert result.entries == [("skipped", "n.txt", CHANGED)]


def test_crlf_and_non_utf8_bytes_round_trip(workspace: Path) -> None:
    original = b"a\r\nb\r\n\xff\xfe\x00 bin"
    turn = run(workspace, ("a.txt", b"a\nb\n"), seed={"a.txt": original})

    restore_turn(turn, workspace)

    assert (workspace / "a.txt").read_bytes() == original


def test_permissions_are_left_as_they_are(workspace: Path) -> None:
    target = workspace / "a.sh"
    target.write_bytes(b"old")
    target.chmod(0o640)
    turn = run(workspace, ("a.sh", b"new"))

    restore_turn(turn, workspace)

    assert target.read_bytes() == b"old"
    assert target.stat().st_mode & 0o7777 == 0o640


def test_no_temp_file_is_left_behind(workspace: Path) -> None:
    turn = run(workspace, ("a.py", b"new"), seed={"a.py": b"old"})

    restore_turn(turn, workspace)

    assert sorted(p.name for p in workspace.iterdir()) == ["a.py"]


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


def test_a_symlink_swapped_in_is_skipped_and_the_outside_file_is_untouched(tmp_path: Path, workspace: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    turn = run(workspace, ("a.py", b"new"), seed={"a.py": b"old"})
    (workspace / "a.py").unlink()
    (workspace / "a.py").symlink_to(outside)

    result = restore_turn(turn, workspace)

    assert outside.read_bytes() == b"secret"
    assert (workspace / "a.py").is_symlink()
    assert result.entries == [("skipped", "a.py", CHANGED)]


def test_a_symlinked_folder_swapped_in_is_skipped(tmp_path: Path, workspace: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "a.py").write_bytes(b"new")
    turn = run(workspace, ("d/a.py", b"new"), seed={"d/a.py": b"old"})
    (workspace / "d" / "a.py").unlink()
    (workspace / "d").rmdir()
    (workspace / "d").symlink_to(elsewhere)

    result = restore_turn(turn, workspace)

    assert (elsewhere / "a.py").read_bytes() == b"new"
    assert result.entries == [("skipped", "d/a.py", CHANGED)]


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


def test_the_notice_lists_restored_deleted_and_skipped_in_order() -> None:
    result = UndoResult(
        [("restored", "a.py", None), ("deleted", "n.txt", None), ("skipped", "b.py", CHANGED), ("skipped", "c", "boom")]
    )

    assert undo_notices(result, 0) == [
        Notice(
            "warn",
            "Undid the last turn: restored a.py · deleted n.txt · skipped b.py (changed since the turn)"
            " · skipped c (boom)",
        )
    ]


def test_the_notice_is_info_when_nothing_was_skipped() -> None:
    result = UndoResult([("restored", "a.py", None)])

    assert undo_notices(result, 0) == [Notice("info", "Undid the last turn: restored a.py")]


def test_the_notice_says_no_files_changed_for_an_empty_result() -> None:
    assert undo_notices(UndoResult([]), 0) == [Notice("info", "Undid the last turn: no files changed")]


def test_the_bash_and_terminal_line_is_singular_and_plural() -> None:
    one = undo_notices(UndoResult([]), 1)
    two = undo_notices(UndoResult([]), 2)

    assert one[1] == Notice("warn", "That turn ran 1 bash/terminal command; its effects were not undone.")
    assert two[1] == Notice("warn", "That turn ran 2 bash/terminal commands; their effects were not undone.")


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


def test_an_unexpected_error_in_one_file_skips_it_and_restores_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.agent import undo as undo_module

    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("old-a")
    b.write_text("old-b")
    log = undo_module.UndoLog(tmp_path)
    log.begin(0, "p")
    for path in (a, b):
        log.before(path)
        path.write_text("new")
        log.after(path)
    log.finish()
    turn = log.pop()
    assert turn is not None
    real = undo_module._restore_one

    def flaky(change, workspace):
        if change.path == a.resolve():
            raise RuntimeError("odd")
        return real(change, workspace)

    monkeypatch.setattr(undo_module, "_restore_one", flaky)
    result = undo_module.restore_turn(turn, tmp_path)

    assert [e[0] for e in result.entries] == ["skipped", "restored"]
    assert "odd" in (result.entries[0][2] or "")
    assert b.read_text() == "old-b"
