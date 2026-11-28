"""TurnLedger: which other sessions' turns overlap one in time, and the paths those changed."""

from __future__ import annotations

import gc
import hashlib
import os
import shutil
import weakref
from pathlib import Path

import pytest
from core.engine.checkpoints.git import GitError
from core.engine.checkpoints.store import CheckpointStore, DiffEntry
from core.engine.guardrails.setup import Guardrails
from core.engine.ledger import TurnLedger
from core.engine.undo import CHANGED_BY_OTHER, MAYBE_BY_OTHER, FileChange, UndoTurn

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def turn() -> UndoTurn:
    return UndoTurn(0, "p")


def digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def test_overlap_means_other_owner_and_intersecting_time_running_turns_included() -> None:
    # Bug caught: counting the same owner's turns, or turns that ended before this one began, as overlapping.
    ledger = TurnLedger(clock=Clock())
    early = ledger.begin("B", turn())
    ledger.finish(early)
    mine = ledger.begin("A", turn())
    same_owner = ledger.begin("A", turn())
    running = ledger.begin("C", turn())
    ledger.finish(same_owner)

    assert ledger.overlapping(mine) == [running]
    ledger.finish(mine)
    assert ledger.overlapping(mine) == [running]
    assert ledger.overlapping(early) == []


def test_a_dropped_turn_leaves_the_ledger() -> None:
    ledger = TurnLedger(clock=Clock())
    first = ledger.begin("A", turn())
    other = ledger.begin("B", turn())
    probe = ledger.begin("C", turn())
    ledger.drop(first)
    assert ledger.overlapping(probe) == [other]


def test_a_retired_owners_entries_still_overlap_and_a_running_one_is_never_removed() -> None:
    # Bug caught: `/new`, a switch or compaction removing an entry that another session's undo still needs, or one
    # whose turn is still running (so `finish` could never set its end).
    ledger = TurnLedger(clock=Clock())
    ended = ledger.begin("A", turn())
    ledger.finish(ended)
    running = ledger.begin("A", turn())
    probe = ledger.begin("B", turn())  # live, overlaps `running` (and not `ended`)

    ledger.retire("A")

    assert ledger.overlapping(probe) == [running]
    ledger.finish(running)  # the retired running entry still gets its end
    assert running.end is not None and running in ledger.entries
    assert ended not in ledger.entries  # retired, ended, and no live entry overlaps it


def test_a_retired_entry_goes_once_no_live_entry_of_another_owner_overlaps_it() -> None:
    ledger = TurnLedger(clock=Clock())
    retired = ledger.begin("A", turn())
    live = ledger.begin("B", turn())
    ledger.finish(retired)
    ledger.retire("A")
    assert retired in ledger.entries  # B's running turn overlaps it
    ledger.finish(live)
    ledger.drop(live)  # B's turn is undone: nothing live holds A's entry any more
    assert ledger.entries == []


def test_an_undone_entry_still_guards_the_live_turn_it_overlapped() -> None:
    # Bug caught: dropping the entry on undo, so the second session's undo of the shared path was no longer skipped.
    ledger = TurnLedger(clock=Clock())
    kept = [turn(), turn()]  # the undo logs hold their turns
    first = ledger.begin("A", kept[0])
    second = ledger.begin("B", kept[1])
    ledger.finish(first)
    ledger.finish(second)

    ledger.undone(first)

    assert ledger.overlapping(second) == [first]
    ledger.undone(second)
    assert ledger.entries == []


def test_the_cap_drops_the_oldest_ended_entries_and_never_a_running_one() -> None:
    ledger = TurnLedger(clock=Clock(), cap=3)
    running = ledger.begin("A", turn())
    for i in range(6):
        ledger.finish(ledger.begin(f"O{i}", turn()))

    assert len(ledger.entries) == 3 and running in ledger.entries


def test_a_finished_entry_holds_no_file_bytes_and_goes_when_its_undo_record_is_evicted() -> None:
    # Bug caught: the ledger keeping every turn's file bytes alive for as long as it guards other sessions.
    ledger = TurnLedger(clock=Clock())
    t = turn()
    t.changes[Path("/w/big.bin")] = FileChange(Path("/w/big.bin"), b"x" * 100_000, True, (), after=digest("y"))
    entry = ledger.begin("A", t)
    probe = ledger.begin("B", turn())
    ledger.finish(entry)
    assert ledger.find(t) is entry
    ref = weakref.ref(t)

    del t
    gc.collect()

    assert ref() is None  # the entry keeps no strong reference to the turn (and so to its bytes)
    assert entry.paths == frozenset({Path("/w/big.bin")}) and entry.digests == {Path("/w/big.bin"): digest("y")}
    assert ledger.overlapping(probe) == [entry]  # it still guards the others


async def test_a_turn_without_snapshots_contributes_its_file_log_paths(tmp_path: Path) -> None:
    # Bug caught: the file-level path of another session missing when it has no checkpoints.
    ledger = TurnLedger(clock=Clock())
    mine = ledger.begin("B", turn())
    theirs = ledger.begin("A", log_only(tmp_path, "s.py", "a.py"))

    found = await ledger.foreign_paths(mine, None, tmp_path)

    assert found == {"s.py": CHANGED_BY_OTHER, "a.py": CHANGED_BY_OTHER}
    ledger.finish(theirs)


def log_only(workspace: Path, *names: str, written: dict[str, str] | None = None) -> UndoTurn:
    t = turn()
    for name in names:
        path = workspace / name
        t.changes[path] = FileChange(path, None, False, (), after=digest(written[name]) if written and name in written else None)
    return t


def make_store(tmp_path: Path) -> tuple[CheckpointStore, Path]:
    workspace = (tmp_path / "ws").resolve()
    workspace.mkdir()
    return CheckpointStore(workspace, tmp_path / "data", Guardrails.off(workspace), dict(os.environ)), workspace


async def snap(store: CheckpointStore) -> str:
    snapshot = await store.snapshot()
    assert snapshot is not None, store.off_reason
    return snapshot.commit


@needs_git
async def test_paths_come_from_before_to_after_and_from_before_to_now_while_running(tmp_path: Path) -> None:
    # Bug caught: a running turn's changes missing (no "now" snapshot) or a finished one's diff taken to "now".
    store, workspace = make_store(tmp_path)
    ledger = TurnLedger(clock=Clock())
    mine = ledger.begin("B", turn())

    done_turn, running_turn = turn(), turn()
    done_turn.before = running_turn.before = await snap(store)
    (workspace / "done.py").write_text("1")
    done_turn.after = await snap(store)
    finished = ledger.begin("A", done_turn)
    ledger.finish(finished)
    ledger.begin("C", running_turn)
    (workspace / "live.py").write_text("2")  # after the last snapshot: only a fresh one sees it

    found = await ledger.foreign_paths(mine, store, workspace)

    assert set(found) == {"done.py", "live.py"}  # the running turn's diff to now holds both files
    ledger.finish(mine)


@needs_git
async def test_an_ended_turn_with_a_before_but_no_after_is_diffed_to_now(tmp_path: Path) -> None:
    # Bug caught: a turn whose undo log was cleared mid-turn (so no `after` snapshot) losing its bash changes.
    store, workspace = make_store(tmp_path)
    ledger = TurnLedger(clock=Clock())
    mine = ledger.begin("B", turn())
    cleared = turn()
    cleared.before = await snap(store)
    entry = ledger.begin("A", cleared)
    ledger.finish(entry)
    (workspace / "late.py").write_text("x")

    found = await ledger.foreign_paths(mine, store, workspace)

    assert set(found) == {"late.py"}


@needs_git
async def test_a_turn_whose_before_snapshot_is_not_taken_yet_adds_only_its_file_log(tmp_path: Path) -> None:
    # Bug caught: diffing from a missing `before` (an error that hides the other paths).
    store, workspace = make_store(tmp_path)
    ledger = TurnLedger(clock=Clock())
    mine = ledger.begin("B", turn())
    ledger.begin("A", log_only(workspace, "w.py"))

    found = await ledger.foreign_paths(mine, store, workspace)

    assert found == {"w.py": CHANGED_BY_OTHER} and not found.failures


@needs_git
async def test_paths_the_other_session_deleted_or_created_are_skipped_too(tmp_path: Path) -> None:
    # Bug caught: only modified paths counted: the D and A rows of the other turn's diff left out.
    store, workspace = make_store(tmp_path)
    (workspace / "gone.py").write_text("old")
    other = turn()
    other.before = await snap(store)
    (workspace / "gone.py").unlink()
    (workspace / "new.py").write_text("n")
    other.after = await snap(store)
    ledger = TurnLedger(clock=Clock())
    mine = ledger.begin("B", turn())
    ledger.finish(ledger.begin("A", other))

    found = await ledger.foreign_paths(mine, store, workspace)

    assert found == {"gone.py": CHANGED_BY_OTHER, "new.py": CHANGED_BY_OTHER}


@needs_git
async def test_a_path_the_undone_turn_wrote_is_exempt_only_while_the_other_turn_left_it_as_written(tmp_path: Path) -> None:
    # Bug caught: dropping from the other turn's diff every path the undone turn logged, so the other session's
    # own (bash) edit of a file the undone turn wrote was reverted.
    store, workspace = make_store(tmp_path)
    ledger = TurnLedger(clock=Clock())
    mine_turn = log_only(workspace, "same.py", "edited.py", written={"same.py": "T", "edited.py": "T"})
    mine = ledger.begin("T", mine_turn)
    other = turn()
    other.before = await snap(store)
    (workspace / "same.py").write_text("T")  # T's write, seen in U's diff as it was left
    (workspace / "edited.py").write_text("U")  # U's bash edit on top of T's write
    other.after = await snap(store)
    ledger.finish(ledger.begin("U", other))

    found = await ledger.foreign_paths(mine, store, workspace)

    assert found == {"edited.py": CHANGED_BY_OTHER}


@needs_git
async def test_a_path_in_the_other_diff_that_matches_the_undone_turns_own_end_state_is_skipped_as_maybe_theirs(
    tmp_path: Path,
) -> None:
    # Bug caught: telling the user another session changed a file that may be the undone turn's own bash change.
    store, workspace = make_store(tmp_path)
    ledger = TurnLedger(clock=Clock())
    mine_turn = turn()
    mine_turn.before = await snap(store)
    mine = ledger.begin("T", mine_turn)
    other = turn()
    entry = ledger.begin("U", other)
    other.before = mine_turn.before
    (workspace / "own.py").write_text("T-bash")
    mine_turn.after = await snap(store)
    (workspace / "theirs.py").write_text("U")
    other.after = await snap(store)
    ledger.finish(mine)
    ledger.finish(entry)

    found = await ledger.foreign_paths(mine, store, workspace)

    assert found == {"own.py": MAYBE_BY_OTHER, "theirs.py": CHANGED_BY_OTHER}


@needs_git
async def test_a_failing_diff_is_reported_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Bug caught: `except Exception: pass`: the undo going on as if no other session had changed anything.
    store, workspace = make_store(tmp_path)

    async def broken(before: str, after: str) -> list[DiffEntry]:
        raise GitError("boom")

    monkeypatch.setattr(store, "diff", broken)
    other = turn()
    other.before = await snap(store)
    other.after = await snap(store)
    ledger = TurnLedger(clock=Clock())
    mine = ledger.begin("B", turn())
    ledger.finish(ledger.begin("A", other))

    found = await ledger.foreign_paths(mine, store, workspace)

    assert len(found.failures) == 1 and "boom" in found.failures[0]
