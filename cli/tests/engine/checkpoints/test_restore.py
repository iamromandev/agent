"""restore_checkpoint: put back what a turn changed, using two real snapshots, never through a symlink."""

from __future__ import annotations

import os
import shutil
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from src.engine.checkpoints import restore as restore_module
from src.engine.checkpoints.restore import git_blob_sha, restore_checkpoint, restore_unsnapshotted
from src.engine.checkpoints.store import CheckpointStore, DiffEntry
from src.engine.guardrails.setup import Guardrails
from src.engine.undo import CHANGED, UndoLog, UndoResult, UndoTurn

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


@pytest.fixture
def store(ws: Path, tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(ws, tmp_path / "data", Guardrails.off(ws), dict(os.environ))


async def undo_turn(
    store: CheckpointStore,
    ws: Path,
    turn: Callable[[], object],
    user: Callable[[], object] | None = None,
    **options: Any,
) -> UndoResult:
    """Snapshot, run `turn`, snapshot, run `user` (an edit after the turn), then restore. `options` override what the
    agent would pass (`before_time`, `large_before`, `exclude_changed`)."""
    before = await store.snapshot()
    before_time = time.time()
    time.sleep(0.02)  # a real turn starts after a model call: what it creates is born after the snapshot
    turn()
    after = await store.snapshot()
    assert before is not None and after is not None
    if user is not None:
        user()
    entries = await store.diff(before.commit, after.commit)
    blobs = await store.read_blobs([e.old_sha for e in entries if e.old_sha])
    arguments: dict[str, Any] = {
        "before_time": before_time,
        "large_before": frozenset(before.skipped_large),
        **options,
    }
    return restore_checkpoint(entries, blobs, ws, **arguments)


def kinds(result: UndoResult) -> dict[str, tuple[str, str | None]]:
    return {name: (kind, reason) for kind, name, reason in result.entries}


async def test_an_added_file_the_user_changed_is_kept(store: CheckpointStore, ws: Path) -> None:
    result = await undo_turn(store, ws, lambda: (ws / "n.py").write_text("n"), lambda: (ws / "n.py").write_text("mine"))

    assert (ws / "n.py").read_text() == "mine"
    assert kinds(result) == {"n.py": ("skipped", CHANGED)}


async def test_a_deleted_symlink_is_restored_as_a_symlink(store: CheckpointStore, ws: Path) -> None:
    (ws / "a.py").write_text("a")
    (ws / "link").symlink_to("a.py")

    await undo_turn(store, ws, lambda: (ws / "link").unlink())

    assert (ws / "link").is_symlink() and os.readlink(ws / "link") == "a.py"


async def test_a_symlink_the_turn_made_is_unlinked_not_followed(store: CheckpointStore, ws: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    (ws / "keep.txt").write_text("k")

    await undo_turn(store, ws, lambda: (ws / "link").symlink_to(outside))

    assert not os.path.lexists(ws / "link")
    assert outside.read_text() == "precious"


async def test_a_file_replaced_by_a_folder_the_user_filled_is_skipped(store: CheckpointStore, ws: Path) -> None:
    (ws / "x").write_text("was a file")

    def turn() -> None:
        (ws / "x").unlink()
        (ws / "x").mkdir()
        (ws / "x" / "f.txt").write_text("f")

    result = await undo_turn(store, ws, turn, lambda: (ws / "x" / "mine.txt").write_text("m"))

    assert (ws / "x" / "mine.txt").read_text() == "m"
    assert kinds(result)["x"] == ("skipped", CHANGED)


async def test_a_folder_swapped_for_a_symlink_the_user_made_is_skipped(
    store: CheckpointStore, ws: Path, tmp_path: Path
) -> None:
    (ws / "sub").mkdir()
    (ws / "sub" / "f.txt").write_text("f")
    outside = tmp_path / "outside"
    outside.mkdir()

    def turn() -> None:
        (ws / "sub" / "f.txt").write_text("f2")

    def user() -> None:
        shutil.rmtree(ws / "sub")
        (ws / "sub").symlink_to(outside)

    result = await undo_turn(store, ws, turn, user)

    assert list(outside.iterdir()) == []
    assert kinds(result)["sub/f.txt"] == ("skipped", CHANGED)


async def test_a_symlinked_parent_is_never_followed_when_deleting(store: CheckpointStore, ws: Path, tmp_path: Path) -> None:
    (ws / "keep.txt").write_text("k")
    outside = tmp_path / "outside"
    outside.mkdir()

    def turn() -> None:
        (ws / "sub").mkdir()
        (ws / "sub" / "n.txt").write_text("n")

    def user() -> None:
        shutil.rmtree(ws / "sub")
        (ws / "sub").symlink_to(outside)
        (outside / "n.txt").write_text("n")  # the same bytes: it would pass a content check

    result = await undo_turn(store, ws, turn, user)

    assert (outside / "n.txt").read_text() == "n"
    assert kinds(result)["sub/n.txt"] == ("skipped", CHANGED)


async def test_one_file_that_cannot_be_written_does_not_stop_the_others(store: CheckpointStore, ws: Path) -> None:
    (ws / "locked").mkdir()
    (ws / "locked" / "f.txt").write_text("f1")
    (ws / "ok.txt").write_text("o1")

    def turn() -> None:
        (ws / "locked" / "f.txt").write_text("f2")
        (ws / "ok.txt").write_text("o2")
        (ws / "locked").chmod(0o555)

    try:
        result = await undo_turn(store, ws, turn)
    finally:
        (ws / "locked").chmod(0o755)

    if os.geteuid() == 0:
        pytest.skip("root ignores folder permissions")
    assert (ws / "ok.txt").read_text() == "o1"
    assert (ws / "locked" / "f.txt").read_text() == "f2"
    assert kinds(result)["locked/f.txt"][0] == "skipped"
    assert kinds(result)["ok.txt"] == ("restored", None)


def test_git_blob_sha_matches_git() -> None:
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_an_entry_with_a_bad_path_is_skipped_not_raised(ws: Path) -> None:
    entry = DiffEntry("../escape.txt", 0, None, 0o100644, "a" * 40)

    result = restore_checkpoint([entry], {}, ws)

    assert kinds(result) == {"../escape.txt": ("skipped", CHANGED)}


def test_a_missing_blob_is_a_skip(ws: Path) -> None:
    (ws / "a.py").write_text("x")
    entry = DiffEntry("a.py", 0o100644, "b" * 40, 0o100644, git_blob_sha(b"x"))

    result = restore_checkpoint([entry], {}, ws)

    assert kinds(result)["a.py"][0] == "skipped"
    assert stat.S_ISREG(os.lstat(ws / "a.py").st_mode)


# --- an added file is not always one the turn created ------------------------------------------------------


async def test_a_changed_exclude_file_skips_every_added_file(store: CheckpointStore, ws: Path) -> None:
    result = await undo_turn(store, ws, lambda: (ws / "n.txt").write_text("n"), exclude_changed=True)

    assert (ws / "n.txt").read_text() == "n"
    assert kinds(result)["n.txt"][0] == "skipped"


@pytest.mark.parametrize("before_time", [None], ids=["by birth time"])
async def test_a_large_file_the_turn_truncated_is_not_deleted(
    store: CheckpointStore, ws: Path, before_time: float | None
) -> None:
    with open(ws / "big.bin", "wb") as f:
        f.truncate(20 * 1024 * 1024)
    options: dict[str, Any] = {} if before_time is None else {"before_time": before_time}

    def turn() -> None:
        with open(ws / "big.bin", "r+b") as f:
            f.truncate(0)

    result = await undo_turn(store, ws, turn, **options)

    assert (ws / "big.bin").exists()
    assert kinds(result)["big.bin"][0] == "skipped"


async def test_without_the_time_of_the_snapshot_nothing_is_deleted(store: CheckpointStore, ws: Path) -> None:
    result = await undo_turn(store, ws, lambda: (ws / "n.txt").write_text("n"), before_time=None)

    assert (ws / "n.txt").exists()
    assert kinds(result)["n.txt"][0] == "skipped"


# --- hard links, permissions -------------------------------------------------------------------------------


@pytest.fixture
def umask_077() -> object:
    old = os.umask(0o077)
    yield None
    os.umask(old)


async def test_a_recreated_folder_gets_the_umask(store: CheckpointStore, ws: Path, umask_077: object) -> None:
    (ws / "d").mkdir()
    (ws / "d" / "f.txt").write_text("f")

    await undo_turn(store, ws, lambda: shutil.rmtree(ws / "d"))

    assert stat.S_IMODE(os.stat(ws / "d").st_mode) == 0o700


# --- the folder is checked and used through one descriptor -------------------------------------------------


@pytest.fixture
def swap_parent(monkeypatch: pytest.MonkeyPatch, ws: Path, tmp_path: Path) -> Callable[[str, str], Path]:
    """Right after the restore has checked a file inside `sub` (`_state`), or just before it writes one (`_write`),
    replace `sub` by a link to an outside folder that holds the same file: what a path-based restore would touch."""
    outside = tmp_path / "outside"

    def arm(name: str, hook: str) -> Path:
        outside.mkdir()
        (outside / name).write_text("same")
        real = getattr(restore_module, hook)
        done: list[bool] = []

        def swap() -> None:
            if not done:
                done.append(True)
                os.rename(ws / "sub", ws / "sub-moved")
                os.symlink(outside, ws / "sub")

        def spying(*args: object, **kwargs: object) -> object:
            if hook == "_write":
                swap()
            result = real(*args, **kwargs)
            if hook == "_state":
                swap()
            return result

        monkeypatch.setattr(restore_module, hook, spying)
        return outside

    return arm


# --- folders and what the user made ------------------------------------------------------------------------


@pytest.fixture
def no_birth_times(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file system without birth times (tmpfs on older kernels, NFS, FUSE): only `ctime` is left."""
    monkeypatch.setattr(restore_module, "birth_time", lambda *_: None)


@pytest.mark.usefixtures("no_birth_times")
async def test_without_birth_times_a_folder_written_into_is_kept(store: CheckpointStore, ws: Path) -> None:
    """Deleting the turn's file moves the folder's ctime, so ctime cannot tell an old folder from a new one: skip."""
    (ws / "empty").mkdir()

    await undo_turn(store, ws, lambda: (ws / "empty" / "f.txt").write_text("f"))

    assert not (ws / "empty" / "f.txt").exists()
    assert (ws / "empty").is_dir()


# --- files the checkpoints leave out, changed through write/edit ------------------------------------------


def write_turn(ws: Path, *changes: tuple[str, bytes]) -> UndoTurn:
    """An undo record of write/edit calls: what the tools do (before, change, after)."""
    log = UndoLog(ws)
    log.begin(0, "go")
    for name, data in changes:
        path = ws / name
        log.before(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        log.after(path)
    log.finish()
    turn = log.pop()
    assert turn is not None
    return turn


def test_ignored_and_agent_files_the_turn_wrote_are_restored_from_the_file_log(ws: Path) -> None:
    (ws / "build").mkdir()
    (ws / "build" / "out.txt").write_text("old")
    (ws / ".agent" / "commands").mkdir(parents=True)
    (ws / ".agent" / "commands" / "c.md").write_text("old command")
    turn = write_turn(ws, ("build/out.txt", b"new"), (".agent/commands/c.md", b"new"), ("build/made.txt", b"m"))

    result = restore_unsnapshotted(turn, [], ws, lambda path: False)

    assert (ws / "build" / "out.txt").read_text() == "old"
    assert (ws / ".agent" / "commands" / "c.md").read_text() == "old command"
    assert not (ws / "build" / "made.txt").exists()
    assert kinds(result) == {
        "build/out.txt": ("restored", None),
        ".agent/commands/c.md": ("restored", None),
        "build/made.txt": ("deleted", None),
    }


def test_a_file_the_checkpoint_diff_already_covers_is_left_to_it(ws: Path) -> None:
    (ws / "a.py").write_text("old")
    turn = write_turn(ws, ("a.py", b"new"))
    entry = DiffEntry("a.py", 0o100644, git_blob_sha(b"old"), 0o100644, git_blob_sha(b"new"))

    result = restore_unsnapshotted(turn, [entry], ws, lambda path: False)

    assert result.entries == []
    assert (ws / "a.py").read_text() == "new"


def test_a_protected_file_is_never_written_by_the_file_log(ws: Path) -> None:
    (ws / "secret.key").write_text("old")
    turn = write_turn(ws, ("secret.key", b"new"))

    result = restore_unsnapshotted(turn, [], ws, lambda path: path.name == "secret.key")

    assert result.entries == []
    assert (ws / "secret.key").read_text() == "new"
