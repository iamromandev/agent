"""Undo for the agent's file changes: a log of what each turn's write/edit calls changed, and the restore.

UI-free. The tools call `before`/`after` from the worker thread that writes, so a snapshot is exactly
what was on disk at write time. Snapshots live in memory for the current run only.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.agent.events import Notice

SNAPSHOT_FILE_LIMIT = 8 * 1024 * 1024
SNAPSHOT_TOTAL_LIMIT = 64 * 1024 * 1024
NOTHING_TO_UNDO = "Nothing to undo"
CHANGED = "changed since the turn"
TOO_LARGE = "too large"
READ_ONLY = "read-only"


@dataclass
class FileChange:
    path: Path  # resolved absolute path the tool wrote (as given to before())
    before: bytes | None  # None: did not exist (or too_large)
    existed: bool
    created_dirs: tuple[Path, ...]  # folders the write creates, outermost first; only when not existed
    after: bytes | None = None  # sha256 digest of what the turn left; None: file absent after
    too_large: bool = False
    broken: str | None = None  # the file could not be read: reason; undo skips it with this


@dataclass
class UndoTurn:
    start: int  # len(history) when the turn began = index of its user message
    prompt: str  # what goes back in the input box
    changes: dict[Path, FileChange] = field(default_factory=dict)  # insertion order = first touch
    bash: int = 0
    size: int = 0  # bytes held in `before` snapshots


def _digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _read_nofollow(path: Path) -> bytes:
    """The bytes of a regular file, never through a symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _hash_nofollow(path: Path) -> bytes:
    """The sha256 digest of a regular file read in chunks, never through a symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("not a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1 << 20):
            digest.update(chunk)
        return digest.digest()
    finally:
        os.close(fd)


def _rmdirs(folders: tuple[Path, ...], workspace: Path) -> None:
    """Remove the empty ones of `folders` (outermost first), innermost first; never through a symlink or outside."""
    for folder in reversed(folders):
        if not _inside(folder, workspace) or folder == workspace:
            break
        try:
            os.rmdir(folder)
        except OSError:
            break  # not empty, or gone: the outer ones cannot be empty either


def _reason(exc: OSError) -> str:
    return exc.strerror or str(exc)


class UndoLog:
    """Undo records of the turns of this run, newest last; also the journal the file tools call."""

    def __init__(
        self,
        workspace: Path,
        file_limit: int = SNAPSHOT_FILE_LIMIT,
        total_limit: int = SNAPSHOT_TOTAL_LIMIT,
    ) -> None:
        self._workspace = workspace.resolve()
        self._file_limit = file_limit
        self._total_limit = total_limit
        self._stack: list[UndoTurn] = []
        self._current: UndoTurn | None = None
        self._fresh: Path | None = None  # added by the before() whose after() has not run yet

    @property
    def size(self) -> int:
        return sum(t.size for t in self._stack) + (self._current.size if self._current else 0)

    def begin(self, start: int, prompt: str) -> None:
        self._current = UndoTurn(start, prompt)
        self._fresh = None

    def finish(self) -> None:
        if self._current is not None:
            self._stack.append(self._current)
        self._current = None
        self._fresh = None

    def clear(self) -> None:
        self._stack.clear()
        self._current = None
        self._fresh = None

    def pop(self) -> UndoTurn | None:
        return self._stack.pop() if self._stack else None

    def note_bash(self) -> None:
        if self._current is not None:
            self._current.bash += 1

    def before(self, path: Path) -> None:
        turn = self._current
        if turn is None:
            return
        path = self._key(turn, path)
        if path in turn.changes:
            return
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            turn.changes[path] = FileChange(path, None, False, self._missing_dirs(path))
            self._fresh = path
            return
        except OSError as exc:
            turn.changes[path] = FileChange(path, None, True, (), broken=_reason(exc))
            self._fresh = path
            return
        change = FileChange(path, None, True, ())
        turn.changes[path] = change
        self._fresh = path
        if not stat.S_ISREG(info.st_mode):
            change.broken = "not a regular file"
            return
        if info.st_size > self._file_limit:
            change.too_large = True
            return
        if turn.size + info.st_size > self._total_limit:
            change.too_large = True  # cannot fit even alone: do not evict older turns for it
            return
        self._make_room(info.st_size)
        if self.size + info.st_size > self._total_limit:
            change.too_large = True
            return
        try:
            change.before = _read_nofollow(path)
        except OSError as exc:
            change.broken = _reason(exc)
            return
        turn.size += len(change.before)

    def after(self, path: Path) -> None:
        turn = self._current
        fresh, self._fresh = self._fresh, None
        if turn is None:
            return
        path = self._key(turn, path)
        if path not in turn.changes:
            return
        change = turn.changes[path]
        try:
            digest: bytes | None = _hash_nofollow(path)
        except FileNotFoundError:
            digest = None
        except OSError as exc:
            change.broken = _reason(exc)
            return
        unchanged = digest == _digest(change.before or b"") if change.existed else digest is None
        if fresh == path and unchanged and not change.too_large and change.broken is None:
            del turn.changes[path]  # the call changed nothing: no trace
            turn.size -= len(change.before or b"")
            if not change.existed:
                _rmdirs(change.created_dirs, self._workspace)  # a failed write may have made the folders
            return
        change.after = digest

    def _key(self, turn: UndoTurn, path: Path) -> Path:
        """The path an entry for `path` is kept under: an earlier spelling of the same file, when there is one."""
        if path in turn.changes:
            return path
        folded = str(path).casefold()
        for known in turn.changes:
            if str(known).casefold() == folded and self._same_file(known, path):
                return known
        return path

    @staticmethod
    def _same_file(a: Path, b: Path) -> bool:
        try:
            return os.path.samefile(a, b)
        except OSError:
            return False

    def _make_room(self, incoming: int) -> None:
        while self._stack and self.size + incoming > self._total_limit:
            self._stack.pop(0)

    def _missing_dirs(self, path: Path) -> tuple[Path, ...]:
        """Folders above `path` that do not exist yet and lie inside the workspace, outermost first."""
        missing: list[Path] = []
        folder = path.parent
        while not os.path.lexists(folder):
            missing.append(folder)
            folder = folder.parent
        inside = [f for f in missing if f.is_relative_to(self._workspace) and f != self._workspace]
        return tuple(reversed(inside))


@dataclass(frozen=True)
class UndoResult:
    # (kind, workspace-relative posix path, reason for a skip), in first-touch order
    entries: list[tuple[Literal["restored", "deleted", "skipped"], str, str | None]]


def _inside(path: Path, workspace: Path) -> bool:
    return os.path.realpath(path) == str(path) and path.is_relative_to(workspace)


def _restore_one(change: FileChange, workspace: Path) -> tuple[Literal["restored", "deleted"], None] | str | None:
    """Undo one file. Returns what was done, a skip reason, or None when there was nothing to do."""
    if change.too_large:
        return TOO_LARGE
    if change.broken is not None:
        return change.broken
    path = change.path
    if not _inside(path, workspace):
        return CHANGED
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        current: bytes | None = None
        info = None
        mode = 0o644
    else:
        if not stat.S_ISREG(info.st_mode):
            return CHANGED
        current = _hash_nofollow(path)
        mode = info.st_mode & 0o7777
    if current != change.after:
        return CHANGED
    if not change.existed:
        if current is None:
            return None
        os.unlink(path)
        return ("deleted", None)
    assert change.before is not None
    if info is not None and (not mode & stat.S_IWUSR or not os.access(path, os.W_OK)):
        return READ_ONLY
    if info is not None and info.st_nlink > 1:  # write in place: a replacement would break the other links
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            _write_all(fd, change.before)
        finally:
            os.close(fd)
        return ("restored", None)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".agent-undo-")
    try:
        try:
            _write_all(fd, change.before)
            os.fchmod(fd, mode)
            if info is not None:
                with contextlib.suppress(OSError):
                    os.fchown(fd, info.st_uid, info.st_gid)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return ("restored", None)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def restore_turn(turn: UndoTurn, workspace: Path) -> UndoResult:
    """Put back every file the turn changed that still holds what the turn left. Blocking; never raises."""
    root = workspace.resolve()
    entries: list[tuple[Literal["restored", "deleted", "skipped"], str, str | None]] = []
    for change in turn.changes.values():
        try:
            name = change.path.relative_to(root).as_posix()
        except ValueError:
            name = str(change.path)
        try:
            outcome = _restore_one(change, root)
        except Exception as exc:  # whatever one file does, the others are still restored
            reason = _reason(exc) if isinstance(exc, OSError) else str(exc) or type(exc).__name__
            entries.append(("skipped", name, reason))
            continue
        if isinstance(outcome, str):
            entries.append(("skipped", name, outcome))
        elif outcome is not None:
            entries.append((outcome[0], name, None))
    for change in turn.changes.values():
        with contextlib.suppress(Exception):
            _rmdirs(change.created_dirs, root)
    return UndoResult(entries)


def undo_notices(result: UndoResult, bash: int) -> list[Notice]:
    if result.entries:
        parts = [
            f"skipped {name} ({reason})" if kind == "skipped" else f"{kind} {name}"
            for kind, name, reason in result.entries
        ]
        level: Literal["info", "warn"] = "warn" if any(e[0] == "skipped" for e in result.entries) else "info"
        notices = [Notice(level, "Undid the last turn: " + " · ".join(parts))]
    else:
        notices = [Notice("info", "Undid the last turn: no files changed")]
    if bash > 0:
        plural = "" if bash == 1 else "s"
        notices.append(Notice("warn", f"That turn ran {bash} bash command{plural}; its effects were not undone."))
    return notices
