"""Put a turn's workspace changes back from two snapshots (`diff-tree before after`), in Python.

Each path is checked against what the turn left (the `after` side) before it is touched, so a file the user changed
since is kept. Every operation goes through a folder descriptor reached by opening each component with `O_NOFOLLOW`
from the workspace root: a folder swapped for a symlink after the check cannot redirect a delete or a write, and
nothing is written outside the workspace or through a symlink. A file is replaced (temporary file, then rename), never
written in place, so a hard link to an outside file is never written through. One failing path never stops the rest.

A path that is only in the `after` snapshot is not always one the turn created (it may have been ignored or too large
before), so it is deleted only when its birth time is after the `before` snapshot and no ignore rule changed."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import secrets
import stat
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Literal

from core.engine.checkpoints.birth import birth_time
from core.engine.checkpoints.store import DiffEntry
from core.engine.undo import CHANGED, TOO_LARGE, FileChange, UndoResult, UndoTurn, _hash_nofollow, restore_turn

LINK_MODE = 0o120000
EXEC_BIT = 0o100
TMP_PREFIX = ".agent-undo-"
EXISTED_BEFORE = "existed before the turn"
IGNORE_CHANGED = "ignore rules changed in the turn"
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

Kind = Literal["restored", "deleted", "skipped"]
State = tuple[str, ...] | None  # ("file", exec, sha) | ("link", sha) | ("dir",) | None (absent)


class _Changed(Exception):
    """A folder above the path is a symlink or not a folder: the workspace is not what the turn left."""


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _file_sha(dirfd: int, name: str) -> tuple[str, int]:
    """`(git blob sha, mode)` of a regular file in the folder `dirfd`, read in chunks and never through a symlink."""
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dirfd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("not a regular file")
        digest = hashlib.sha1(b"blob %d\0" % info.st_size, usedforsecurity=False)
        while chunk := os.read(fd, 1 << 20):
            digest.update(chunk)
        return digest.hexdigest(), info.st_mode
    finally:
        os.close(fd)


def _state(dirfd: int, name: str) -> State:
    try:
        info = os.lstat(name, dir_fd=dirfd)
    except (FileNotFoundError, NotADirectoryError):
        return None
    if stat.S_ISLNK(info.st_mode):
        return ("link", git_blob_sha(os.fsencode(os.readlink(name, dir_fd=dirfd))))
    if stat.S_ISDIR(info.st_mode):
        return ("dir",)
    if stat.S_ISREG(info.st_mode):
        sha, mode = _file_sha(dirfd, name)
        return ("file", str(mode & EXEC_BIT), sha)
    return ("other",)


def _expected(entry: DiffEntry) -> State:
    """What the turn left at the path."""
    if entry.new_sha is None:
        return None
    if entry.new_mode == LINK_MODE:
        return ("link", entry.new_sha)
    return ("file", str(entry.new_mode & EXEC_BIT), entry.new_sha)


def _open_parent(root_fd: int, rel: str, *, create: bool) -> int | None:
    """A descriptor for the folder that holds `rel`, reached one component at a time with `O_NOFOLLOW`. The caller
    closes it. None when a folder above `rel` is missing and `create` is off; with `create` the missing ones are
    made (mode `0o777 & ~umask`). `_Changed` when a component is a symlink or not a folder."""
    current = os.dup(root_fd)
    try:
        for part in rel.split("/")[:-1]:
            try:
                nxt = os.open(part, DIR_FLAGS, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    os.close(current)
                    return None
                with contextlib.suppress(FileExistsError):
                    os.mkdir(part, dir_fd=current)
                nxt = _open_folder(current, part)
            except OSError as exc:
                raise _folder_error(exc) from None
            os.close(current)
            current = nxt
        return current
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(current)
        raise


def _open_folder(dirfd: int, name: str) -> int:
    try:
        return os.open(name, DIR_FLAGS, dir_fd=dirfd)
    except OSError as exc:
        raise _folder_error(exc) from None


def _folder_error(exc: OSError) -> Exception:
    """`ELOOP` (a symlink, with `O_NOFOLLOW`) and `ENOTDIR` mean the folder is not what the turn left; anything else
    (permissions, ...) is reported as it is."""
    return _Changed() if exc.errno in (errno.ELOOP, errno.ENOTDIR) else exc


def _safe_name(rel: str) -> bool:
    parts = rel.split("/")
    return bool(rel) and not rel.startswith("/") and all(p not in ("", ".", "..") for p in parts)


def _is_new(parent: int, name: str, info: os.stat_result, before_time: float | None) -> bool:
    """Was `name` in `parent` (whose `lstat` is `info`) made after the `before` snapshot? Its birth time where the
    system keeps one (macOS, Linux `statx`), else, for a file, its ctime, which also moves when the file is written or
    renamed (so a file changed since counts as new). A folder without a birth time never counts as new: its ctime
    moves whenever an entry is added or removed, so one the turn wrote into, or emptied, would always look new."""
    if before_time is None:
        return False
    born = birth_time(parent, name, info)
    if born is None:
        return not stat.S_ISDIR(info.st_mode) and info.st_ctime > before_time
    return born > before_time


def _with_exec(mode: int, executable: bool) -> int:
    """`mode` with the exec bits set where the file can be read (as git does) or cleared."""
    return mode | ((mode & 0o444) >> 2) if executable else mode & ~0o111


def _write(dirfd: int, name: str, entry: DiffEntry, data: bytes) -> None:
    """Replace `name` in the folder `dirfd` by the `before` side of `entry`: a temporary file, then a rename, so the
    other names of a hard-linked file keep what they have. A file that exists keeps its permission bits (and owner
    where allowed) and only gets its exec bits set or cleared; a file that is made again gets `0o666 & ~umask`."""
    tmp = f"{TMP_PREFIX}{secrets.token_hex(6)}"
    if entry.old_mode == LINK_MODE:
        os.symlink(os.fsdecode(data), tmp, dir_fd=dirfd)
        _replace(dirfd, tmp, name)
        return
    try:
        existing = os.lstat(name, dir_fd=dirfd)
    except FileNotFoundError:
        existing = None
    kept = existing if existing is not None and stat.S_ISREG(existing.st_mode) else None
    # 0o600 while the old permissions are not on yet: the content is never readable by more than the final mode
    mode = 0o600 if kept is not None else 0o666
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=dirfd)
    try:
        try:
            if kept is not None:
                with contextlib.suppress(OSError):
                    os.fchown(fd, kept.st_uid, kept.st_gid)
                base = stat.S_IMODE(kept.st_mode)
            else:
                base = stat.S_IMODE(os.fstat(fd).st_mode)
            os.fchmod(fd, _with_exec(base, bool(entry.old_mode & EXEC_BIT)))
            _write_all(fd, data)
        finally:
            os.close(fd)
        _replace(dirfd, tmp, name)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp, dir_fd=dirfd)
        raise


def _replace(dirfd: int, tmp: str, name: str) -> None:
    try:
        os.replace(tmp, name, src_dir_fd=dirfd, dst_dir_fd=dirfd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp, dir_fd=dirfd)
        raise


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _reason(exc: Exception) -> str:
    return (exc.strerror if isinstance(exc, OSError) else None) or str(exc) or type(exc).__name__


def restore_checkpoint(
    entries: list[DiffEntry],
    blobs: dict[str, bytes],
    workspace: Path,
    *,
    before_time: float | None = None,
    large_before: frozenset[str] = frozenset(),
    exclude_changed: bool = False,
    skip: Mapping[str, str] | None = None,
) -> UndoResult:
    """Undo `entries` (what changed between the turn's two snapshots) in `workspace`. Blocking; never raises.

    `blobs`: the `before` content by sha; one that is missing was too large to hold and its path is skipped.
    `before_time`: when the `before` snapshot was taken (`time.time()`); a file only in the `after` snapshot is
    deleted only when it was born later (None: never). `large_before`: files over the size cap at that snapshot.
    `exclude_changed`: the turn changed the workspace's `.git/info/exclude`. A changed `.gitignore` (any folder) or
    exclude file means the diff cannot tell a file the turn created from one it stopped hiding: no added file is
    deleted then. `skip`: paths (and why) that another session's overlapping turn changed: never touched, named as
    skipped (a changed `.gitignore` among them still counts for the rule above)."""
    root = workspace.resolve()
    try:
        root_fd = os.open(root, DIR_FLAGS)
    except OSError as exc:
        return UndoResult([("skipped", e.path, _reason(exc)) for e in entries])
    try:
        return _restore(entries, blobs, root_fd, before_time, large_before, exclude_changed, skip or {})
    finally:
        os.close(root_fd)


def _restore(
    entries: list[DiffEntry],
    blobs: dict[str, bytes],
    root_fd: int,
    before_time: float | None,
    large_before: frozenset[str],
    exclude_changed: bool,
    skip: Mapping[str, str],
) -> UndoResult:
    done: list[tuple[Kind, str, str | None]] = []
    ignore_changed = exclude_changed or any(e.path.rpartition("/")[2].lower() == ".gitignore" for e in entries)
    folded = {name.casefold(): reason for name, reason in skip.items()}  # the filesystem may be case-insensitive
    held = [e for e in entries if e.path.casefold() in folded]
    done.extend(("skipped", e.path, folded[e.path.casefold()]) for e in held)
    entries = [e for e in entries if e.path.casefold() not in folded]
    added = sorted((e for e in entries if e.old_sha is None), key=lambda e: e.path.count("/"), reverse=True)
    cleaned: set[str] = set()  # folders above a file that was deleted (checked, real folders)
    for entry in added:
        try:
            reason = _remove(root_fd, entry, before_time, large_before, ignore_changed)
        except _Changed:
            done.append(("skipped", entry.path, CHANGED))
            continue
        except Exception as exc:
            done.append(("skipped", entry.path, _reason(exc)))
            continue
        if reason is not None:
            done.append(("skipped", entry.path, reason))
            continue
        done.append(("deleted", entry.path, None))
        cleaned.add(entry.path.rpartition("/")[0])
    folders: set[str] = set()
    for folder in cleaned:
        while folder:
            folders.add(folder)
            folder = folder.rpartition("/")[0]
    for folder in sorted(folders, key=lambda f: f.count("/"), reverse=True):
        with contextlib.suppress(Exception):
            _remove_folder(root_fd, folder, before_time)
    for entry in entries:
        if entry.old_sha is None:
            continue
        try:
            outcome = _put_back(entry, blobs, root_fd)
        except _Changed:
            outcome = CHANGED
        except Exception as exc:
            done.append(("skipped", entry.path, _reason(exc)))
            continue
        done.append(("restored", entry.path, None) if outcome is None else ("skipped", entry.path, outcome))
    return UndoResult(done)


def _remove(
    root_fd: int, entry: DiffEntry, before_time: float | None, large_before: frozenset[str], ignore_changed: bool
) -> str | None:
    """Delete a file only the `after` snapshot has, when the turn made it; a skip reason, or None when deleted."""
    if not _safe_name(entry.path):
        return CHANGED
    if ignore_changed:
        return IGNORE_CHANGED
    if entry.path in large_before:
        return TOO_LARGE  # it was over the size cap before the turn: the turn shrank it, it did not create it
    parent = _open_parent(root_fd, entry.path, create=False)
    if parent is None:
        return CHANGED
    try:
        name = entry.path.rpartition("/")[2]
        if _state(parent, name) != _expected(entry):
            return CHANGED
        if not _is_new(parent, name, os.lstat(name, dir_fd=parent), before_time):
            return EXISTED_BEFORE
        os.unlink(name, dir_fd=parent)
    finally:
        os.close(parent)
    return None


def _remove_folder(root_fd: int, rel: str, before_time: float | None) -> None:
    """Remove the folder `rel` when it is empty and was made after the `before` snapshot."""
    parent = _open_parent(root_fd, rel, create=False)
    if parent is None:
        return
    try:
        name = rel.rpartition("/")[2]
        info = os.lstat(name, dir_fd=parent)
        if stat.S_ISDIR(info.st_mode) and _is_new(parent, name, info, before_time):
            os.rmdir(name, dir_fd=parent)  # fails when something is in it
    finally:
        os.close(parent)


def _put_back(entry: DiffEntry, blobs: dict[str, bytes], root_fd: int) -> str | None:
    """Restore the `before` side of one path; a skip reason, or None when done."""
    if not _safe_name(entry.path):
        return CHANGED
    assert entry.old_sha is not None
    data = blobs.get(entry.old_sha)
    if data is None:
        return TOO_LARGE
    if entry.old_mode != LINK_MODE and entry.old_mode & 0o170000 != 0o100000:
        return "not a regular file"
    name = entry.path.rpartition("/")[2]
    parent = _open_parent(root_fd, entry.path, create=False)
    try:
        current = None if parent is None else _state(parent, name)
        if current != _expected(entry):
            return CHANGED
        if parent is None:
            parent = _open_parent(root_fd, entry.path, create=True)
        assert parent is not None
        _write(parent, name, entry, data)
    finally:
        if parent is not None:
            os.close(parent)
    return None


def _already_undone(change: FileChange) -> bool:
    """Does the file already hold what it held before the turn (or not exist, when the turn created it)?"""
    if change.too_large or change.broken is not None:
        return False
    try:
        if not change.existed:
            return not os.path.lexists(change.path)
        return change.before is not None and _hash_nofollow(change.path) == hashlib.sha256(change.before).digest()
    except OSError:
        return False


def restore_unsnapshotted(
    turn: UndoTurn,
    entries: list[DiffEntry],
    workspace: Path,
    is_protected: Callable[[Path], bool],
    skip: Mapping[str, str] | None = None,
) -> UndoResult:
    """Undo, from the file-level log, what the turn's write/edit calls changed in files the checkpoints leave out
    (ignored files, `.agent/` files). Files in `entries` (the checkpoint restore did them), protected files and files
    that already hold their old content are left out. Blocking; never raises."""
    root = workspace.resolve()
    covered = {e.path.casefold() for e in entries}  # the filesystem may be case-insensitive (macOS)
    changes: dict[Path, FileChange] = {}
    for path, change in turn.changes.items():
        try:
            if path.relative_to(root).as_posix().casefold() in covered:
                continue
        except ValueError:
            pass  # outside the workspace: the file-level restore names it as skipped
        if is_protected(path) or _already_undone(change):
            continue
        changes[path] = change
    if not changes:
        return UndoResult([])
    return restore_turn(replace(turn, changes=changes), workspace, skip)
