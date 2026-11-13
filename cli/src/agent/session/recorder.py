"""Appending session records to a JSONL file, safely.

Crash safety (NFR1): every record is one `os.write` of the whole line on an `O_APPEND`
descriptor, so a kill leaves at most a cut last line, which loading skips.
Security (NFR3): the folder is 0700 and the file 0600 whatever the umask.
Failures (R2): records stay queued in order and are retried first on the next call.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.agent.provider.base import Message
from src.agent.session.records import (
    SessionHeader,
    ToolMeta,
    dumps,
    message_to_obj,
)
from src.agent.session.records import title_from as title_from
from src.agent.session.ulid import new_ulid

DIR_MODE = 0o700
FILE_MODE = 0o600


class SessionWriteError(OSError):
    """A record could not be written; it stays queued and is retried on the next call."""


class Recorder(Protocol):
    def message(self, message: Message, meta: ToolMeta | None) -> None: ...

    def compaction(
        self,
        history: Sequence[Message],
        tool_meta: Mapping[str, ToolMeta],
        pruned: int,
        summarized: int,
        before: int,
        after: int,
    ) -> None: ...

    def undo(self, keep: int) -> None: ...

    # All raise SessionWriteError on I/O failure.


class NullRecorder:
    """Records nothing. The Agent's default."""

    def message(self, message: Message, meta: ToolMeta | None) -> None:
        return None

    def compaction(
        self,
        history: Sequence[Message],
        tool_meta: Mapping[str, ToolMeta],
        pruned: int,
        summarized: int,
        before: int,
        after: int,
    ) -> None:
        return None

    def undo(self, keep: int) -> None:
        return None

    def close(self) -> None:
        return None


def _encode(obj: dict[str, Any]) -> bytes:
    return (dumps(obj) + "\n").encode()


def _terminate_last_line(fd: int) -> None:
    """Make the next record start on its own line: a cut tail would otherwise swallow it."""
    size = os.fstat(fd).st_size
    if size == 0 or os.pread(fd, 1, size - 1) == b"\n":
        return
    if os.write(fd, b"\n") != 1:
        raise SessionWriteError("could not end the last line")


def _make_missing_ancestors(path: Path) -> None:
    """Create the folders above the sessions root that don't exist, 0700. Existing ones are left alone."""
    missing: list[Path] = []
    while not os.path.lexists(path):
        missing.append(path)
        path = path.parent
    for folder in reversed(missing):
        with contextlib.suppress(FileExistsError):  # created meanwhile by another process
            os.mkdir(folder, DIR_MODE)


def _secure_dir(path: Path, *, create: bool) -> None:
    """Make `path` a real folder of ours with mode 0700, creating it first when `create`.

    A symlink or a folder of another user is refused: the recorder must not chmod or fill it.
    """
    if create:
        with contextlib.suppress(FileExistsError):
            os.mkdir(path, DIR_MODE)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode):
        raise SessionWriteError(errno.ENOTDIR, f"{path} is not a folder (a symlink is not followed)")
    if info.st_uid != os.getuid():
        raise SessionWriteError(errno.EPERM, f"{path} is not owned by the current user")
    os.chmod(path, DIR_MODE)  # whatever the umask, and retried on every open


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class FileRecorder:
    def __init__(self, directory: Path, workspace: Path, now: Callable[[], datetime] = _utc_now) -> None:
        self._directory = directory
        self._workspace = workspace.resolve()
        self._now = now
        self._path: Path | None = None
        self._started = False  # header (and title) queued
        self._creating = False  # a new session whose file does not exist yet
        self._fd: int | None = None
        self._pending: list[bytes] = []

    @classmethod
    def resume(cls, path: Path, workspace: Path) -> FileRecorder:
        """Append to an existing session file: no header, no title."""
        recorder = cls(path.parent, workspace)
        recorder._path = path
        recorder._started = True
        return recorder

    @property
    def path(self) -> Path | None:
        return self._path

    def message(self, message: Message, meta: ToolMeta | None) -> None:
        stamp = _stamp(self._now())
        line = _encode({**message_to_obj(message, meta), "at": stamp})  # build everything before changing state
        self._start(message, stamp)
        self._pending.append(line)
        self._flush()

    def compaction(
        self,
        history: Sequence[Message],
        tool_meta: Mapping[str, ToolMeta],
        pruned: int,
        summarized: int,
        before: int,
        after: int,
    ) -> None:
        stamp = _stamp(self._now())
        line = _encode(
            {
                "type": "compaction",
                "history": [message_to_obj(m, tool_meta.get(m.tool_call_id or ""), embedded=True) for m in history],
                "pruned": pruned,
                "summarized": summarized,
                "before": before,
                "after": after,
                "at": stamp,
            }
        )
        self._start(None, stamp)
        self._pending.append(line)
        self._flush()

    def undo(self, keep: int) -> None:
        stamp = _stamp(self._now())
        line = _encode({"type": "undo", "keep": keep, "at": stamp})
        self._start(None, stamp)
        self._pending.append(line)
        self._flush()

    def close(self) -> None:
        """Make one last attempt to write queued records, then release the file. Never raises."""
        try:
            if self._pending:
                self._flush()
        except OSError:
            pass
        fd, self._fd = self._fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _start(self, first: Message | None, stamp: str) -> None:
        """Queue the header, and the title when the first message is the user's.

        The records are built first and the state changes after, so a failure here starts nothing.
        """
        if self._started:
            return
        session_id = new_ulid()
        lines = [_encode(SessionHeader(session_id, str(self._workspace), stamp).to_obj())]
        if first is not None and first.role == "user" and (title := title_from(first.content)):
            lines.append(_encode({"type": "title", "title": title, "at": stamp}))
        self._path = self._directory / f"{session_id}.jsonl"
        self._started = True
        self._creating = True
        self._pending.extend(lines)

    def _open(self) -> int:
        if self._fd is not None:
            return self._fd
        assert self._path is not None
        # Every open checks the folders again, so a failed chmod is retried and a swapped folder is caught.
        # Only a new session creates folders (and its file, and never over an existing one). Resuming or
        # reopening something that is gone must fail instead of leaving a headerless file listing skips forever.
        root = self._directory.parent
        if self._creating:
            _make_missing_ancestors(root.parent)
            _secure_dir(root, create=True)
        else:
            _secure_dir(root, create=False)
        _secure_dir(self._directory, create=self._creating)
        flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC  # RDWR: the tail is read
        if self._creating:
            flags |= os.O_CREAT | os.O_EXCL
        fd = os.open(self._path, flags, FILE_MODE)
        self._creating = False  # the file now exists: a later reopen must not create it
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise SessionWriteError(errno.EINVAL, f"{self._path} is not a regular file")
            if info.st_uid != os.getuid():
                raise SessionWriteError(errno.EPERM, f"{self._path} is not owned by the current user")
            os.fchmod(fd, FILE_MODE)
            _terminate_last_line(fd)
        except OSError:
            os.close(fd)
            raise
        self._fd = fd
        return fd

    def _flush(self) -> None:
        try:
            fd = self._open()
            while self._pending:
                data = self._pending[0]
                written = os.write(fd, data)
                if written != len(data):
                    # Keep the unwritten tail at the head of the queue, so that the retry
                    # completes this line instead of repeating its first part.
                    if written > 0:
                        self._pending[0] = data[written:]
                    raise SessionWriteError(f"short write ({written} of {len(data)} bytes)")
                self._pending.pop(0)
        except SessionWriteError:
            raise
        except OSError as error:
            raise SessionWriteError(error.errno, error.strerror or str(error)) from error
