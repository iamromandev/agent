"""The session lock: `<sessions>/<id>.lock`, an `flock` held by whichever process has the session's recorder open, so a
session open in one agent process (the app, headless mode or the daemon) is refused by another.

flock locks belong to an open file, so a second open in the same process would conflict with the first: locks this
process holds are kept in a registry and shared (counted), and `held_elsewhere` leaves them out. The lock file is
removed on the last release; `acquire` checks after locking that the path still names the file it locked, so a
process that locked a file just removed tries again."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import stat
import threading
from pathlib import Path

from core.engine.session.store import SessionError

FILE_MODE = 0o600
ATTEMPTS = 5  # a lock file removed under us between open and flock is retried this often

_held: dict[Path, SessionLock] = {}  # this process's locks, by lock file path
_registry = threading.Lock()


class SessionLocked(SessionError):
    """Another process has the session open."""


def lock_path(directory: Path, session_id: str) -> Path:
    return directory / f"{session_id}.lock"


def _open(path: Path, *, create: bool) -> int:
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | (os.O_CREAT if create else 0)
    fd = os.open(path, flags, FILE_MODE)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.EINVAL, f"{path} is not a regular file")
        if info.st_uid != os.getuid():
            raise OSError(errno.EPERM, f"{path} is not owned by the current user")
        os.fchmod(fd, FILE_MODE)
    except OSError:
        os.close(fd)
        raise
    return fd


def _same_file(fd: int, path: Path) -> bool:
    try:
        now = os.lstat(path)
    except FileNotFoundError:
        return False
    mine = os.fstat(fd)
    return (now.st_dev, now.st_ino) == (mine.st_dev, mine.st_ino)


class SessionLock:
    def __init__(self, path: Path, fd: int, session_id: str) -> None:
        self.path = path
        self.session_id = session_id
        self._fd: int | None = fd
        self._count = 1

    @classmethod
    def acquire(cls, directory: Path, session_id: str) -> SessionLock:
        """Lock `session_id` in `directory` (created 0600, never through a symlink). `SessionLocked` when another
        process holds it; `OSError` when the file cannot be used. In this process the lock is shared."""
        path = lock_path(directory, session_id)
        with _registry:
            mine = _held.get(path)
            if mine is not None:
                mine._count += 1
                return mine
            for _ in range(ATTEMPTS):
                fd = _open(path, create=True)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    raise SessionLocked(f"session {session_id} is open in another agent process") from None
                except OSError:
                    os.close(fd)
                    raise
                if _same_file(fd, path):
                    lock = cls(path, fd, session_id)
                    _held[path] = lock
                    return lock
                os.close(fd)  # the file was removed by its last holder meanwhile: lock the new one
            raise OSError(errno.EAGAIN, f"could not lock {path}")

    def release(self) -> None:
        """Give up this hold; the last one removes the file and unlocks. Never raises."""
        with _registry:
            if self._fd is None:
                return
            self._count -= 1
            if self._count > 0:
                return
            fd, self._fd = self._fd, None
            if _held.get(self.path) is self:
                del _held[self.path]
            with contextlib.suppress(OSError):
                if _same_file(fd, self.path):
                    os.unlink(self.path)  # while still locked: whoever opens it now finds no file or a new one
            with contextlib.suppress(OSError):
                os.close(fd)  # unlocks

    @staticmethod
    def held_by_us(directory: Path, session_id: str) -> bool:
        with _registry:
            return lock_path(directory, session_id) in _held

    @staticmethod
    def held_elsewhere(directory: Path, session_id: str) -> bool:
        """Another process has the session open (a lock this process holds does not count). Never raises."""
        path = lock_path(directory, session_id)
        with _registry:
            if path in _held:
                return False
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        except OSError:
            return False  # no lock file: nobody has it open
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return False
        finally:
            os.close(fd)  # also unlocks what this probe took
        return False
