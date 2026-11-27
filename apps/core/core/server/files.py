"""The daemon's pid file (an `flock` held for as long as it lives) and its listening socket (0600, made by us, never by
uvicorn: its `Config(uds=)` chmods the socket to 0666)."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import socket
import stat
from pathlib import Path

from core.engine.session.lock import same_file

FILE_MODE = 0o600
BACKLOG = 64
ATTEMPTS = 5  # a pid file removed under us between open and flock is retried this often


class AlreadyRunning(Exception):
    """Another daemon holds the workspace's pid file."""

    def __init__(self, pid: int | None) -> None:
        super().__init__(f"pid {pid}" if pid is not None else "unknown pid")
        self.pid = pid


def _read_pid(fd: int) -> int | None:
    with contextlib.suppress(OSError, ValueError):
        os.lseek(fd, 0, os.SEEK_SET)
        return int(os.read(fd, 32).decode().strip())
    return None


class PidLock:
    """The pid file, locked: one daemon per workspace."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd: int | None = fd

    @classmethod
    def acquire(cls, path: Path) -> PidLock:
        """Lock `path` and write our pid into it. `AlreadyRunning` (with the holder's pid) when another daemon has it.
        The last holder removes the file while it still holds it, so after locking the path must still name the file
        that was locked; if not, the new file is locked instead."""
        for _ in range(ATTEMPTS):
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, FILE_MODE)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise OSError(errno.EINVAL, f"{path} is not a regular file")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise AlreadyRunning(_read_pid(fd)) from None
                if not same_file(fd, path):
                    os.close(fd)  # removed by its last holder meanwhile: lock the new one
                    continue
                cls._write_pid(fd)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            return cls(path, fd)
        raise OSError(errno.EAGAIN, f"could not lock {path}")

    @staticmethod
    def _write_pid(fd: int) -> None:
        os.fchmod(fd, FILE_MODE)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())

    def release(self) -> None:
        """Remove the file and unlock. Never raises."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            os.unlink(self.path)  # while still locked: a new daemon makes a new file
        with contextlib.suppress(OSError):
            os.close(fd)


def read_pid(path: Path) -> int | None:
    """The pid in a pid file, or None (no file, or not a number)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        return _read_pid(fd)
    finally:
        os.close(fd)


def pid_held(path: Path) -> bool:
    """A daemon holds the pid file's lock now (it is alive; a zombie has let go). Never raises."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError:
        return False  # no file: released (the holder removes it before it unlocks)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)  # also drops the lock this probe took
    return False


def remove_stale_socket(path: Path) -> None:
    """A socket file nobody listens on (a daemon that was killed) is removed. A socket someone still listens on, or a
    file that is not a socket, is an `OSError`: it is never removed."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(info.st_mode):
        raise OSError(errno.EEXIST, f"{path} exists and is not a socket")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(path))
    except ConnectionRefusedError:
        os.unlink(path)  # stale
        return
    except FileNotFoundError:
        return
    finally:
        probe.close()
    raise OSError(errno.EADDRINUSE, f"{path} is in use by another process")


def bind_socket(path: Path) -> socket.socket:
    """A listening AF_UNIX socket at `path` with mode 0600 from the start: bound under umask 0o177 and chmodded before
    `listen`, so no other user can connect between the two."""
    remove_stale_socket(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        old = os.umask(0o177)
        try:
            sock.bind(str(path))
        finally:
            os.umask(old)
        os.chmod(path, FILE_MODE)
        sock.listen(BACKLOG)
        sock.setblocking(False)
    except BaseException:
        sock.close()
        raise
    return sock
