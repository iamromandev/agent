"""Where the daemon of a workspace keeps its socket, pid file and log: a private run folder (0700, ours, no symlink)."""

from __future__ import annotations

import contextlib
import hashlib
import os
import stat
import tempfile
from pathlib import Path

DIR_MODE = 0o700
MAX_SOCKET_PATH = 100  # bytes: AF_UNIX paths are limited (104 on macOS, 108 on Linux)


class RunDirError(OSError):
    """The run folder is not a private folder of the current user."""


def workspace_hash(workspace: Path) -> str:
    return hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:16]


def _data_run() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base and os.path.isabs(base) else Path.home() / ".local" / "share"
    return root / "agent" / "run"


def _tmp_run() -> Path:
    base = os.environ.get("TMPDIR")  # read each time (`tempfile.gettempdir` keeps its first answer)
    return Path(base if base and os.path.isabs(base) else tempfile.gettempdir()) / f"agent-{os.getuid()}"


def _base() -> Path:
    """The data folder's run folder, or the temp one when a socket path there would be too long."""
    folder = _data_run()
    longest = folder / f"{'0' * 16}.sock"
    return folder if len(os.fsencode(longest)) <= MAX_SOCKET_PATH else _tmp_run()


def _private(folder: Path) -> Path:
    """Make `folder` (0700) if it is missing and check it is a real folder of ours with mode 0700."""
    if not os.path.lexists(folder):
        folder.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):  # made meanwhile by another process: checked below
            os.mkdir(folder, DIR_MODE)
            os.chmod(folder, DIR_MODE)  # whatever the umask
    info = os.lstat(folder)
    if not stat.S_ISDIR(info.st_mode):
        raise RunDirError(f"{folder} is not a folder (a symlink is not followed)")
    if info.st_uid != os.getuid():
        raise RunDirError(f"{folder} is not owned by the current user")
    if stat.S_IMODE(info.st_mode) != DIR_MODE:
        raise RunDirError(f"{folder} has mode {stat.S_IMODE(info.st_mode):o}, not 700")
    return folder


def run_dir() -> Path:
    """The run folder, made when missing. `RunDirError` when it is a symlink, someone else's, or not 0700."""
    return _private(_base())


def existing_run_dir() -> Path | None:
    """The run folder when it exists (checked like `run_dir`), never made: None when there is none (no daemon ever ran
    here, so none runs now). `RunDirError` when it is not ours alone."""
    folder = _base()
    if not os.path.lexists(folder):
        return None
    return _private(folder)


def existing_socket_path(workspace: Path) -> Path | None:
    """Where the socket of `workspace`'s daemon is, without making the run folder; None when there is no run folder."""
    folder = existing_run_dir()
    return None if folder is None else folder / f"{workspace_hash(workspace)}.sock"


def socket_path(workspace: Path) -> Path:
    return run_dir() / f"{workspace_hash(workspace)}.sock"


def pid_path(workspace: Path) -> Path:
    return run_dir() / f"{workspace_hash(workspace)}.pid"


def log_path(workspace: Path) -> Path:
    return run_dir() / f"{workspace_hash(workspace)}.log"
