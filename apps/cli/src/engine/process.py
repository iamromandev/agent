"""What every spawn site that runs a shell command shares: the shell to use and how to kill a process group."""

from __future__ import annotations

import contextlib
import os
import signal
from asyncio.subprocess import Process


def shell_argv(command: str) -> list[str]:
    """`$SHELL -c command` (`/bin/sh` when `$SHELL` is unset or empty)."""
    return [os.environ.get("SHELL") or "/bin/sh", "-c", command]


async def kill_group(proc: Process) -> None:
    """SIGKILL the command's whole process group, then reap the shell."""
    # Ignore "already gone"; macOS answers EPERM for a group that holds only zombies.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()
