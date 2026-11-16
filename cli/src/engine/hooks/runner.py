"""Run one hook command: sandbox-wrapped like `bash`, filtered environment, bounded output, killed on timeout."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from asyncio.subprocess import PIPE, Process
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.engine.guardrails.setup import Guardrails
from src.engine.process import kill_group, shell_argv

MAX_STREAM = 64 * 1024  # bytes kept of each of stdout and stderr; the rest is read and dropped
REASON_CHARS = 2_000  # a block's reason is cut to this
_CHUNK = 65_536
PIPE_GRACE = 0.5  # seconds the pipes get after the shell exited: a background process may hold them open for ever


@dataclass(frozen=True)
class HookRun:
    exit_code: int | None  # None: it did not start, or it timed out
    stdout: str  # utf-8, undecodable bytes replaced; at most MAX_STREAM bytes
    stderr: str
    timed_out: bool
    error: str | None  # why it did not start


class Runner(Protocol):
    async def run(self, command: str, stdin: bytes, timeout: float, env_extra: Mapping[str, str]) -> HookRun: ...


async def _read(stream: asyncio.StreamReader | None, into: bytearray) -> None:
    """Read to EOF, keeping the first MAX_STREAM bytes: the process must never block on a full pipe."""
    if stream is None:
        return
    while chunk := await stream.read(_CHUNK):
        room = MAX_STREAM - len(into)
        if room > 0:
            into += chunk[:room]


async def _feed(proc: Process, data: bytes) -> None:
    """Write the event to stdin and close it. A command that never reads it is fine (broken pipe)."""
    assert proc.stdin is not None
    try:
        proc.stdin.write(data)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()


class ProcessRunner:
    def __init__(self, workspace: Path, guard: Guardrails) -> None:
        self._workspace = workspace
        self._guard = guard
        self._live: set[Process] = set()  # hooks whose shell has not exited yet
        self._closed = False

    def kill_all(self) -> None:
        """SIGKILL the process group of every hook still running and start no more. Synchronous, so it works when the
        app's loop is gone. A process a finished hook left in the background is not a running hook: it stays."""
        self._closed = True
        for proc in list(self._live):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)

    async def run(self, command: str, stdin: bytes, timeout: float, env_extra: Mapping[str, str]) -> HookRun:
        if self._closed:
            return HookRun(None, "", "", False, "could not start: hooks are closed")
        guard = self._guard
        argv = guard.sandbox.wrap(shell_argv(command))
        # The parent's environment as it is (no colour forcing: a hook's text is read by a person and a model).
        env = guard.env.filter(os.environ) | dict(env_extra)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._workspace,
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                start_new_session=True,  # own process group, so the whole tree can be killed
                env=env,
            )
        except OSError as exc:  # missing shell or workspace
            return HookRun(None, "", "", False, f"could not start: {exc}")
        self._live.add(proc)
        out, err = bytearray(), bytearray()
        pipes = [
            asyncio.ensure_future(job) for job in (_feed(proc, stdin), _read(proc.stdout, out), _read(proc.stderr, err))
        ]
        try:
            async with asyncio.timeout(timeout):
                await proc.wait()
                self._live.discard(proc)  # reaped: from here on only what it left behind could be in the group
                # The shell is done. What it wrote is read in a moment; a process it left running in the background
                # (the user started it on purpose) may keep the pipes open: stop reading, keep what came, leave it be.
                await asyncio.wait(pipes, timeout=PIPE_GRACE)
        except TimeoutError:
            await kill_group(proc)
            return HookRun(None, _text(out), _text(err), True, None)
        except BaseException:  # CancelledError (Esc) or anything unexpected: never leave a process behind
            await kill_group(proc)
            raise
        finally:
            self._live.discard(proc)
            for pipe in pipes:
                pipe.cancel()
            await asyncio.gather(*pipes, return_exceptions=True)
        return HookRun(proc.returncode, _text(out), _text(err), False, None)


def _text(data: bytearray) -> str:
    return bytes(data).decode("utf-8", errors="replace")
