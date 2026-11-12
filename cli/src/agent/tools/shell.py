"""The bash tool: run a shell command with a timeout, in its own process group."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from asyncio.subprocess import DEVNULL, PIPE, STDOUT, Process
from typing import Any, ClassVar

from .base import Prepared, Preview, Tool, ToolContext, ToolResult
from .text import strip_ansi


class BashTool(Tool):
    name: ClassVar[str] = "bash"
    description: ClassVar[str] = (
        "Run a shell command in the workspace and return its combined stdout/stderr and exit code. "
        "The command has no stdin and is killed after the timeout. Each call needs user approval."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
        },
        "required": ["command"],
    }
    needs_approval: ClassVar[bool] = True

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        command: str = args["command"]
        preview = Preview("command", f"bash {command}", command)

        async def execute() -> ToolResult:
            return await _run(command, ctx)

        return Prepared(preview, execute)


# Output is kept bounded while the command runs: the first HEAD bytes and the last TAIL bytes.
# Together they stay under the loop's TOOL_OUTPUT_LIMIT (30,000 chars) with room for the marker.
HEAD = 10_000
TAIL = 15_000
_CHUNK = 65_536


class _Capture:
    """First `HEAD` bytes plus a rolling last `TAIL` bytes of a stream; counts what is dropped."""

    def __init__(self) -> None:
        self._head = bytearray()
        self._tail = bytearray()
        self._dropped = 0

    def feed(self, chunk: bytes) -> None:
        if len(self._head) < HEAD:
            take = chunk[: HEAD - len(self._head)]
            self._head += take
            chunk = chunk[len(take) :]
        self._tail += chunk
        excess = len(self._tail) - TAIL
        if excess > 0:
            self._dropped += excess
            del self._tail[:excess]

    def text(self) -> str:
        if self._dropped:
            data = bytes(self._head) + f"\n[{self._dropped} bytes omitted]\n".encode() + bytes(self._tail)
        else:
            data = bytes(self._head) + bytes(self._tail)
        return strip_ansi(data.decode("utf-8", errors="replace")).rstrip("\n")


async def _drain(proc: Process, capture: _Capture) -> None:
    """Read the merged output to EOF (never leave the pipe full), then reap the shell."""
    assert proc.stdout is not None
    while chunk := await proc.stdout.read(_CHUNK):
        capture.feed(chunk)
    await proc.wait()


async def _run(command: str, ctx: ToolContext) -> ToolResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            os.environ.get("SHELL") or "/bin/sh",
            "-c",
            command,
            cwd=ctx.workspace,
            stdin=DEVNULL,  # a command that reads stdin gets EOF instead of hanging
            stdout=PIPE,
            stderr=STDOUT,
            start_new_session=True,  # own session + process group, so we can kill the whole tree
            env={**os.environ, "NO_COLOR": "1"},
        )
    except OSError as exc:  # missing shell, missing or unreadable workspace
        return ToolResult(f"Could not start shell: {exc}", is_error=True, summary="failed to start")
    capture = _Capture()
    try:
        await asyncio.wait_for(_drain(proc, capture), ctx.bash_timeout)
    except TimeoutError:
        await _kill_group(proc)
        message = f"Command timed out after {ctx.bash_timeout:g}s and was killed"
        text = capture.text()
        return ToolResult(f"{text}\n{message}" if text else message, is_error=True, summary="timed out")
    except BaseException:  # CancelledError (Esc) or anything unexpected: never leave a process behind
        await _kill_group(proc)
        raise
    text = capture.text()
    code = proc.returncode
    output = f"{text}\nexit {code}" if text else f"exit {code}"
    return ToolResult(output, summary=f"exit {code}")


async def _kill_group(proc: Process) -> None:
    """SIGKILL the command's whole process group, then reap the shell."""
    # Ignore "already gone"; macOS answers EPERM for a group that holds only zombies.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()
