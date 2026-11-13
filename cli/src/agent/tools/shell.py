"""The bash tool: run a shell command with a timeout, in its own process group."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
from asyncio.subprocess import DEVNULL, PIPE, STDOUT, Process
from collections import deque
from typing import Any, ClassVar

from .base import Prepared, Preview, Tool, ToolContext, ToolResult
from .text import DANGLING, ESCAPE, SGR, keep_sgr, strip_ansi


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


_ESCAPE = re.compile(ESCAPE.pattern.encode())
_SGR = re.compile(SGR.pattern.encode())
_PARTIAL = re.compile(DANGLING.pattern.encode())
_MAX_TOKENS = 65_536  # style sequences without text between them must not grow the buffers
_SGR_MAX = 67  # longest style token SGR allows


def _trim_trailing(text: str) -> str:
    """Drop the trailing newlines, but keep the style sequences among them (a reset). Linear: walks back."""
    end = len(text)
    kept: list[str] = []
    while end:
        if text[end - 1] == "\n":
            end -= 1
            continue
        start = text.rfind("\x1b", max(0, end - _SGR_MAX), end) if text[end - 1] == "m" else -1
        if start < 0 or not SGR.fullmatch(text, start, end):
            break
        kept.append(text[start:end])
        end = start
    return text[:end] + "".join(reversed(kept))


class _Capture:
    """First `HEAD` and last `TAIL` bytes of the text of a stream; counts what is dropped.

    Escape sequences do not count towards those caps: colors (SGR) are kept in place, every other
    sequence is dropped on arrival, and a sequence cut by a chunk boundary is completed first. So colored
    output keeps as much text as plain output, and the plain copy is the display copy with its colors removed.
    """

    def __init__(self) -> None:
        self._head = bytearray()
        self._head_text = 0
        self._tail: deque[tuple[bytes, bool]] = deque()  # (bytes, is_text)
        self._tail_text = 0
        self._dropped = 0
        self._pending = b""
        self._raw: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._raw = None
        data = self._pending + chunk
        partial = _PARTIAL.search(data)
        self._pending = data[partial.start() :] if partial else b""
        if partial:
            data = data[: partial.start()]
        position = 0
        for match in _ESCAPE.finditer(data):
            self._text(data[position : match.start()])
            if _SGR.fullmatch(match.group()):
                self._style(match.group())
            position = match.end()
        self._text(data[position:])

    def _head_open(self) -> bool:
        return self._head_text < HEAD and not self._tail and len(self._head) < 16 * HEAD

    def _style(self, sequence: bytes) -> None:
        if self._head_open():
            self._head += sequence
        else:
            self._tail.append((sequence, False))
            self._trim()

    def _text(self, piece: bytes) -> None:
        if not piece:
            return
        if self._head_open():
            take = piece[: HEAD - self._head_text]
            self._head += take
            self._head_text += len(take)
            piece = piece[len(take) :]
        if piece:
            self._tail.append((piece, True))
            self._tail_text += len(piece)
        self._trim()

    def _trim(self) -> None:
        while self._tail:
            crowded = len(self._tail) > _MAX_TOKENS
            excess = self._tail_text - TAIL
            if excess <= 0 and not crowded:
                break
            data, is_text = self._tail.popleft()
            if not is_text:
                continue
            drop = len(data) if crowded else min(len(data), excess)
            if drop < len(data):
                self._tail.appendleft((data[drop:], True))
            self._dropped += drop
            self._tail_text -= drop

    def _decoded(self) -> str:
        tail = b"".join(data for data, _ in self._tail)
        # the tail's style state is unknown after the cut: reset what the head left open
        reset = b"\x1b[0m" if b"\x1b" in self._head else b""
        marker = reset + f"\n[{self._dropped} bytes omitted]\n".encode() if self._dropped else b""
        return keep_sgr(bytes(self._head + marker + tail).decode("utf-8", errors="replace"))

    def raw(self) -> str:
        """What the screen shows: colors and styles kept, every other escape sequence removed."""
        if self._raw is None:
            self._raw = _trim_trailing(self._decoded())
        return self._raw

    def text(self) -> str:
        """What the model sees: the screen's text without any escape sequence."""
        return strip_ansi(self.raw())


async def _drain(proc: Process, capture: _Capture) -> None:
    """Read the merged output to EOF (never leave the pipe full), then reap the shell."""
    assert proc.stdout is not None
    while chunk := await proc.stdout.read(_CHUNK):
        capture.feed(chunk)
    await proc.wait()


def _environment() -> dict[str, str]:
    """The parent's environment, asking tools for color (the screen shows it; the model gets it stripped)."""
    env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"}
    return env | {"FORCE_COLOR": "1", "CLICOLOR_FORCE": "1"}


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
            env=_environment(),
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
        raw = capture.raw()
        display = (f"{raw}\n{message}" if raw else message) if raw != text else None
        return ToolResult(
            f"{text}\n{message}" if text else message, is_error=True, summary="timed out", display=display
        )
    except BaseException:  # CancelledError (Esc) or anything unexpected: never leave a process behind
        await _kill_group(proc)
        raise
    text = capture.text()
    code = proc.returncode
    output = f"{text}\nexit {code}" if text else f"exit {code}"
    raw = capture.raw()
    display = (f"{raw}\nexit {code}" if raw else f"exit {code}") if raw != text else None
    return ToolResult(output, summary=f"exit {code}", display=display)


async def _kill_group(proc: Process) -> None:
    """SIGKILL the command's whole process group, then reap the shell."""
    # Ignore "already gone"; macOS answers EPERM for a group that holds only zombies.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()
