"""The terminal tools: start a long-running process in a named terminal, read its output, stop it."""

from __future__ import annotations

import signal
from typing import Any, ClassVar

from src.agent.terminal.manager import Status, TerminalError, TerminalManager

from .base import Prepared, Preview, Tool, ToolContext, ToolResult
from .text import TOOL_OUTPUT_LIMIT

DEFAULT_LINES = 100
MAX_LINES = 5_000
_RESERVE = 200  # room kept in the result for the omitted-lines note and the status line


def _describe(status: Status) -> str:
    """`running`, `exited 3` or `exited: SIGTERM`."""
    state, code = status
    if state == "running" or code is None:
        return "running"
    if code >= 0:
        return f"exited {code}"
    try:
        return f"exited: {signal.Signals(-code).name}"
    except ValueError:
        return f"exited: signal {-code}"


def _unknown(manager: TerminalManager, name: str) -> ToolResult:
    names = ", ".join(manager.names()) or "none"
    return ToolResult(f"no terminal named {name}; terminals: {names}", True, "no such terminal")


class TermStart(Tool):
    name: ClassVar[str] = "term_start"
    description: ClassVar[str] = (
        "Start a long-running command (a dev server, a watcher) in a named terminal tab in the workspace and "
        "return at once. Read its output with term_read and end it with term_stop. Use this instead of bash "
        "for commands that do not exit. Each call needs user approval."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Terminal name: 1-32 letters, digits, - or _."},
            "command": {"type": "string", "description": "The shell command to run."},
        },
        "required": ["name", "command"],
    }
    needs_approval: ClassVar[bool] = True
    mutating: ClassVar[bool] = True
    grant_arg: ClassVar[str | None] = "command"  # "Always" covers that command only, like bash

    def __init__(self, terminals: TerminalManager) -> None:
        self._terminals = terminals

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        name: str = args["name"]
        command: str = args["command"]
        try:
            self._terminals.check_name(name)
        except TerminalError as exc:
            return ToolResult(str(exc), True, "refused")

        async def execute() -> ToolResult:
            try:
                terminal = self._terminals.start(name, command)
            except TerminalError as exc:  # taken while the call waited for approval
                return ToolResult(str(exc), True, "refused")
            return ToolResult(f"started {name} (pid {terminal.process.pid})", summary="started")

        return Prepared(Preview("command", f"term_start {name}", command), execute)


class TermRead(Tool):
    name: ClassVar[str] = "term_read"
    description: ClassVar[str] = (
        "Return the most recent output of a terminal started with term_start (or the user's shell) as plain "
        "text, then whether it is still running or how it exited."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The terminal name."},
            "lines": {
                "type": "integer",
                "description": f"How many of the last lines to return (default {DEFAULT_LINES}, at most {MAX_LINES}).",
            },
        },
        "required": ["name"],
    }

    def __init__(self, terminals: TerminalManager) -> None:
        self._terminals = terminals

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        name: str = args["name"]
        terminal = self._terminals.get(name)
        if terminal is None:
            return _unknown(self._terminals, name)
        wanted: int = args.get("lines", DEFAULT_LINES)
        lines = terminal.buffer.tail(max(1, min(wanted, MAX_LINES)))
        status = _describe(terminal.status)
        # Keep the newest lines that fit: the loop would cut the end, and the status is at the end.
        budget = TOOL_OUTPUT_LIMIT - _RESERVE
        first = len(lines)  # index of the oldest line kept
        used = 0
        while first and used + len(lines[first - 1]) + 1 <= budget:
            first -= 1
            used += len(lines[first]) + 1
        shown = lines[first:]
        text = "\n".join(shown)
        if first:
            text = f"[{first} earlier lines omitted]\n{text}"
        output = f"{text}\n\n{status}" if text else status
        return ToolResult(output, summary=f"{len(shown)} lines, {status}")


class TermStop(Tool):
    name: ClassVar[str] = "term_stop"
    description: ClassVar[str] = (
        "Stop a terminal's whole process group (SIGTERM, then SIGKILL after a grace period). The tab stays, "
        "marked exited. Each call needs user approval."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The terminal name."}},
        "required": ["name"],
    }
    needs_approval: ClassVar[bool] = True
    mutating: ClassVar[bool] = True

    def __init__(self, terminals: TerminalManager) -> None:
        self._terminals = terminals

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        name: str = args["name"]
        if self._terminals.get(name) is None:
            return _unknown(self._terminals, name)

        async def execute() -> ToolResult:
            terminal = self._terminals.get(name)
            if terminal is None:  # closed while the call waited for approval
                return _unknown(self._terminals, name)
            if terminal.status[0] == "exited":
                return ToolResult(f"{name} already exited ({_describe(terminal.status)})", summary="already exited")
            status = await terminal.stop(self._terminals.grace)
            if status[0] == "running":  # SIGKILL did not take it down in time (stuck in the kernel)
                return ToolResult(f"could not stop {name}: it is still running after SIGKILL", True, "still running")
            if terminal.survivors():  # SIGKILL did not take down everything of the terminal
                return ToolResult(
                    f"stopped {name} ({_describe(status)}), but some of its processes are still running",
                    True,
                    "partly stopped",
                )
            return ToolResult(f"stopped {name} ({_describe(status)})", summary="stopped")

        return Prepared(Preview("command", f"term_stop {name}", f"stop {name}"), execute)
