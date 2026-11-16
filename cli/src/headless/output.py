"""What headless mode writes: progress lines on stderr, the final reply on stdout."""

from __future__ import annotations

import os
import sys
from typing import TextIO

from src.engine.events import AgentEvent, Compacting, Notice, ToolFinished, ToolStarted
from src.engine.text import escape, escape_display

ARG_LIMIT = 120  # characters of a tool call's argument shown on its line

_COLORS = {"warn": "33", "error": "31", "info": "2"}
_LEVEL_WORDS = {"warn": "warning", "error": "error", "info": "info"}


def want_color(stream: TextIO, env: dict[str, str] | os._Environ[str] = os.environ) -> bool:
    """Colour only for a terminal, and never when `NO_COLOR` is set."""
    return stream.isatty() and not env.get("NO_COLOR")


class StderrSink:
    """An event sink that writes one line per tool call and notice. Text, usage and the child agent's events are
    not progress; with `quiet` only warnings and errors are written. Everything from a file, the model or a tool
    is escaped, so it cannot move the cursor or rewrite the lines before it."""

    def __init__(self, stream: TextIO, quiet: bool, color: bool) -> None:
        self._stream = stream
        self._quiet = quiet
        self._color = color
        self._labels: dict[str, str] = {}  # call id -> label, from ToolStarted to ToolFinished

    def __call__(self, event: AgentEvent) -> None:
        match event:
            case ToolStarted(call_id, label):
                self._labels[call_id] = label
            case ToolFinished(call_id, result):
                label = self._labels.pop(call_id, "")
                if not self._quiet:
                    self._tool_line(label, result.summary, result.is_error)
            case Notice(level, text):
                if level != "info" or not self._quiet:
                    word = _LEVEL_WORDS[level]
                    if self._color:
                        word = f"\x1b[{_COLORS[level]}m{word}\x1b[0m"
                    self._write(f"{word}: {escape_display(text)}")
            case Compacting(active=True) if not self._quiet:
                self._write("compacting the context")
            case _:
                pass

    def _tool_line(self, label: str, summary: str, is_error: bool) -> None:
        name, _, arg = label.partition(" ")
        shown = f"{escape(name)}: {escape(arg[:ARG_LIMIT])}" if arg else escape(name)
        self._write(f"tool {shown} ({escape(summary) or ('error' if is_error else 'ok')})")

    def _write(self, line: str) -> None:
        self._stream.write(line + "\n")
        self._stream.flush()


def write_stdout(text: str, stream: TextIO | None = None) -> bool:
    """Write the final reply and a newline. False when the reader went away (`| head -1`): no traceback, and the
    rest of the process writes to the null device so the interpreter's exit flush cannot fail either."""
    out = stream if stream is not None else sys.stdout
    try:
        out.write(text + "\n")
        out.flush()
        return True
    except BrokenPipeError:
        if out is sys.stdout:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            os.close(devnull)
        return False
