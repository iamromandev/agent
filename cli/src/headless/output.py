"""What headless mode writes: progress lines on stderr, the final reply on stdout."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TextIO

from src.engine.events import AgentEvent, Compacting, Notice, ToolFinished, ToolStarted, UsageUpdate
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


StopReason = Literal["done", "error", "step_limit", "blocked", "interrupted", "usage"]


@dataclass
class Outcome:
    """How a headless run ended: the exit code, why, and the error text when there is one (raw, not escaped)."""

    exit_code: int
    stop_reason: StopReason
    error: str | None = None


def json_line(obj: object) -> str:
    """`obj` as compact JSON on one line. Control characters are always escaped; text stays readable unless it
    holds a lone surrogate (not writable as UTF-8), which falls back to `\\u` escapes for the whole object."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    return text


class JsonCollector:
    """An event sink for `--output json|stream-json`: it records the main agent's tool calls, steps and usage for
    the final `result` object and, when `streaming`, writes each event as one NDJSON line as it happens. Text from
    the run goes through `redact`; the child agent's events (`TaskProgress`) are ignored."""

    def __init__(self, stream: TextIO | None, redact: Callable[[str], str], *, streaming: bool) -> None:
        self._stream = stream
        self._redact = redact
        self._streaming = streaming
        self._began = time.perf_counter()
        self._pending: list[dict[str, object]] | None = []  # notices from before `start`, written right after it
        self._calls: list[dict[str, object]] = []
        self._open: dict[str, dict[str, object]] = {}  # call id -> its entry until ToolFinished
        self._steps = 0
        self._input = 0
        self._output = 0

    def __call__(self, event: AgentEvent) -> None:
        match event:
            case ToolStarted(call_id, _, _, name, args):
                entry: dict[str, object] = {"id": call_id, "name": self._redact(name), "args": self._args(args), "summary": "interrupted", "is_error": True}
                self._calls.append(entry)
                self._open[call_id] = entry
                self._emit({"type": "tool_call", "id": call_id, "name": entry["name"], "args": entry["args"]})
            case ToolFinished(call_id, result):
                summary = self._redact(result.summary or ("error" if result.is_error else "ok"))
                entry = self._open.pop(call_id, None) or {"id": call_id, "name": ""}
                entry["summary"], entry["is_error"] = summary, result.is_error
                name = self._redact(str(entry.get("name", "")))
                self._emit({"type": "tool_result", "id": call_id, "name": name, "summary": summary, "is_error": result.is_error})
            case Notice(level, text):
                self._emit({"type": "notice", "level": level, "text": self._redact(text)})
            case Compacting(active=True):
                self._emit({"type": "notice", "level": "info", "text": "compacting the context"})
            case UsageUpdate(estimated=False, prompt_tokens=prompt, completion_tokens=completion):
                self._steps += 1
                self._input += prompt
                self._output += completion
            case _:
                pass

    def start(self, session_id: str | None, model: str | None, cwd: str) -> None:
        """Stream the `start` event, then the notices that came before it (launch warnings)."""
        self._emit({"type": "start", "session_id": session_id, "model": model, "cwd": cwd})
        self.flush()

    def flush(self) -> None:
        """Write the notices held back for `start` (a run that ends before it still shows them)."""
        pending, self._pending = self._pending, None
        for obj in pending or []:
            self._emit(obj)

    def result(self, outcome: Outcome, reply: str, session_id: str | None, model: str | None) -> dict[str, Any]:
        return {
            "type": "result",
            "result": reply,
            "is_error": outcome.exit_code != 0,
            "exit_code": outcome.exit_code,
            "stop_reason": outcome.stop_reason,
            "error": None if outcome.error is None else self._redact(outcome.error),
            "session_id": session_id,
            "model": model,
            "duration_ms": int((time.perf_counter() - self._began) * 1000),
            "steps": self._steps,
            "usage": {"input_tokens": self._input, "output_tokens": self._output},
            "tool_calls": self._calls,
        }

    def _emit(self, obj: dict[str, object]) -> None:
        if self._pending is not None and obj["type"] == "notice":
            self._pending.append(obj)
            return
        if self._streaming and not write_stdout(json_line(obj), self._stream):
            self._streaming = False  # the reader went away: stop writing, keep collecting

    def _args(self, value: object) -> object:
        if isinstance(value, str):
            return self._redact(value)
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, Mapping):
            return {self._redact(str(k)): self._args(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self._args(v) for v in value]
        return self._redact(repr(value))
