"""`json_line`, `Outcome` and `JsonCollector`: one parseable line per object, redaction, steps, usage, timing."""

from __future__ import annotations

import io
import json

import pytest
from src.engine.events import Compacting, Notice, ToolFinished, ToolStarted
from src.engine.tools.base import ToolResult
from src.headless.output import JsonCollector, Outcome, json_line

TOKEN = "ghp_" + "a" * 36


def _redact(text: str) -> str:
    return text.replace(TOKEN, "[redacted:github-token]")


def _collector(streaming: bool = False) -> tuple[JsonCollector, io.StringIO]:
    stream = io.StringIO()
    return JsonCollector(stream, _redact, streaming=streaming), stream


def _call(c: JsonCollector, call_id: str, name: str, args: dict, summary: str = "ok!", is_error: bool = False) -> None:
    c(ToolStarted(call_id, f"{name} x", name=name, args=args))
    c(ToolFinished(call_id, ToolResult("out", is_error, summary), None, True))


@pytest.mark.parametrize("value", ["lone \ud800 surrogate"])
def test_json_line_is_one_parseable_line_for_any_text(value: str) -> None:
    line = json_line({"v": value})

    assert "\n" not in line and "\r" not in line
    assert all(ord(ch) >= 0x20 for ch in line)
    line.encode("utf-8")  # writable
    assert json.loads(line)["v"] is not None


def test_args_are_redacted_recursively_and_non_json_values_become_text() -> None:
    c, _ = _collector()
    c(ToolStarted("c1", "x", name="x", args={"a": f"t {TOKEN}", "n": 3, "l": [f"{TOKEN}", {"d": TOKEN}], "o": object, "b": True, "z": None}))
    c(ToolFinished("c1", ToolResult("", False, ""), None, True))

    call = c.result(Outcome(0, "done"), "", None, None)["tool_calls"][0]

    args = call["args"]
    assert args["a"] == "t [redacted:github-token]" and args["n"] == 3 and args["b"] is True and args["z"] is None
    assert args["l"] == ["[redacted:github-token]", {"d": "[redacted:github-token]"}]
    assert isinstance(args["o"], str)
    assert TOKEN not in json_line(call)
    assert call["summary"] == "ok"


def test_streaming_writes_each_event_as_one_line_in_order() -> None:
    c, stream = _collector(streaming=True)

    c.start("sid", "m", "/ws")
    _call(c, "c1", "read", {"path": "a"}, summary="1 line")
    c(Notice("warn", "careful"))
    c(Compacting(active=True))
    c(Compacting(active=False))

    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert lines == [
        {"type": "start", "session_id": "sid", "model": "m", "cwd": "/ws"},
        {"type": "tool_call", "id": "c1", "name": "read", "args": {"path": "a"}},
        {"type": "tool_result", "id": "c1", "name": "read", "summary": "1 line", "is_error": False},
        {"type": "notice", "level": "warn", "text": "careful"},
        {"type": "notice", "level": "info", "text": "compacting the context"},
    ]


class _Broken(io.StringIO):
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None: ...


def test_a_closed_stream_does_not_raise_while_collecting() -> None:
    c = JsonCollector(_Broken(), _redact, streaming=True)

    c.start("s", "m", "/")
    _call(c, "c1", "read", {})

    assert c.result(Outcome(0, "done"), "", None, None)["tool_calls"][0]["name"] == "read"


def test_flush_writes_held_notices_for_a_run_that_never_started() -> None:
    c, stream = _collector(streaming=True)

    c(Notice("warn", "w"))
    c.flush()

    assert [json.loads(x)["text"] for x in stream.getvalue().splitlines()] == ["w"]
