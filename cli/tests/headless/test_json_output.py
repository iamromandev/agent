"""`json_line`, `Outcome` and `JsonCollector`: one parseable line per object, redaction, steps, usage, timing."""

from __future__ import annotations

import io
import json
import time

import pytest
from src.engine.events import Compacting, Notice, TaskProgress, ToolFinished, ToolStarted, UsageUpdate
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


@pytest.mark.parametrize("value", ["a\x1b[2Jb", "line1\nline2\r\n", "tab\there\x00\x1f", "café \u2028", "lone \ud800 surrogate"])
def test_json_line_is_one_parseable_line_for_any_text(value: str) -> None:
    line = json_line({"v": value})

    assert "\n" not in line and "\r" not in line
    assert all(ord(ch) >= 0x20 for ch in line)
    line.encode("utf-8")  # writable
    assert json.loads(line)["v"] is not None


def test_json_line_keeps_letters_readable_and_is_compact() -> None:
    assert json_line({"a": "café", "b": [1, None]}) == '{"a":"café","b":[1,null]}'


def test_the_result_has_every_field_in_order() -> None:
    c, _ = _collector()
    _call(c, "c1", "read", {"path": "a.py"}, summary="12 lines")
    c(UsageUpdate(15, 100, prompt_tokens=10, completion_tokens=5))

    obj = c.result(Outcome(0, "done"), "done", "sid", "m")

    assert list(obj) == [
        "type", "result", "is_error", "exit_code", "stop_reason", "error", "session_id", "model", "duration_ms", "steps", "usage", "tool_calls",
    ]
    assert (obj["type"], obj["result"], obj["is_error"], obj["exit_code"], obj["stop_reason"], obj["error"]) == ("result", "done", False, 0, "done", None)
    assert (obj["session_id"], obj["model"], obj["steps"]) == ("sid", "m", 1)
    assert obj["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert obj["tool_calls"] == [{"id": "c1", "name": "read", "args": {"path": "a.py"}, "summary": "12 lines", "is_error": False}]
    assert isinstance(obj["duration_ms"], int)


def test_an_outcome_with_an_error_is_an_error_result_and_the_text_is_redacted() -> None:
    c, _ = _collector()

    obj = c.result(Outcome(1, "error", f"bad {TOKEN}"), "", None, None)

    assert (obj["is_error"], obj["exit_code"], obj["stop_reason"], obj["error"]) == (True, 1, "error", "bad [redacted:github-token]")
    assert (obj["session_id"], obj["model"], obj["steps"], obj["tool_calls"]) == (None, None, 0, [])


def test_estimated_updates_are_not_steps_and_tokens_are_summed() -> None:
    c, _ = _collector()

    c(UsageUpdate(5, 100, estimated=True))
    c(UsageUpdate(15, 100, prompt_tokens=10, completion_tokens=5))
    c(UsageUpdate(30, 100, prompt_tokens=20, completion_tokens=10))

    obj = c.result(Outcome(0, "done"), "", None, None)
    assert obj["steps"] == 2 and obj["usage"] == {"input_tokens": 30, "output_tokens": 15}


def test_task_progress_is_ignored() -> None:
    c, stream = _collector(streaming=True)
    c(ToolStarted("t1", "task x", task=True, name="task", args={"prompt": "p"}))
    c(TaskProgress("t1", ToolStarted("k1", "read a", name="read", args={})))
    c(TaskProgress("t1", UsageUpdate(40, 100, prompt_tokens=30, completion_tokens=10)))
    c(TaskProgress("t1", ToolFinished("k1", ToolResult("o", False, "s"), None, True)))
    c(ToolFinished("t1", ToolResult("report", False, "3 reads"), None, True))

    obj = c.result(Outcome(0, "done"), "", None, None)

    assert [t["name"] for t in obj["tool_calls"]] == ["task"] and obj["steps"] == 0
    assert obj["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert [json.loads(line)["type"] for line in stream.getvalue().splitlines()] == ["tool_call", "tool_result"]


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


def test_an_error_result_without_a_summary_says_error() -> None:
    c, _ = _collector()
    _call(c, "c1", "bash", {}, summary="", is_error=True)

    call = c.result(Outcome(1, "error"), "", None, None)["tool_calls"][0]

    assert (call["summary"], call["is_error"]) == ("error", True)


def test_a_call_that_never_finished_is_interrupted() -> None:
    c, _ = _collector()
    c(ToolStarted("c1", "bash x", name="bash", args={}))

    call = c.result(Outcome(130, "interrupted"), "", None, None)["tool_calls"][0]

    assert (call["summary"], call["is_error"]) == ("interrupted", True)


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


def test_without_streaming_nothing_is_written_until_the_result() -> None:
    c, stream = _collector(streaming=False)
    c.start("sid", "m", "/ws")
    _call(c, "c1", "read", {})
    c(Notice("info", "n"))

    assert stream.getvalue() == ""


def test_a_stream_notice_is_redacted() -> None:
    c, stream = _collector(streaming=True)

    c(Notice("info", f"x {TOKEN}"))
    c.flush()

    assert "[redacted:github-token]" in stream.getvalue() and TOKEN not in stream.getvalue()


class _Broken(io.StringIO):
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None: ...


def test_a_closed_stream_does_not_raise_while_collecting() -> None:
    c = JsonCollector(_Broken(), _redact, streaming=True)

    c.start("s", "m", "/")
    _call(c, "c1", "read", {})

    assert c.result(Outcome(0, "done"), "", None, None)["tool_calls"][0]["name"] == "read"


def test_a_hundred_tool_calls_and_the_result_take_under_5_ms() -> None:
    c, _ = _collector()
    started = time.perf_counter()
    for i in range(100):
        _call(c, f"c{i}", "read", {"path": f"f{i}.py"}, summary="1 line")
        c(UsageUpdate(15, 100, prompt_tokens=10, completion_tokens=5))
    json_line(c.result(Outcome(0, "done"), "done", "sid", "m"))

    assert time.perf_counter() - started < 0.005


def test_notices_before_start_are_written_after_it() -> None:
    c, stream = _collector(streaming=True)

    c(Notice("warn", "launch warning"))
    assert stream.getvalue() == ""
    c.start("s", "m", "/")
    c(Notice("info", "later"))

    assert [json.loads(x)["type"] + ":" + json.loads(x).get("text", "") for x in stream.getvalue().splitlines()] == [
        "start:", "notice:launch warning", "notice:later",
    ]


def test_flush_writes_held_notices_for_a_run_that_never_started() -> None:
    c, stream = _collector(streaming=True)

    c(Notice("warn", "w"))
    c.flush()

    assert [json.loads(x)["text"] for x in stream.getvalue().splitlines()] == ["w"]
