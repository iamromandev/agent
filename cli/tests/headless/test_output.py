"""`StderrSink`: one line per tool call and notice, quiet mode, no colour unless asked, escaped text."""

from __future__ import annotations

import io

from src.engine.approval import ApprovalRequest, Choice
from src.engine.events import AssistantText, Compacting, Notice, TaskProgress, ToolFinished, ToolStarted, TurnEnded
from src.engine.tools.base import ToolResult
from src.headless.approver import HeadlessApprover
from src.headless.output import StderrSink


def _sink(quiet: bool = False, color: bool = False) -> tuple[StderrSink, io.StringIO]:
    stream = io.StringIO()
    return StderrSink(stream, quiet=quiet, color=color), stream


def _call(sink: StderrSink, label: str, summary: str = "", is_error: bool = False, call_id: str = "1") -> None:
    sink(ToolStarted(call_id, label))
    sink(ToolFinished(call_id, ToolResult("out", is_error, summary), None, True))


def test_a_tool_call_is_one_line_with_the_name_the_argument_and_the_summary() -> None:
    sink, stream = _sink()

    _call(sink, "read src/x.py", "31 lines")

    assert stream.getvalue() == "tool read: src/x.py (31 lines)\n"


def test_a_call_without_a_summary_or_argument_says_ok_or_error() -> None:
    sink, stream = _sink()

    _call(sink, "todo", "", call_id="1")
    _call(sink, "bash false", "", is_error=True, call_id="2")

    assert stream.getvalue() == "tool todo (ok)\ntool bash: false (error)\n"


def test_the_argument_is_cut_to_120_characters_and_escaped() -> None:
    sink, stream = _sink()

    _call(sink, "bash " + "x" * 200, "exit 0", call_id="1")
    _call(sink, "bash echo \x1b[31mred\nnext", "exit 0", call_id="2")

    first, second = stream.getvalue().splitlines()
    assert first == f"tool bash: {'x' * 120} (exit 0)"
    assert second == "tool bash: echo \\x1b[31mred\\x0anext (exit 0)"


def test_the_summary_is_escaped_too() -> None:
    sink, stream = _sink()

    _call(sink, "read a", "bad \x1b[2J")

    assert stream.getvalue() == "tool read: a (bad \\x1b[2J)\n"


def test_notices_are_prefixed_by_their_level_and_escaped() -> None:
    sink, stream = _sink()

    sink(Notice("warn", "careful\x1b[0m"))
    sink(Notice("error", "boom"))
    sink(Notice("info", "fyi"))

    assert stream.getvalue() == "warning: careful\\x1b[0m\nerror: boom\ninfo: fyi\n"


def test_quiet_keeps_only_warnings_and_errors() -> None:
    sink, stream = _sink(quiet=True)

    _call(sink, "read a", "1 line")
    sink(Notice("info", "fyi"))
    sink(Compacting(True))
    sink(Notice("warn", "w"))
    sink(Notice("error", "e"))

    assert stream.getvalue() == "warning: w\nerror: e\n"


def test_compaction_is_one_line_when_it_starts() -> None:
    sink, stream = _sink()

    sink(Compacting(True))
    sink(Compacting(False))

    assert stream.getvalue() == "compacting the context\n"


def test_text_task_progress_and_turn_end_write_nothing() -> None:
    sink, stream = _sink()

    sink(AssistantText("hello"))
    sink(TaskProgress("1", Notice("warn", "child")))
    sink(TaskProgress("1", ToolStarted("2", "read a")))
    sink(TurnEnded("done"))

    assert stream.getvalue() == ""


def test_there_are_no_colour_codes_without_colour_and_some_with_it() -> None:
    plain, plain_stream = _sink(color=False)
    colored, colored_stream = _sink(color=True)

    for sink in (plain, colored):
        sink(Notice("warn", "w"))

    assert "\x1b" not in plain_stream.getvalue()
    assert colored_stream.getvalue().startswith("\x1b[")
    assert colored_stream.getvalue().endswith("w\n")


async def test_the_headless_approver_denies_everything_with_the_note() -> None:
    request = ApprovalRequest("bash", "bash ls", None, True, save_rule="bash(ls)")

    decision = await HeadlessApprover().ask(request)

    assert decision.choice is Choice.DENY
    assert decision.note == "no approval in headless mode (use --allow)"
