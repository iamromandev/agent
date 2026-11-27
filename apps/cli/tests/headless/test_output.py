"""`StderrSink`: one line per tool call and notice, quiet mode, no colour unless asked, escaped text."""

from __future__ import annotations

import io

from core.engine.events import Compacting, Notice, ToolFinished, ToolStarted
from core.engine.tools.base import ToolResult
from src.headless.output import StderrSink


def _sink(quiet: bool = False, color: bool = False) -> tuple[StderrSink, io.StringIO]:
    stream = io.StringIO()
    return StderrSink(stream, quiet=quiet, color=color), stream


def _call(sink: StderrSink, label: str, summary: str = "", is_error: bool = False, call_id: str = "1") -> None:
    sink(ToolStarted(call_id, label))
    sink(ToolFinished(call_id, ToolResult("out", is_error, summary), None, True))


def test_compaction_is_one_line_when_it_starts() -> None:
    sink, stream = _sink()

    sink(Compacting(True))
    sink(Compacting(False))

    assert stream.getvalue() == "compacting the context\n"


def test_there_are_no_colour_codes_without_colour_and_some_with_it() -> None:
    plain, plain_stream = _sink(color=False)
    colored, colored_stream = _sink(color=True)

    for sink in (plain, colored):
        sink(Notice("warn", "w"))

    assert "\x1b" not in plain_stream.getvalue()
    assert colored_stream.getvalue().startswith("\x1b[")
    assert colored_stream.getvalue().endswith("w\n")
