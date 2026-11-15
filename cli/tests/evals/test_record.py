"""The event sink fills a RunRecord: steps, tokens (children included), tools, final text, errors."""

from __future__ import annotations

from evals.record import RunRecord, sink
from src.engine.events import (
    AssistantText,
    Notice,
    TaskProgress,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)
from src.engine.tools.base import ToolResult


def test_steps_and_tokens_count_only_real_usage_counts() -> None:
    record = RunRecord("t", 1)
    emit = sink(record)
    emit(UsageUpdate(100, 1000))
    emit(UsageUpdate(50, 1000, estimated=True))
    emit(UsageUpdate(200, 1000))
    assert (record.steps, record.tokens) == (2, 300)


def test_child_usage_adds_tokens_but_not_steps() -> None:
    record = RunRecord("t", 1)
    emit = sink(record)
    emit(TaskProgress("c1", UsageUpdate(40, 1000)))
    emit(UsageUpdate(10, 1000))
    assert (record.steps, record.tokens) == (1, 50)


def test_tools_are_counted_by_name_and_results_kept() -> None:
    record = RunRecord("t", 1)
    emit = sink(record)
    emit(ToolStarted("1", "bash python -m pytest -q"))
    emit(ToolFinished("1", ToolResult("3 passed"), None, True))
    emit(ToolStarted("2", "task look around", task=True))
    emit(TaskProgress("2", ToolStarted("x", "grep foo")))
    assert record.tools == {"bash": 1, "task": 1}
    assert record.tool_results == [("bash", "bash python -m pytest -q", "3 passed")]


def test_final_text_is_the_text_of_the_last_step() -> None:
    record = RunRecord("t", 1)
    emit = sink(record)
    emit(AssistantText("thinking"))
    emit(UsageUpdate(10, 1000))
    emit(AssistantText("the "))
    emit(AssistantText("answer"))
    emit(UsageUpdate(10, 1000))
    emit(TurnEnded("done"))
    assert record.final_text == "the answer"


def test_error_notice_and_turn_status_are_recorded() -> None:
    record = RunRecord("t", 1)
    emit = sink(record)
    emit(Notice("error", "boom"))
    emit(TurnEnded("error"))
    assert record.error == "boom"
    assert record.ended == "error"


def test_step_limit_is_recorded_but_is_not_an_error() -> None:
    record = RunRecord("t", 1)
    sink(record)(TurnEnded("step_limit"))
    assert record.ended == "step_limit"
    assert record.error is None


def test_error_without_notice_gets_a_default_reason() -> None:
    record = RunRecord("t", 1)
    sink(record)(TurnEnded("error"))
    assert record.error == "turn ended with an error"


def test_tokens_are_unknown_until_a_usage_count_arrives() -> None:
    record = RunRecord("t", 1)
    assert record.tokens is None
    sink(record)(UsageUpdate(10, 1000))
    assert record.tokens == 10


def test_refused_tool_results_are_recorded_by_summary() -> None:
    record = RunRecord("t", 1)
    emit = sink(record)
    emit(ToolStarted("1", "bash rm x"))
    emit(ToolFinished("1", ToolResult("denied by rule x", True, "denied by rule"), None, True))
    emit(ToolStarted("2", "bash ls"))
    emit(ToolFinished("2", ToolResult("ok", False, "exit 0"), None, True))
    assert record.refusals == [("bash rm x", "denied by rule")]
