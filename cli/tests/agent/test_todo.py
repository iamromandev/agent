"""The todo module: validation, derivation from the history, the note, and the tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from src.agent.provider.base import Message, ToolCall
from src.agent.todo import (
    MAX_CONTENT,
    MAX_ITEMS,
    Status,
    TodoItem,
    active,
    current_todos,
    parse_items,
    render_note,
    summary,
)
from src.agent.tools.base import ToolContext, ToolResult
from src.agent.tools.todo import TodoTool


def item(content: str = "x", status: str = "pending") -> dict[str, str]:
    return {"content": content, "status": status}


def items_of(*pairs: tuple[str, Status]) -> tuple[TodoItem, ...]:
    return tuple(TodoItem(c, s) for c, s in pairs)


@pytest.mark.parametrize(
    "args, problem",
    [
        ({"items": [item()] * (MAX_ITEMS + 1)}, "at most 30"),
        ({"items": [item("")]}, "content"),
        ({"items": [item("   ")]}, "content"),
        ({"items": [item("a" * (MAX_CONTENT + 1))]}, "200"),
        ({"items": [item("a", "doing")]}, "status"),
        ({"items": "nope"}, "list"),
        ({"items": ["nope"]}, "object"),
        ({"items": [{"content": "a"}]}, "status"),
        ({"items": [{"status": "done"}]}, "content"),
        ({"items": [{"content": 3, "status": "done"}]}, "content"),
        ({}, "items"),
    ],
)
def test_parse_items_rejects_with_a_message(args: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_items(args)


def test_parse_items_strips_content_and_accepts_limits() -> None:
    assert parse_items({"items": [item("  a  ", "done")]}) == (TodoItem("a", "done"),)
    assert len(parse_items({"items": [item("a" * MAX_CONTENT)] * MAX_ITEMS})) == MAX_ITEMS
    assert parse_items({"items": []}) == ()


def test_active_means_non_empty_and_not_all_done() -> None:
    assert not active(())
    assert not active(items_of(("a", "done"), ("b", "done")))
    assert active(items_of(("a", "done"), ("b", "pending")))


def test_summary_counts_done() -> None:
    assert summary(items_of(("a", "done"), ("b", "pending"), ("c", "pending"))) == "1/3 done"


def test_render_note_lists_each_item_with_its_status() -> None:
    note = render_note(items_of(("Find uses", "in_progress"), ("Rename", "pending")))
    assert note.splitlines() == [
        "Current todo list (update it with the todo tool as you work):",
        "- [in_progress] Find uses",
        "- [pending] Rename",
    ]


def test_note_for_the_largest_list_stays_under_2500_characters() -> None:
    # 30 x 200 characters of text is 6,000 on its own, so the note shortens long items (the panel and
    # the saved call keep the full text).
    big = tuple(TodoItem("a" * MAX_CONTENT, "in_progress") for _ in range(MAX_ITEMS))
    assert len(render_note(big)) < 2500


def test_note_shortens_a_long_item_with_an_ellipsis() -> None:
    line = render_note(items_of(("a" * MAX_CONTENT, "pending"))).splitlines()[1]
    assert line.endswith("\u2026") and len(line) < MAX_CONTENT


def test_note_for_a_typical_list_stays_under_400_characters() -> None:
    typical = items_of(
        ("Find all uses of num_ctx", "in_progress"),
        ("Rename in src", "pending"),
        ("Update tests", "pending"),
        ("Update README and AGENTS.md", "pending"),
        ("Run the checks", "pending"),
    )
    assert len(render_note(typical)) < 400


def call_message(call_id: str, args: dict[str, Any], name: str = "todo") -> Message:
    return Message("assistant", "", [ToolCall(call_id, name, args)])


def result_message(call_id: str) -> Message:
    return Message("tool", "ok", tool_call_id=call_id, name="todo")


def test_current_todos_is_the_base_without_calls() -> None:
    base = items_of(("a", "pending"))
    assert current_todos([Message("user", "hi")], base) == base


def test_current_todos_takes_the_last_valid_call_with_a_result() -> None:
    history = [
        Message("user", "go"),
        call_message("1", {"items": [item("first")]}),
        result_message("1"),
        call_message("2", {"items": [item("second", "in_progress")]}),
        result_message("2"),
    ]
    assert current_todos(history, ()) == (TodoItem("second", "in_progress"),)


def test_current_todos_ignores_a_call_without_a_result() -> None:
    history = [
        call_message("1", {"items": [item("first")]}),
        result_message("1"),
        call_message("2", {"items": [item("second")]}),
    ]
    assert current_todos(history, ()) == (TodoItem("first", "pending"),)


def test_current_todos_ignores_an_invalid_call() -> None:
    history = [
        call_message("1", {"items": [item("first")]}),
        result_message("1"),
        call_message("2", {"items": [item("bad", "doing")]}),
        result_message("2"),
    ]
    assert current_todos(history, ()) == (TodoItem("first", "pending"),)


def test_current_todos_empty_call_clears_the_base() -> None:
    history = [call_message("1", {"items": []}), result_message("1")]
    assert current_todos(history, items_of(("old", "pending"))) == ()


def test_current_todos_ignores_other_tools() -> None:
    history = [call_message("1", {"items": [item()]}, name="read"), result_message("1")]
    assert current_todos(history, ()) == ()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path, bash_timeout=1.0)


async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    result = await TodoTool().prepare(args, ctx)
    assert isinstance(result, ToolResult)
    return result


def test_tool_needs_no_approval_and_is_allowed_in_plan_mode() -> None:
    tool = TodoTool()
    assert tool.name == "todo"
    assert not tool.needs_approval and not tool.mutating
    assert tool.parameters["required"] == ["items"]
    assert tool.label({"items": []}) == "todo"


async def test_tool_reports_done_over_total(ctx: ToolContext) -> None:
    items = [item("a", "done"), item("b", "done"), item("c"), item("d"), item("e", "in_progress")]
    result = await run({"items": items}, ctx)
    assert (result.output, result.summary, result.is_error) == ("Updated: 2/5 done", "2/5 done", False)


async def test_tool_notes_more_than_one_in_progress(ctx: ToolContext) -> None:
    result = await run({"items": [item("a", "in_progress"), item("b", "in_progress")]}, ctx)
    assert not result.is_error
    assert result.output == "Updated: 0/2 done Note: keep one item in_progress at a time."


async def test_tool_error_names_the_problem(ctx: ToolContext) -> None:
    result = await run({"items": [item("a", "doing")]}, ctx)
    assert result.is_error and "status" in result.output and result.summary == "invalid list"


def test_current_todos_ignores_a_call_whose_result_is_a_placeholder() -> None:
    from src.agent.todo import INTERRUPTED, NOT_COMPLETED, SKIPPED

    base = items_of(("a", "pending"))
    for text in (INTERRUPTED, SKIPPED, NOT_COMPLETED):
        history = [
            call_message("t1", {"items": [item("b")]}),
            Message("tool", text, tool_call_id="t1", name="todo"),
        ]
        assert current_todos(history, base) == base


def test_render_note_collapses_whitespace_and_drops_control_characters() -> None:
    note = render_note((TodoItem("a\nb\r\n- [done] fake\x1b[2J\x00 c", "pending"),))

    assert note.count("\n") == 1
    assert "\x1b" not in note and "\x00" not in note and "\r" not in note
    assert note.splitlines()[1] == "- [pending] a b - [done] fake[2J c"
