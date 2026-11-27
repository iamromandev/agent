"""The todo module: validation, derivation from the history, the note, and the tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from core.engine.provider.base import Message, ToolCall
from core.engine.todo import (
    MAX_CONTENT,
    MAX_ITEMS,
    Status,
    TodoItem,
    active,
    current_todos,
    parse_items,
)
from core.engine.tools.base import ToolContext, ToolResult
from core.engine.tools.todo import TodoTool


def item(content: str = "x", status: str = "pending") -> dict[str, str]:
    return {"content": content, "status": status}


def items_of(*pairs: tuple[str, Status]) -> tuple[TodoItem, ...]:
    return tuple(TodoItem(c, s) for c, s in pairs)


@pytest.mark.parametrize(
    "args, problem",
    [
        ({"items": [item()] * (MAX_ITEMS + 1)}, "at most 30"),
        ({"items": [item("   ")]}, "content"),
        ({"items": [item("a" * (MAX_CONTENT + 1))]}, "200"),
        ({"items": "nope"}, "list"),
        ({"items": ["nope"]}, "object"),
        ({}, "items"),
    ],
)
def test_parse_items_rejects_with_a_message(args: dict[str, Any], problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_items(args)


def test_active_means_non_empty_and_not_all_done() -> None:
    assert not active(())
    assert not active(items_of(("a", "done"), ("b", "done")))
    assert active(items_of(("a", "done"), ("b", "pending")))


def call_message(call_id: str, args: dict[str, Any], name: str = "todo") -> Message:
    return Message("assistant", "", [ToolCall(call_id, name, args)])


def result_message(call_id: str) -> Message:
    return Message("tool", "ok", tool_call_id=call_id, name="todo")


def test_current_todos_ignores_an_invalid_call() -> None:
    history = [
        call_message("1", {"items": [item("first")]}),
        result_message("1"),
        call_message("2", {"items": [item("bad", "doing")]}),
        result_message("2"),
    ]
    assert current_todos(history, ()) == (TodoItem("first", "pending"),)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path, bash_timeout=1.0)


async def run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    result = await TodoTool().prepare(args, ctx)
    assert isinstance(result, ToolResult)
    return result


async def test_tool_error_names_the_problem(ctx: ToolContext) -> None:
    result = await run({"items": [item("a", "doing")]}, ctx)
    assert result.is_error and "status" in result.output and result.summary == "invalid list"
