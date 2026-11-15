"""The todo list the model keeps with the `todo` tool: derived from the history, nothing is stored apart from it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from src.agent.provider.base import Message

Status = Literal["pending", "in_progress", "done"]
STATUSES: tuple[str, ...] = get_args(Status)
MAX_ITEMS = 30
MAX_CONTENT = 200
NOTE_CONTENT = 60  # characters of an item kept in the note: 30 items stay under 2,500 characters in all
MARK: dict[Status, str] = {"pending": "○", "in_progress": "▸", "done": "✓"}
# What the loop writes for a call that never ran; a `todo` call with one of these as its result changed nothing.
INTERRUPTED = "Interrupted by user"
SKIPPED = "Skipped: the turn was interrupted before this call ran"
NOT_COMPLETED = "Not completed: the turn ended with an internal error"
PLACEHOLDERS = frozenset({INTERRUPTED, SKIPPED, NOT_COMPLETED})
NOTE_HEADER = "Current todo list (update it with the todo tool as you work):"


@dataclass(frozen=True)
class TodoItem:
    content: str
    status: Status


def parse_items(args: Mapping[str, Any]) -> tuple[TodoItem, ...]:
    """The items of a call. Raises ValueError with a message for the model when anything is wrong."""
    raw = args.get("items")
    if raw is None:
        raise ValueError('missing "items"')
    if not isinstance(raw, list):
        raise ValueError('"items" must be a list')
    if len(raw) > MAX_ITEMS:
        raise ValueError(f"at most {MAX_ITEMS} items, got {len(raw)}")
    items: list[TodoItem] = []
    for number, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"item {number} must be an object with content and status")
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"item {number}: content must be a non-empty string")
        content = content.strip()
        if len(content) > MAX_CONTENT:
            raise ValueError(f"item {number}: content is {len(content)} characters, the limit is {MAX_CONTENT}")
        status = entry.get("status")
        if status not in STATUSES:
            raise ValueError(f"item {number}: status must be one of {', '.join(STATUSES)}")
        items.append(TodoItem(content, cast(Status, status)))  # checked against STATUSES above
    return tuple(items)


def active(items: Sequence[TodoItem]) -> bool:
    """A list worth showing and reminding of: not empty and not all done."""
    return any(i.status != "done" for i in items)


def summary(items: Sequence[TodoItem]) -> str:
    return f"{sum(i.status == 'done' for i in items)}/{len(items)} done"


def render_note(items: Sequence[TodoItem]) -> str:
    lines = [NOTE_HEADER]
    for i in items:
        flat = "".join(" " if c.isspace() else c for c in i.content if c.isspace() or c.isprintable())
        flat = " ".join(flat.split())
        text = flat if len(flat) <= NOTE_CONTENT else flat[: NOTE_CONTENT - 1] + "…"
        lines.append(f"- [{i.status}] {text}")
    return "\n".join(lines)


def current_todos(history: Sequence[Message], base: tuple[TodoItem, ...]) -> tuple[TodoItem, ...]:
    """The list as the history leaves it: the last valid `todo` call that has a real result (not a placeholder for a call that never ran), else `base`."""
    answered = {m.tool_call_id for m in history if m.role == "tool" and m.tool_call_id and m.content not in PLACEHOLDERS}
    found = base
    for message in history:
        if message.role != "assistant":
            continue
        for call in message.tool_calls:
            if call.name != "todo" or call.id not in answered:
                continue
            try:
                found = parse_items(call.arguments)
            except ValueError:
                continue
    return found
