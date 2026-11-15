"""The `todo` tool. The list itself lives in the history (see `src.agent.todo`): a call and its result change it."""

from __future__ import annotations

from typing import Any, ClassVar

from src.agent.todo import STATUSES, parse_items, summary
from src.agent.tools.base import Prepared, Tool, ToolContext, ToolResult

MANY_IN_PROGRESS = " Note: keep one item in_progress at a time."


class TodoTool(Tool):
    name: ClassVar[str] = "todo"
    description: ClassVar[str] = (
        "Keep a checklist for a task with several steps. Each call replaces the whole list: send every item, "
        "with its status (pending, in_progress or done). Keep one item in_progress; mark items done when finished. "
        "An empty list clears it."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "The whole list, in order.",
                "items": {
                    "type": "object",
                    "required": ["content", "status"],
                    "properties": {
                        "content": {"type": "string", "description": "What to do, up to 200 characters."},
                        "status": {"type": "string", "enum": list(STATUSES)},
                    },
                },
            }
        },
        "required": ["items"],
    }

    def label(self, args: dict[str, Any]) -> str:
        return "todo"

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        try:
            items = parse_items(args)
        except ValueError as exc:
            return ToolResult(str(exc), True, "invalid list")
        note = MANY_IN_PROGRESS if sum(i.status == "in_progress" for i in items) > 1 else ""
        return ToolResult(f"Updated: {summary(items)}{note}", False, summary(items))
