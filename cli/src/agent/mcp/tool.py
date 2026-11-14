"""One tool of an MCP server, as the model sees it: every call needs approval and is refused in plan mode."""

from __future__ import annotations

import json
from typing import Any, ClassVar, Protocol

from src.agent.mcp.contract import CallResult, McpError, ToolInfo
from src.agent.tools.base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from src.agent.tools.text import TOOL_OUTPUT_LIMIT


class Caller(Protocol):
    async def call(self, server: str, tool: str, args: dict[str, Any]) -> CallResult: ...


def _note(shown: int, total: int) -> str:
    return f"\n[truncated: showing {shown} of {total} chars]"


def _join(parts: list[str]) -> str:
    text = "\n".join(parts)
    if not text:
        return "(no output)"
    total = len(text)
    if total <= TOOL_OUTPUT_LIMIT:
        return text
    room = TOOL_OUTPUT_LIMIT - len(_note(TOOL_OUTPUT_LIMIT, total))
    return text[:room] + _note(room, total)


class McpTool(Tool):
    # The name, description and schema come from the server, so they are set per instance.
    name: str  # ty: ignore[invalid-attribute-override]
    description: str  # ty: ignore[invalid-attribute-override]
    parameters: dict[str, Any]  # ty: ignore[invalid-attribute-override]
    needs_approval: ClassVar[bool] = True  # nothing a server offers is trusted to run unasked
    mutating: ClassVar[bool] = True  # unknown effects: hidden and refused in plan mode, like write and edit
    uncancellable: ClassVar[bool | None] = False  # but a call to a server can be interrupted: Esc must not wait for it
    grant_arg: ClassVar[str | None] = None  # "Always" covers this tool (by name), not one argument value

    def __init__(self, caller: Caller, server: str, info: ToolInfo, name: str) -> None:
        self._caller = caller
        self._server = server
        self._tool = info.name
        self.name = name
        self.description = info.description
        self.parameters = info.input_schema

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if (problem := check_args(self.parameters, args)) is not None:
            return ToolResult(f"{self.name}: {problem}", True, "invalid arguments")

        async def execute() -> ToolResult:
            try:
                result = await self._caller.call(self._server, self._tool, args)
            except McpError as exc:
                return ToolResult(str(exc), True, exc.summary)
            return ToolResult(_join(result.parts), result.is_error, "error" if result.is_error else "ok")

        return Prepared(Preview("content", self.name, json.dumps(args, indent=2, ensure_ascii=False)), execute)
