from __future__ import annotations

from pathlib import Path
from typing import Any

from core.engine.mcp.contract import CallResult, McpError, ToolInfo
from core.engine.mcp.tool import McpTool
from core.engine.tools.base import Prepared, ToolContext, ToolResult
from core.engine.tools.text import TOOL_OUTPUT_LIMIT

SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "times": {"type": "integer"}},
    "required": ["text"],
}
CTX = ToolContext(Path("/ws"), 5.0)


class FakeCaller:
    def __init__(self, outcome: CallResult | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> CallResult:
        self.calls.append((server, tool, args))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def make(outcome: CallResult | Exception | None = None) -> tuple[McpTool, FakeCaller]:
    caller = FakeCaller(outcome if outcome is not None else CallResult(["hi"], False))
    return McpTool(caller, "fake", ToolInfo("echo", "Echo it", SCHEMA), "mcp__fake__echo"), caller


async def run(tool: McpTool, args: dict[str, Any]) -> ToolResult:
    prepared = await tool.prepare(args, CTX)
    assert isinstance(prepared, Prepared)
    return await prepared.execute()


async def test_wrong_arguments_are_refused_without_a_preview_or_a_call() -> None:
    tool, caller = make()
    for args in ({}, {"text": 3}, {"text": "x", "times": "two"}, {"__invalid__": "{"}):
        result = await tool.prepare(args, CTX)
        assert isinstance(result, ToolResult)
        assert (result.is_error, result.summary) == (True, "invalid arguments")
    assert caller.calls == []


async def test_a_result_without_text_says_so() -> None:
    tool, _ = make(CallResult([], False))
    assert (await run(tool, {"text": "x"})).output == "(no output)"


async def test_long_output_is_cut_to_the_limit_with_a_note() -> None:
    tool, _ = make(CallResult(["y" * 50_000], False))
    result = await run(tool, {"text": "x"})
    assert len(result.output) <= TOOL_OUTPUT_LIMIT
    assert result.output.endswith("chars]")
    assert "[truncated: showing " in result.output
    assert "of 50000 chars]" in result.output
    assert result.output.startswith("yyyy")


async def test_any_other_mcp_error_becomes_an_error_result() -> None:
    tool, _ = make(McpError("bad request"))
    result = await run(tool, {"text": "x"})
    assert (result.output, result.is_error, result.summary) == ("bad request", True, "error")
