from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.mcp.contract import CallResult, McpError, McpExited, McpTimeout, ToolInfo
from src.agent.mcp.tool import McpTool
from src.agent.tools.base import Prepared, ToolContext, ToolResult
from src.agent.tools.text import TOOL_OUTPUT_LIMIT

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


def test_it_offers_the_servers_description_and_schema_under_the_cleaned_name() -> None:
    tool, _ = make()
    spec = tool.spec()
    assert (spec.name, spec.description, spec.parameters) == ("mcp__fake__echo", "Echo it", SCHEMA)


def test_it_needs_approval_and_is_hidden_in_plan_mode() -> None:
    tool, _ = make()
    assert tool.needs_approval is True
    assert tool.mutating is True


def test_an_always_grant_covers_the_tool_name_not_an_argument() -> None:
    tool, _ = make()
    assert tool.grant_arg is None


def test_the_label_is_the_name_and_the_first_argument() -> None:
    tool, _ = make()
    assert tool.label({"text": "hi"}) == "mcp__fake__echo hi"


async def test_the_preview_shows_the_name_and_the_arguments_as_pretty_json() -> None:
    tool, caller = make()
    prepared = await tool.prepare({"text": "héllo", "times": 2}, CTX)
    assert isinstance(prepared, Prepared)
    assert prepared.preview is not None
    assert (prepared.preview.kind, prepared.preview.title) == ("content", "mcp__fake__echo")
    assert prepared.preview.body == json.dumps({"text": "héllo", "times": 2}, indent=2, ensure_ascii=False)
    assert caller.calls == []  # nothing runs before approval


async def test_wrong_arguments_are_refused_without_a_preview_or_a_call() -> None:
    tool, caller = make()
    for args in ({}, {"text": 3}, {"text": "x", "times": "two"}, {"__invalid__": "{"}):
        result = await tool.prepare(args, CTX)
        assert isinstance(result, ToolResult)
        assert (result.is_error, result.summary) == (True, "invalid arguments")
    assert caller.calls == []


async def test_execute_calls_the_server_tool_by_its_own_name() -> None:
    tool, caller = make()
    result = await run(tool, {"text": "hi"})
    assert caller.calls == [("fake", "echo", {"text": "hi"})]
    assert result == ToolResult("hi", False, "ok")


async def test_text_parts_are_joined_with_newlines() -> None:
    tool, _ = make(CallResult(["first", "[image omitted]", "second"], False))
    result = await run(tool, {"text": "x"})
    assert result.output == "first\n[image omitted]\nsecond"


async def test_an_error_result_reaches_the_model_as_an_error() -> None:
    tool, _ = make(CallResult(["it broke"], True))
    result = await run(tool, {"text": "x"})
    assert (result.output, result.is_error, result.summary) == ("it broke", True, "error")


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


async def test_output_exactly_at_the_limit_is_kept_whole() -> None:
    tool, _ = make(CallResult(["z" * TOOL_OUTPUT_LIMIT], False))
    assert (await run(tool, {"text": "x"})).output == "z" * TOOL_OUTPUT_LIMIT


async def test_a_timeout_becomes_an_error_result_with_a_timeout_summary() -> None:
    tool, _ = make(McpTimeout("timeout after 60s"))
    result = await run(tool, {"text": "x"})
    assert (result.output, result.is_error, result.summary) == ("timeout after 60s", True, "timeout")


async def test_an_exited_server_becomes_an_error_result() -> None:
    tool, _ = make(McpExited("the server exited"))
    result = await run(tool, {"text": "x"})
    assert (result.output, result.is_error, result.summary) == ("the server exited", True, "server exited")


async def test_any_other_mcp_error_becomes_an_error_result() -> None:
    tool, _ = make(McpError("bad request"))
    result = await run(tool, {"text": "x"})
    assert (result.output, result.is_error, result.summary) == ("bad request", True, "error")
