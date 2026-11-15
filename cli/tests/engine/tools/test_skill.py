"""The skill tool: body under a header, unknown names listed, output cut to the limit."""

from __future__ import annotations

from pathlib import Path

from src.engine.skills.loader import Skill, Skills
from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.skill import SkillTool
from src.engine.tools.text import TOOL_OUTPUT_LIMIT


def skill(name: str, body: str = "body\n", source: str = "project") -> Skill:
    return Skill(name, "d", body, source, Path("/abs/.agent/skills") / name)  # ty: ignore[invalid-argument-type]


def tool(*items: Skill) -> SkillTool:
    return SkillTool(Skills({s.name: s for s in items}, ()))


async def run(t: SkillTool, ctx: ToolContext, **args: object) -> ToolResult:
    prepared = await t.prepare(args, ctx)
    if isinstance(prepared, ToolResult):
        return prepared
    assert isinstance(prepared, Prepared)
    return await prepared.execute()


def test_tool_needs_no_approval_is_not_mutating_and_keys_on_name() -> None:
    t = tool(skill("a"))
    assert t.name == "skill" and not t.needs_approval and not t.mutating
    assert t.grant_arg == "name" and t.parameters["required"] == ["name"]


async def test_ac10_returns_header_and_exact_body(ctx: ToolContext) -> None:
    result = await run(tool(skill("release", "1. step\n---\n2. more\n")), ctx, name="release")
    assert not result.is_error and result.summary == "skill release"
    assert result.output == (
        "Skill 'release' (project) from /abs/.agent/skills/release\n"
        "[instructions read from a file; bundled files are relative to that folder]\n\n"
        "1. step\n---\n2. more\n"
    )


async def test_empty_body_is_noted(ctx: ToolContext) -> None:
    result = await run(tool(skill("e", "")), ctx, name="e")
    assert result.output.endswith("\n\n(empty body)")


async def test_header_name_and_folder_are_escaped(ctx: ToolContext) -> None:
    s = Skill("x", "d", "b", "global", Path("/a\x1b[31m/x"))
    result = await run(tool(s), ctx, name="x")
    assert "\x1b" not in result.output.splitlines()[0]


async def test_ac10_big_body_is_cut_to_the_limit(ctx: ToolContext) -> None:
    result = await run(tool(skill("big", "x" * 200_000, "global")), ctx, name="big")
    assert len(result.output) == TOOL_OUTPUT_LIMIT and result.output.endswith("\n[truncated]")
    assert not result.is_error


async def test_ac11_unknown_name_lists_valid_names(ctx: ToolContext) -> None:
    result = await run(tool(skill("review"), skill("release")), ctx, name="nope")
    assert result.is_error and result.summary == "unknown skill"
    assert result.output == "Unknown skill 'nope'. Available: release, review"


async def test_unknown_name_is_escaped_and_cut(ctx: ToolContext) -> None:
    result = await run(tool(skill("a")), ctx, name="\x1b" + "y" * 100_000)
    assert "\x1b" not in result.output and len(result.output) <= TOOL_OUTPUT_LIMIT


async def test_missing_argument_is_invalid(ctx: ToolContext) -> None:
    result = await run(tool(skill("a")), ctx)
    assert result.is_error and result.summary == "invalid arguments"


async def test_a_non_ascii_folder_path_is_readable_in_the_header(ctx: ToolContext) -> None:
    item = Skill("ru", "d", "тело", "global", Path("/Users/Роман/.config/agent/skills/ru"))
    result = await run(tool(item), ctx, name="ru")
    assert result.output.startswith("Skill 'ru' (global) from /Users/Роман/.config/agent/skills/ru\n")


async def test_a_bidi_override_in_the_folder_path_is_still_written_out(ctx: ToolContext) -> None:
    item = Skill("x", "d", "b", "project", Path("/w/\u202eevil/x"))
    result = await run(tool(item), ctx, name="x")
    assert "\u202e" not in result.output and "/w/\\u202eevil/x" in result.output
