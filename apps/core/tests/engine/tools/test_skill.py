"""The skill tool: body under a header, unknown names listed, output cut to the limit."""

from __future__ import annotations

from pathlib import Path

from core.engine.skills.loader import Skill, Skills
from core.engine.tools.base import Prepared, ToolContext, ToolResult
from core.engine.tools.skill import SkillTool


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


async def test_header_name_and_folder_are_escaped(ctx: ToolContext) -> None:
    s = Skill("x", "d", "b", "global", Path("/a\x1b[31m/x"))
    result = await run(tool(s), ctx, name="x")
    assert "\x1b" not in result.output.splitlines()[0]


async def test_ac11_unknown_name_lists_valid_names(ctx: ToolContext) -> None:
    result = await run(tool(skill("review"), skill("release")), ctx, name="nope")
    assert result.is_error and result.summary == "unknown skill"
    assert result.output == "Unknown skill 'nope'. Available: release, review"


async def test_missing_argument_is_invalid(ctx: ToolContext) -> None:
    result = await run(tool(skill("a")), ctx)
    assert result.is_error and result.summary == "invalid arguments"


async def test_header_names_a_builtin_skill_and_its_folder(ctx: ToolContext, tmp_path: Path) -> None:
    from core.engine.skills.loader import load_skills

    folder = tmp_path / "b" / "debug"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\ndescription: d\n---\nsteps\n")
    skills = load_skills(tmp_path / "ws", {"XDG_CONFIG_HOME": str(tmp_path / "cfg")}, builtin=tmp_path / "b")
    result = await run(SkillTool(skills), ctx, name="debug")
    assert result.output.splitlines()[0] == f"Skill 'debug' (builtin) from {folder.resolve()}"
