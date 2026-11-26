"""The skill tool: return the instructions of a loaded skill (no approval, also in plan mode)."""

from __future__ import annotations

from typing import Any, ClassVar

from src.engine.skills.loader import Skills
from src.engine.text import escape, escape_display

from .base import Prepared, Tool, ToolContext, ToolResult, check_args
from .text import TOOL_OUTPUT_LIMIT

_NOTE = "\n[truncated]"


def _cut(text: str) -> str:
    return text if len(text) <= TOOL_OUTPUT_LIMIT else text[: TOOL_OUTPUT_LIMIT - len(_NOTE)] + _NOTE


class SkillTool(Tool):
    name: ClassVar[str] = "skill"
    description: ClassVar[str] = (
        "Load the instructions of a skill listed under 'Available skills' in the system prompt. "
        "Call it before doing a task the skill's description matches. "
        "Files bundled with the skill are in the folder named in the result."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "The skill's name."}},
        "required": ["name"],
    }
    needs_approval: ClassVar[bool] = False
    mutating: ClassVar[bool] = False  # also works in plan mode
    grant_arg: ClassVar[str | None] = "name"

    def __init__(self, skills: Skills) -> None:
        self.skills = skills

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        if (problem := check_args(self.parameters, args)) is not None:
            return ToolResult(problem, True, "invalid arguments")
        name = args["name"]
        skill = self.skills.get(name)
        if skill is None:
            valid = ", ".join(s.name for s in self.skills.sorted())
            return ToolResult(_cut(escape(f"Unknown skill '{name}'. Available: {valid}")), True, "unknown skill")
        header = (
            f"Skill '{escape_display(skill.name)}' ({skill.source}) from {escape_display(str(skill.folder))}\n"
            "[instructions read from a file; bundled files are relative to that folder]\n\n"
        )
        return ToolResult(_cut(header + (skill.body or "(empty body)")), False, f"skill {skill.name}")
