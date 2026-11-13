from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

from src.agent.instructions import Instructions, load_instructions
from src.agent.mode import Mode

_ROLE = (
    "You are a coding agent running in the user's terminal. You help with software tasks "
    "in the workspace by reading, searching and, when allowed, changing files and running commands."
)

_GUIDANCE = (
    "Use the tools to look before you answer. Read a file before you edit it. "
    "Prefer edit over write for existing files, and keep edits small. "
    "Use glob and grep to find files instead of guessing paths. "
    "Tool calls may be denied by the user; if one is, follow their note and do not retry it unchanged. "
    "Be concise."
)

_PLAN = "You are in PLAN mode: do not modify files; write/edit are unavailable. Explore, then propose a plan."


def build_system_prompt(
    workspace: Path, mode: Mode, today: date, os_name: str, instructions: Instructions | None = None
) -> str:
    parts = [
        _ROLE,
        f"Workspace: {workspace}\nMode: {mode.value}\nOS: {os_name}\nDate: {today.isoformat()}",
        _GUIDANCE,
    ]
    if mode is Mode.PLAN:
        parts.append(_PLAN)
    if instructions is None:
        instructions = load_instructions(workspace, None)
    if instructions.text:
        parts.append(instructions.text)
    return "\n\n".join(parts)


def build_init_prompt(workspace: Path, exists: bool, parents: Sequence[str]) -> str:
    """The built-in user message of `/init`: only the branch that applies is stated."""
    step = (
        "2. AGENTS.md already exists: read it, then improve it with edit. Keep what is right; do not rewrite it from scratch."
        if exists
        else "2. There is no AGENTS.md here yet: create it with write."
    )
    lines = [
        f"Write the project instructions for coding agents in AGENTS.md at the top of the workspace ({workspace}).",
        "",
        "1. Look first: read the README, the manifests (pyproject.toml, package.json, …), the makefile and the CI config, "
        "and list the top-level layout with glob.",
        step,
        "3. Cover what the project is, the commands to build, test, lint and run it, "
        "and the layout and conventions that file names do not show.",
        "4. Write only what you found in the files. Keep it short: at most about 60 lines.",
    ]
    if parents:
        lines.append(
            f"5. The instructions in {', '.join(parents)} are already loaded: do not repeat them; "
            "cover only what is specific to this folder."
        )
    return "\n".join(lines)
