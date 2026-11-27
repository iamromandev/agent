from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

from core.engine.guardrails.paths import ProtectedPaths
from core.engine.instructions import Instructions, load_instructions
from core.engine.mode import Mode

_ROLE = (
    "You are a coding agent running in the user's terminal. You help with software tasks "
    "in the workspace by reading, searching and, when allowed, changing files and running commands."
)

_GUIDANCE = (
    "Use the tools to look before you answer. Read a file before you edit it. "
    "Prefer edit over write for existing files, and keep edits small. "
    "Use glob and grep to find files instead of guessing paths. "
    "Tool calls may be denied by the user; if one is, follow their note and do not retry it unchanged. "
    "Use term_start, not bash, for servers and watchers that do not exit. "
    "Be concise."
)

_TODO = (
    "For a task with three or more steps, keep a checklist with the todo tool: send the whole list each time, "
    "keep exactly one item in_progress, and mark items done as soon as they are finished. "
    "Don't use it for simple one-step requests."
)

_TASK_ROLE = (
    "You are a research subagent. Investigate the task with the read-only tools you have. "
    "You cannot change files or run commands. Finish with a concise report for another agent: "
    "what you found, with file paths, function names and line numbers; say what you could not find."
)

_PLAN = "You are in PLAN mode: do not modify files; write/edit are unavailable. Explore, then propose a plan."


def build_system_prompt(
    workspace: Path,
    mode: Mode,
    today: date,
    os_name: str,
    instructions: Instructions | None = None,
    todo: bool = False,
    skills: str | None = None,
    agents: str | None = None,
    paths: ProtectedPaths | None = None,
) -> str:
    parts = [
        _ROLE,
        f"Workspace: {workspace}\nMode: {mode.value}\nOS: {os_name}\nDate: {today.isoformat()}",
        _GUIDANCE,
    ]
    if todo:
        parts.append(_TODO)
    if mode is Mode.PLAN:
        parts.append(_PLAN)
    if skills:
        parts.append(skills)
    if agents:
        parts.append(agents)
    if instructions is None:
        instructions = load_instructions(workspace, None, paths)
    if instructions.text:
        parts.append(instructions.text)
    return "\n\n".join(parts)


def build_task_prompt(
    workspace: Path,
    today: date,
    os_name: str,
    instructions: Instructions | None,
    skills: str | None = None,
    paths: ProtectedPaths | None = None,
    role: str | None = None,
) -> str:
    """The system prompt of a `task` child: a research role (or `role`), the workspace facts and the AGENTS.md text."""
    parts = [_TASK_ROLE if role is None else role, f"Workspace: {workspace}\nOS: {os_name}\nDate: {today.isoformat()}"]
    if skills:
        parts.append(skills)
    if instructions is None:
        instructions = load_instructions(workspace, None, paths)
    if instructions.text:
        parts.append(instructions.text)
    return "\n\n".join(parts)


_LOOK_FIRST = (
    "1. Look first: read the README, the manifests (pyproject.toml, package.json, …), the makefile and the CI config, "
    "and list the top-level layout with glob."
)
_FROM_FACTS = (
    "1. The facts below were read from the project's files. Use them for every command, "
    "and read the README for what the project is. Do not write a command that is not in the facts."
)


def build_init_prompt(workspace: Path, exists: bool, parents: Sequence[str], facts: str = "") -> str:
    """The built-in user message of `/init`: only the branch that applies is stated.

    `facts` is the rendered scan of the project; with it step 1 points at the facts and the block ends the message.
    """
    step = (
        "2. AGENTS.md already exists: read it, then improve it with edit. Keep what is right; do not rewrite it from scratch."
        if exists
        else "2. There is no AGENTS.md here yet: create it with write."
    )
    lines = [
        f"Write the project instructions for coding agents in AGENTS.md at the top of the workspace ({workspace}).",
        "",
        _FROM_FACTS if facts else _LOOK_FIRST,
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
    text = "\n".join(lines)
    return f"{text}\n\n{facts}" if facts else text
