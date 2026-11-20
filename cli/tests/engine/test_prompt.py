from __future__ import annotations

from datetime import date
from pathlib import Path

from src.engine.mode import Mode
from src.engine.prompt import build_init_prompt, build_system_prompt, build_task_prompt

TODAY = date(2026, 11, 12)


def _prompt(workspace: Path, mode: Mode = Mode.BUILD, os_name: str = "Darwin") -> str:
    return build_system_prompt(workspace, mode, TODAY, os_name)


def test_prompt_names_workspace_mode_date_and_os(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path, Mode.PLAN, "Linux")
    assert f"Workspace: {tmp_path}" in prompt
    assert "Mode: plan" in prompt
    assert "Date: 2026-11-12" in prompt
    assert "OS: Linux" in prompt


# --- build_init_prompt (R12) -----------------------------------------------------------------


def test_init_prompt_names_the_parents_only_when_there_are_some(tmp_path: Path) -> None:
    assert "already loaded" not in build_init_prompt(tmp_path, False, [])

    prompt = build_init_prompt(tmp_path, False, ["AGENTS.md", "api/AGENTS.md"])

    assert prompt.splitlines()[-1] == (
        "5. The instructions in AGENTS.md, api/AGENTS.md are already loaded: do not repeat them; "
        "cover only what is specific to this folder."
    )


# build_init_prompt with and without facts (R8, R10, AC9)

TODAYS_INIT_PROMPT = (
    "Write the project instructions for coding agents in AGENTS.md at the top of the workspace (/w).\n"
    "\n"
    "1. Look first: read the README, the manifests (pyproject.toml, package.json, …), the makefile and the CI config, "
    "and list the top-level layout with glob.\n"
    "2. There is no AGENTS.md here yet: create it with write.\n"
    "3. Cover what the project is, the commands to build, test, lint and run it, "
    "and the layout and conventions that file names do not show.\n"
    "4. Write only what you found in the files. Keep it short: at most about 60 lines."
)
FACTS_STEP_1 = (
    "1. The facts below were read from the project's files. Use them for every command, "
    "and read the README for what the project is. Do not write a command that is not in the facts."
)


def test_task_prompt_has_the_role_workspace_and_agents_md(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Project rules here.")
    prompt = build_task_prompt(tmp_path, TODAY, "Linux", None)
    assert "You are a research subagent." in prompt
    assert "You cannot change files or run commands." in prompt
    assert f"Workspace: {tmp_path}" in prompt
    assert "Date: 2026-11-12" in prompt
    assert "Project rules here." in prompt
    assert "Mode:" not in prompt


SECTION = "Available skills (call the skill tool ...):\n- a: does a"


def test_skills_section_comes_before_the_instructions_text(tmp_path: Path) -> None:
    from src.engine.instructions import Instructions

    instructions = Instructions("RULES-TEXT", (), None)
    prompt = build_system_prompt(tmp_path, Mode.PLAN, TODAY, "Linux", instructions=instructions, skills=SECTION)
    assert prompt.index("PLAN mode") < prompt.index(SECTION) < prompt.index("RULES-TEXT")


def test_task_prompt_has_skills_section_only_when_given(tmp_path: Path) -> None:
    from src.engine.instructions import Instructions

    empty = Instructions("", (), None)
    assert "Available skills" not in build_task_prompt(tmp_path, TODAY, "Linux", empty)
    assert SECTION in build_task_prompt(tmp_path, TODAY, "Linux", empty, skills=SECTION)
