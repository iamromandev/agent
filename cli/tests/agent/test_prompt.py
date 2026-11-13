from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import date
from pathlib import Path

from src.agent.instructions import AGENTS_MD_LIMIT
from src.agent.mode import Mode
from src.agent.prompt import build_init_prompt, build_system_prompt

TODAY = date(2026, 11, 12)


def _prompt(workspace: Path, mode: Mode = Mode.BUILD, os_name: str = "Darwin") -> str:
    return build_system_prompt(workspace, mode, TODAY, os_name)


def test_prompt_names_workspace_mode_date_and_os(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path, Mode.PLAN, "Linux")
    assert f"Workspace: {tmp_path}" in prompt
    assert "Mode: plan" in prompt
    assert "Date: 2026-11-12" in prompt
    assert "OS: Linux" in prompt


def test_build_mode_is_named_build(tmp_path: Path) -> None:
    assert "Mode: build" in _prompt(tmp_path, Mode.BUILD)


def test_prompt_guides_tool_use(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path)
    assert "prefer edit over write" in prompt.lower()
    assert "read" in prompt.lower()


def test_plan_text_appears_only_in_plan_mode(tmp_path: Path) -> None:
    marker = "You are in PLAN mode: do not modify files; write/edit are unavailable"
    assert marker in _prompt(tmp_path, Mode.PLAN)
    assert "PLAN mode" not in _prompt(tmp_path, Mode.BUILD)


def test_agents_md_is_included(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Use tabs\n")
    prompt = _prompt(tmp_path)
    assert "## AGENTS.md\nUse tabs" in prompt


def test_no_agents_md_section_without_the_file(tmp_path: Path) -> None:
    assert "AGENTS.md" not in _prompt(tmp_path)


def test_a_100_kb_agents_md_is_cut_to_32_kb_with_a_note(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Z" * (100 * 1024))
    prompt = _prompt(tmp_path)
    assert AGENTS_MD_LIMIT == 32 * 1024
    assert "[AGENTS.md truncated at 32 KB]" in prompt
    assert prompt.count("Z") == AGENTS_MD_LIMIT


def test_a_small_agents_md_has_no_truncation_note(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Z" * AGENTS_MD_LIMIT)
    assert "truncated" not in _prompt(tmp_path)


def test_a_multibyte_cut_does_not_break_the_prompt(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("a" + "é" * AGENTS_MD_LIMIT)  # the cut lands inside an é
    prompt = _prompt(tmp_path)
    assert "[AGENTS.md truncated at 32 KB]" in prompt
    assert "�" not in prompt


def test_an_unreadable_agents_md_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").mkdir()  # a directory with that name
    assert "AGENTS.md" not in _prompt(tmp_path)


async def test_a_fifo_named_agents_md_does_not_block_the_prompt(tmp_path: Path) -> None:
    fifo = tmp_path / "AGENTS.md"
    os.mkfifo(fifo)
    try:
        prompt = await asyncio.wait_for(asyncio.to_thread(_prompt, tmp_path), timeout=2.0)
    finally:
        with contextlib.suppress(OSError):  # frees a reader stuck in open(), so the worker thread can end
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
    assert "AGENTS.md" not in prompt


def test_a_symlink_to_an_endless_device_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").symlink_to("/dev/zero")
    assert "AGENTS.md" not in _prompt(tmp_path)


def test_a_symlink_to_a_regular_file_is_read(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("Use tabs\n")
    (tmp_path / "AGENTS.md").symlink_to("real.md")
    assert "## AGENTS.md\nUse tabs" in _prompt(tmp_path)


def test_an_invalid_byte_mid_file_is_replaced_not_dropped(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"ab\xffcd\n")
    assert "ab\ufffdcd" in _prompt(tmp_path)


def test_an_invalid_byte_mid_file_in_a_large_file_is_replaced(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"x\xfe" + b"Z" * (2 * AGENTS_MD_LIMIT))
    prompt = _prompt(tmp_path)
    assert "x\ufffdZ" in prompt
    assert "[AGENTS.md truncated at 32 KB]" in prompt


def test_a_file_that_ends_inside_a_multibyte_character_is_replaced_not_dropped(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_bytes("ok é".encode()[:-1])  # complete file, last char cut
    assert "ok \ufffd" in _prompt(tmp_path)


def test_the_cut_at_the_limit_drops_only_the_partial_character(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("é" * AGENTS_MD_LIMIT)  # 2 bytes each
    prompt = _prompt(tmp_path)
    assert prompt.count("é") == AGENTS_MD_LIMIT // 2
    assert "\ufffd" not in prompt


def test_a_single_small_agents_md_gives_exactly_todays_prompt(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Use tabs\n")
    expected = (
        "You are a coding agent running in the user's terminal. You help with software tasks "
        "in the workspace by reading, searching and, when allowed, changing files and running commands."
        f"\n\nWorkspace: {tmp_path}\nMode: build\nOS: Darwin\nDate: 2026-11-12"
        "\n\nUse the tools to look before you answer. Read a file before you edit it. "
        "Prefer edit over write for existing files, and keep edits small. "
        "Use glob and grep to find files instead of guessing paths. "
        "Tool calls may be denied by the user; if one is, follow their note and do not retry it unchanged. "
        "Be concise."
        "\n\n## AGENTS.md\nUse tabs\n"
    )
    assert _prompt(tmp_path) == expected


def test_passed_instructions_are_appended_instead_of_loading(tmp_path: Path) -> None:
    from src.agent.instructions import Instructions

    (tmp_path / "AGENTS.md").write_text("from disk")
    prompt = build_system_prompt(tmp_path, Mode.BUILD, TODAY, "Darwin", Instructions("## X\nhi", (), 10))
    assert prompt.endswith("\n\n## X\nhi")
    assert "from disk" not in prompt


# --- build_init_prompt (R12) -----------------------------------------------------------------


def test_init_prompt_names_the_workspace_and_covers_each_r12_point(tmp_path: Path) -> None:
    prompt = build_init_prompt(tmp_path, False, [])

    assert f"AGENTS.md at the top of the workspace ({tmp_path})" in prompt
    for needle in ("README", "manifests", "makefile", "CI config", "glob"):
        assert needle in prompt  # look at the project first
    assert "what the project is" in prompt
    assert "build, test, lint and run" in prompt
    assert "conventions that file names do not show" in prompt
    assert "only what you found in the files" in prompt
    assert "60 lines" in prompt


def test_init_prompt_creates_with_write_when_there_is_no_file(tmp_path: Path) -> None:
    prompt = build_init_prompt(tmp_path, False, [])

    assert "no AGENTS.md here yet: create it with write" in prompt
    assert "improve it with edit" not in prompt


def test_init_prompt_improves_with_edit_when_the_file_exists(tmp_path: Path) -> None:
    prompt = build_init_prompt(tmp_path, True, [])

    assert "already exists: read it, then improve it with edit" in prompt
    assert "create it with write" not in prompt


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


def test_the_init_prompt_without_facts_is_exactly_todays_text() -> None:
    assert build_init_prompt(Path("/w"), False, []) == TODAYS_INIT_PROMPT
    assert build_init_prompt(Path("/w"), False, [], facts="") == TODAYS_INIT_PROMPT


def test_the_init_prompt_with_facts_replaces_step_1_and_ends_with_the_block() -> None:
    block = "Facts found\n\nMake targets\n- make test"
    text = build_init_prompt(Path("/w"), False, [], facts=block)
    lines = TODAYS_INIT_PROMPT.splitlines()
    expected = "\n".join([lines[0], lines[1], FACTS_STEP_1, *lines[3:]]) + "\n\n" + block
    assert text == expected
    assert "read the README, the manifests" not in text and "makefile" not in text.split("\n\n")[1]


def test_facts_keep_steps_2_to_5_intact(tmp_path: Path) -> None:
    plain = build_init_prompt(tmp_path, True, ["AGENTS.md", "x/AGENTS.md"]).splitlines()
    with_facts = build_init_prompt(tmp_path, True, ["AGENTS.md", "x/AGENTS.md"], facts="F").splitlines()
    assert with_facts[:2] == plain[:2]
    assert with_facts[3 : len(plain)] == plain[3:]
    assert with_facts[-2:] == ["", "F"]
    assert "5. The instructions in AGENTS.md, x/AGENTS.md are already loaded" in "\n".join(with_facts)
