from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import date
from pathlib import Path

from src.agent.mode import Mode
from src.agent.prompt import AGENTS_MD_LIMIT, build_system_prompt

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
