"""Finding, reading and fitting AGENTS.md files (spec R1-R6, AC1-AC7, AC17)."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

import pytest
from core.engine.instructions import (
    Instructions,
    levels,
    load_instructions,
)


def _root(path: Path) -> Path:
    (path / ".git").mkdir(parents=True)
    return path


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
def test_an_unreadable_parent_stops_the_walk_without_error(tmp_path: Path) -> None:
    root = _root(tmp_path / "repo")
    mid = root / "mid"
    ws = mid / "ws"
    ws.mkdir(parents=True)
    mid.chmod(0o000)
    try:
        result = levels(mid / "ws")
    finally:
        mid.chmod(0o755)
    assert result == [("AGENTS.md", ws / "AGENTS.md")]


def _write(folder: Path, text: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "AGENTS.md").write_text(text)


def _repo_with_cli(tmp_path: Path, root_text: str, cli_text: str) -> Path:
    root = _root(tmp_path / "repo")
    _write(root, root_text)
    _write(root / "cli", cli_text)
    return root / "cli"


# --- loading without a budget (R1-R3, AC1, AC2, AC4) ---


def _bad_at_root(tmp_path: Path, make) -> Path:
    root = _root(tmp_path / "repo")
    make(root / "AGENTS.md")
    _write(root / "cli", "cli rules")
    return root / "cli"


def _fifo(path: Path) -> None:
    os.mkfifo(path)


def _make_bad_kinds() -> dict:
    return {
        "folder": lambda p: p.mkdir(),
        "fifo": _fifo,
        "zero": lambda p: p.symlink_to("/dev/zero"),
        "dangling": lambda p: p.symlink_to("nowhere"),
        "empty": lambda p: p.write_text(""),
        "blank": lambda p: p.write_text(" \n\t\n"),
        "mode000": lambda p: (p.write_text("secret"), p.chmod(0)),
    }


@pytest.mark.parametrize("kind", list(_make_bad_kinds()))
async def test_a_bad_level_is_skipped_within_100_ms(tmp_path: Path, kind: str) -> None:
    if kind == "mode000" and os.geteuid() == 0:
        pytest.skip("root ignores file modes")
    ws = _bad_at_root(tmp_path, _make_bad_kinds()[kind])
    try:
        result = await asyncio.wait_for(asyncio.to_thread(load_instructions, ws, None), 0.1)
    finally:
        fifo = tmp_path / "repo" / "AGENTS.md"
        if kind == "fifo":
            with contextlib.suppress(OSError):
                os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        if kind == "mode000":
            fifo.chmod(0o644)
    assert result.text == "## cli/AGENTS.md\ncli rules"
    assert [f.label for f in result.files] == ["cli/AGENTS.md"]


def test_the_refusals_are_part_of_the_key_and_the_notice_follows_the_files(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "root", "cli")
    (ws.parent / "AGENTS.md").unlink()
    (ws.parent / "real.md").write_text("x")
    (ws.parent / "AGENTS.md").symlink_to("real.md")
    result = load_instructions(ws, None)
    assert result.key == (("cli/AGENTS.md", "whole"), ("AGENTS.md (symlink)", "refused"))
    level, text = result.notice() or ("", "")
    assert level == "warn"
    assert text.startswith("Instructions: cli/AGENTS.md (~") and text.endswith("; refused: AGENTS.md (symlink)")


# --- fitting (R5, R6, AC6, AC7, AC17) ---


def _budget_chars(tokens: int) -> int:
    return tokens * 4


def _single(tmp_path: Path, budget: int) -> Instructions:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "AGENTS.md").write_text("x" * 5000)
    return load_instructions(ws, budget)


def _kept(budget: int) -> int:
    return budget * 4 - len("## AGENTS.md\n") - len("\n[AGENTS.md cut to fit the instruction budget]") - 2


def test_an_omitted_note_that_does_not_fit_is_dropped(tmp_path: Path) -> None:
    result = _single(tmp_path, 5)  # 20 characters
    assert result.files[0].state == "omitted"
    assert result.text == ""
    assert len(result.text) <= 20


SIZES = [0, 1, 199, 200, 1024, 8 * 1024, 32 * 1024, 100 * 1024]


def _tree(tmp_path: Path, sizes: list[int]) -> Path:
    root = _root(tmp_path / "repo")
    folder = root
    for index, size in enumerate(sizes):
        if index:
            folder = folder / f"d{index}"
        _write(folder, "é" * size if size % 2 else "y" * size)
    return folder


# --- the notice (R9, AC10) ---


def test_the_notice_marks_cut_and_omitted_files_as_a_warning(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "a" * 12000, "b" * 9000)
    level, text = load_instructions(ws, 2048).notice() or ("", "")
    assert level == "warn"
    assert "AGENTS.md (omitted, ~3000)" in text
    assert "cli/AGENTS.md (~" in text and ", cut from ~2250)" in text


# --- review fixes: omitted lines, precedence, capped report ---

OMITTED_ROOT = "[AGENTS.md omitted: no room in the instruction budget]"


def test_a_capped_file_is_reported_in_characters_and_as_first_32_kb(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "r", "é" * (100 * 1024 // 2))  # 100 KB of 2-byte characters
    result = load_instructions(ws, None)
    cli = result.files[-1]
    assert cli.capped and cli.state == "cut"
    assert cli.original == 100 * 1024 // 2 // 4  # characters, not bytes
    level, text = result.notice() or ("", "")
    assert level == "warn"
    assert f"cli/AGENTS.md (~{cli.tokens}, first 32 KB of ~{cli.original})" in text


# --- fingerprint (R14) -----------------------------------------------------------------------
