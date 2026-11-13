"""Finding, reading and fitting AGENTS.md files (spec R1-R6, AC1-AC7, AC17)."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import random
import statistics
import time
from pathlib import Path

import pytest
from src.agent.instructions import (
    AGENTS_MD_LIMIT,
    MIN_KEPT,
    PRECEDENCE,
    InstructionFile,
    Instructions,
    fingerprint,
    levels,
    load_instructions,
)


def _root(path: Path) -> Path:
    (path / ".git").mkdir(parents=True)
    return path


def test_the_workspace_is_the_git_root(tmp_path: Path) -> None:
    root = _root(tmp_path / "repo")
    assert levels(root) == [("AGENTS.md", root / "AGENTS.md")]


def test_two_levels_run_from_the_root_down(tmp_path: Path) -> None:
    root = _root(tmp_path / "repo")
    cli = root / "cli"
    cli.mkdir()
    assert levels(cli) == [("AGENTS.md", root / "AGENTS.md"), ("cli/AGENTS.md", cli / "AGENTS.md")]


def test_three_levels_have_posix_labels(tmp_path: Path) -> None:
    root = _root(tmp_path / "repo")
    deep = root / "a" / "b"
    deep.mkdir(parents=True)
    assert [label for label, _ in levels(deep)] == ["AGENTS.md", "a/AGENTS.md", "a/b/AGENTS.md"]


def test_without_a_git_root_only_the_workspace_counts(tmp_path: Path) -> None:
    ws = tmp_path / "x" / "y"
    ws.mkdir(parents=True)
    assert levels(ws) == [("AGENTS.md", ws / "AGENTS.md")]


def test_git_as_a_file_is_a_root(tmp_path: Path) -> None:
    outer = _root(tmp_path / "outer")
    inner = outer / "sub"
    ws = inner / "pkg"
    ws.mkdir(parents=True)
    (inner / ".git").write_text("gitdir: ../.git/modules/sub\n")
    assert [label for label, _ in levels(ws)] == ["AGENTS.md", "pkg/AGENTS.md"]
    assert levels(ws)[0][1] == inner / "AGENTS.md"


def test_the_nearest_git_wins(tmp_path: Path) -> None:
    outer = _root(tmp_path / "outer")
    inner = _root(outer / "inner")
    assert levels(inner) == [("AGENTS.md", inner / "AGENTS.md")]


def test_a_file_above_the_root_is_not_a_level(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("above")
    root = _root(tmp_path / "repo")
    assert all(path.parent != tmp_path for _, path in levels(root))


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


def test_root_and_workspace_files_come_outermost_first_with_the_precedence_line(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "root rules\n", "cli rules\n")
    text = load_instructions(ws, None).text
    assert PRECEDENCE in text
    assert text.index("## AGENTS.md\nroot rules") < text.index("## cli/AGENTS.md\ncli rules")
    assert text.index(PRECEDENCE) < text.index("## AGENTS.md")


def test_a_single_file_has_no_precedence_line(tmp_path: Path) -> None:
    root = _root(tmp_path / "repo")
    _write(root, "only\n")
    result = load_instructions(root, None)
    assert result.text == "## AGENTS.md\nonly\n"
    assert result.files == (InstructionFile("AGENTS.md", 2, 2, "whole"),)


def test_a_file_above_the_git_root_is_not_loaded(tmp_path: Path) -> None:
    _write(tmp_path, "ABOVE")
    root = _root(tmp_path / "repo")
    _write(root, "inside")
    text = load_instructions(root, None).text
    assert "ABOVE" not in text
    assert "inside" in text


def test_outside_git_only_the_workspace_file_is_loaded(tmp_path: Path) -> None:
    _write(tmp_path, "PARENT")
    ws = tmp_path / "ws"
    _write(ws, "own")
    assert load_instructions(ws, None).text == "## AGENTS.md\nown"


def test_loading_stops_at_a_git_file(tmp_path: Path) -> None:
    outer = _root(tmp_path / "outer")
    _write(outer, "OUTER")
    inner = outer / "sub"
    _write(inner, "inner")
    (inner / ".git").write_text("gitdir: x\n")
    assert "OUTER" not in load_instructions(inner, None).text


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


def test_a_symlink_to_a_regular_file_is_read(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "real.md").write_text("linked")
    (ws / "AGENTS.md").symlink_to("real.md")
    assert load_instructions(ws, None).text == "## AGENTS.md\nlinked"


def test_invalid_utf8_is_replaced(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "AGENTS.md").write_bytes(b"ab\xffcd")
    assert "ab\ufffdcd" in load_instructions(ws, None).text


def test_only_agents_md_files_up_to_the_root_are_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "ABOVE")
    ws = _repo_with_cli(tmp_path, "r", "c")
    opened: list[str] = []
    real = os.open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy)
    load_instructions(ws, None)
    root = tmp_path / "repo"
    assert opened == [str(root / "AGENTS.md"), str(ws / "AGENTS.md")]


def test_a_file_over_the_cap_reads_32_kb_with_a_labelled_note(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "r", "Z" * (100 * 1024))
    result = load_instructions(ws, None)
    assert result.text.count("Z") == AGENTS_MD_LIMIT
    assert "\n[cli/AGENTS.md truncated at 32 KB]" in result.text
    cli = result.files[-1]
    assert cli.state == "cut"
    assert cli.original == 100 * 1024 // 4


# --- fitting (R5, R6, AC6, AC7, AC17) ---


def _budget_chars(tokens: int) -> int:
    return tokens * 4


def test_the_workspace_file_is_whole_and_the_root_file_is_cut_to_fit(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "R" * (12 * 1024), "C" * 2048)
    result = load_instructions(ws, 2048)
    assert len(result.text) <= _budget_chars(2048)
    assert "C" * 2048 in result.text
    assert [f.state for f in result.files] == ["cut", "whole"]
    assert result.text.endswith("C" * 2048)
    assert "\n[AGENTS.md cut to fit the instruction budget]" in result.text
    assert PRECEDENCE in result.text


def test_a_workspace_file_over_the_budget_is_cut_and_outer_files_are_omitted(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "R" * 3000, "C" * 5000)
    result = load_instructions(ws, 300)  # 1200 characters
    assert len(result.text) <= 1200
    assert [f.state for f in result.files] == ["omitted", "cut"]
    assert result.text.rstrip().endswith("[cli/AGENTS.md cut to fit the instruction budget]")
    assert "R" * 10 not in result.text


def test_an_omitted_outer_file_is_named_when_the_line_fits(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "R" * 5000, "C" * 3640)
    result = load_instructions(ws, 1000)  # 4000 characters
    assert [f.state for f in result.files] == ["omitted", "whole"]
    assert result.text.startswith("[AGENTS.md omitted: no room in the instruction budget]\n\n## cli/AGENTS.md")
    assert len(result.text) <= 4000


def _single(tmp_path: Path, budget: int) -> Instructions:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    (ws / "AGENTS.md").write_text("x" * 5000)
    return load_instructions(ws, budget)


def _kept(budget: int) -> int:
    return budget * 4 - len("## AGENTS.md\n") - len("\n[AGENTS.md cut to fit the instruction budget]") - 2


def test_a_file_keeping_min_kept_characters_is_cut_and_one_keeping_fewer_is_omitted(tmp_path: Path) -> None:
    smallest = next(b for b in range(1, 1000) if _kept(b) >= MIN_KEPT)
    assert _kept(smallest - 1) < MIN_KEPT
    assert _single(tmp_path, smallest).files[0].state == "cut"
    assert _single(tmp_path, smallest - 1).files[0].state == "omitted"


def test_an_omitted_note_that_does_not_fit_is_dropped(tmp_path: Path) -> None:
    result = _single(tmp_path, 5)  # 20 characters
    assert result.files[0].state == "omitted"
    assert result.text == ""
    assert len(result.text) <= 20


def test_a_budget_of_zero_loads_nothing_and_does_not_raise(tmp_path: Path) -> None:
    assert _single(tmp_path, 0).text == ""


def test_a_fitting_file_is_whole_under_a_budget(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write(ws, "small")
    result = load_instructions(ws, 100)
    assert result.text == "## AGENTS.md\nsmall"
    assert result.files[0].state == "whole"


SIZES = [0, 1, 199, 200, 1024, 8 * 1024, 32 * 1024, 100 * 1024]


def _tree(tmp_path: Path, sizes: list[int]) -> Path:
    root = _root(tmp_path / "repo")
    folder = root
    for index, size in enumerate(sizes):
        if index:
            folder = folder / f"d{index}"
        _write(folder, "é" * size if size % 2 else "y" * size)
    return folder


@pytest.mark.parametrize("window", [2048, 8192])
@pytest.mark.parametrize("share", [0.01, 0.1, 0.25, 0.5])
@pytest.mark.parametrize("size", SIZES)
def test_the_instructions_never_exceed_the_budget(tmp_path: Path, size: int, share: float, window: int) -> None:
    budget = int(share * window)
    for depth in (1, 3, 8):
        ws = _tree(tmp_path / str(depth), [size] * depth)
        assert len(load_instructions(ws, budget).text) <= budget * 4


def test_random_trees_never_exceed_the_budget_or_raise(tmp_path: Path) -> None:
    rng = random.Random(1234)
    for n in range(300):
        sizes = [rng.choice([*SIZES, rng.randrange(0, 40000)]) for _ in range(rng.randint(1, 8))]
        budget = int(rng.choice([0.01, 0.1, 0.25, 0.5]) * rng.choice([2048, 8192]))
        ws = _tree(tmp_path / str(n), sizes)
        assert len(load_instructions(ws, budget).text) <= budget * 4


def test_eight_levels_of_32_kb_load_in_20_ms(tmp_path: Path) -> None:
    ws = _tree(tmp_path, [32 * 1024] * 8)
    for _ in range(5):  # warm up: the page cache and the import-time work
        load_instructions(ws, 2048)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        load_instructions(ws, 2048)
        times.append(time.perf_counter() - start)
    assert statistics.median(times) <= 0.02


# --- the notice (R9, AC10) ---


def test_the_notice_lists_whole_files_as_info(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "a" * 5000, "b" * 1200)
    result = load_instructions(ws, 2048)
    level, text = result.notice() or ("", "")
    assert level == "info"
    assert text == f"Instructions: AGENTS.md (~1250), cli/AGENTS.md (~300) \u00b7 ~{result.tokens} of ~2048 tokens"


def test_the_notice_marks_cut_and_omitted_files_as_a_warning(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "a" * 12000, "b" * 9000)
    level, text = load_instructions(ws, 2048).notice() or ("", "")
    assert level == "warn"
    assert "AGENTS.md (omitted, ~3000)" in text
    assert "cli/AGENTS.md (~" in text and ", cut from ~2250)" in text


def test_the_notice_shows_a_cut_file_with_its_original_size(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "a" * 12000, "b" * 2000)
    level, text = load_instructions(ws, 2048).notice() or ("", "")
    assert level == "warn"
    assert text.startswith("Instructions: AGENTS.md (~") and ", cut from ~3000), cli/AGENTS.md (~500) " in text


def test_the_notice_without_a_budget_has_no_total(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write(ws, "a" * 400)
    result = load_instructions(ws, None)
    assert result.notice() == ("info", "Instructions: AGENTS.md (~100) \u00b7 ~25 tokens".replace("~25", f"~{result.tokens}"))


def test_there_is_no_notice_without_files(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    assert load_instructions(ws, 2048).notice() is None


def test_the_key_names_each_file_and_its_state(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "a", "b")
    assert load_instructions(ws, None).key == (("AGENTS.md", "whole"), ("cli/AGENTS.md", "whole"))


def test_tokens_round_up_from_the_text_length() -> None:
    assert Instructions("x" * 9, (), None).tokens == math.ceil(9 / 4)


# --- review fixes: omitted lines, precedence, capped report ---

OMITTED_ROOT = "[AGENTS.md omitted: no room in the instruction budget]"


def test_a_workspace_file_over_the_budget_still_names_the_omitted_root_file(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "R" * 3000, "C" * 5000)
    result = load_instructions(ws, 500)
    assert OMITTED_ROOT in result.text
    assert len(result.text) <= 2000
    assert [f.state for f in result.files] == ["omitted", "cut"]
    assert result.text.rstrip().endswith("[cli/AGENTS.md cut to fit the instruction budget]")


def test_three_levels_all_outer_files_are_named_when_the_inner_one_is_cut(tmp_path: Path) -> None:
    root = _root(tmp_path / "repo")
    _write(root, "R" * 3000)
    _write(root / "a", "A" * 3000)
    _write(root / "a" / "b", "B" * 5000)
    result = load_instructions(root / "a" / "b", 500)
    assert OMITTED_ROOT in result.text
    assert "[a/AGENTS.md omitted: no room in the instruction budget]" in result.text
    assert len(result.text) <= 2000
    assert [f.state for f in result.files] == ["omitted", "omitted", "cut"]


def test_a_tiny_budget_drops_the_notes_but_keeps_the_workspace_content(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "R" * 3000, "C" * 5000)
    # Room for the cut workspace file with MIN_KEPT characters, but not for the root's note as well.
    overhead = len("## cli/AGENTS.md\n") + len("\n[cli/AGENTS.md cut to fit the instruction budget]") + 2
    budget = math.ceil((overhead + MIN_KEPT) / 4)
    result = load_instructions(ws, budget)
    assert result.files[-1].state == "cut"
    assert result.text.count("C") >= MIN_KEPT
    assert OMITTED_ROOT not in result.text
    assert len(result.text) <= budget * 4


def test_the_precedence_line_is_not_reserved_for_a_single_included_file(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "R" * 500, "W" * 100)
    result = load_instructions(ws, 40)  # 160 characters
    assert [f.state for f in result.files] == ["omitted", "whole"]
    assert "W" * 100 in result.text
    assert PRECEDENCE not in result.text
    assert len(result.text) <= 160


def test_two_included_sections_carry_the_precedence_line_within_the_budget(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "R" * 300, "W" * 100)
    need = len(PRECEDENCE) + len("## AGENTS.md\n") + 300 + len("## cli/AGENTS.md\n") + 100 + 6
    budget = math.ceil(need / 4)
    result = load_instructions(ws, budget)
    assert [f.state for f in result.files] == ["whole", "whole"]
    assert PRECEDENCE in result.text
    assert len(result.text) <= budget * 4


def test_a_capped_file_is_reported_in_characters_and_as_first_32_kb(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "r", "é" * (100 * 1024 // 2))  # 100 KB of 2-byte characters
    result = load_instructions(ws, None)
    cli = result.files[-1]
    assert cli.capped and cli.state == "cut"
    assert cli.original == 100 * 1024 // 2 // 4  # characters, not bytes
    level, text = result.notice() or ("", "")
    assert level == "warn"
    assert f"cli/AGENTS.md (~{cli.tokens}, first 32 KB of ~{cli.original})" in text


def test_a_budget_cut_is_not_marked_as_capped(tmp_path: Path) -> None:
    ws = _repo_with_cli(tmp_path, "r", "C" * 5000)
    cli = load_instructions(ws, 300).files[-1]
    assert cli.state == "cut" and not cli.capped
    assert ", cut from ~" in (load_instructions(ws, 300).notice() or ("", ""))[1]


# --- fingerprint (R14) -----------------------------------------------------------------------


def test_fingerprint_of_a_missing_file_or_a_folder_is_none(tmp_path: Path) -> None:
    assert fingerprint(tmp_path / "AGENTS.md") is None
    (tmp_path / "dir").mkdir()
    assert fingerprint(tmp_path / "dir") is None
    assert fingerprint(tmp_path / "no" / "such" / "AGENTS.md") is None


def test_fingerprint_changes_with_the_content_size(tmp_path: Path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text("one")
    before = fingerprint(path)
    assert before is not None and before[0] == 3

    path.write_text("three")
    assert fingerprint(path) != before


def test_fingerprint_changes_with_the_mtime_and_is_stable_otherwise(tmp_path: Path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text("same")
    before = fingerprint(path)
    assert fingerprint(path) == before

    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    assert fingerprint(path) != before
