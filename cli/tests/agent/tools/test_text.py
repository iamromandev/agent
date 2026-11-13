from __future__ import annotations

import random
from pathlib import Path

import pytest
from src.agent.tools.text import (
    TOOL_OUTPUT_LIMIT,
    is_binary,
    keep_sgr,
    strip_ansi,
    truncate,
    truncate_ansi,
    unified_diff,
)


def test_truncate_cuts_100000_characters_to_30000_plus_a_note() -> None:
    out = truncate("x" * 100_000)
    assert out == "x" * 30_000 + "\n[70000 characters truncated]"


def test_truncate_leaves_short_text_alone() -> None:
    assert truncate("short") == "short"
    assert truncate("x" * TOOL_OUTPUT_LIMIT) == "x" * TOOL_OUTPUT_LIMIT


def test_truncate_honours_a_custom_limit() -> None:
    assert truncate("abcdef", limit=4) == "abcd\n[2 characters truncated]"


def test_strip_ansi_removes_colour_and_cursor_codes() -> None:
    assert strip_ansi("\x1b[31mred\x1b[0m \x1b[1;32mgreen\x1b[0m") == "red green"
    assert strip_ansi("a\x1b[2Kb\x1b[?25lc") == "abc"


def test_strip_ansi_removes_osc_sequences() -> None:
    assert strip_ansi("\x1b]0;title\x07text") == "text"


def test_strip_ansi_leaves_plain_text() -> None:
    assert strip_ansi("plain\ntext") == "plain\ntext"


def test_is_binary_detects_nul_bytes(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc\x00def")
    assert is_binary(path)


def test_is_binary_detects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    assert is_binary(path)


def test_is_binary_accepts_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("héllo wörld\n", encoding="utf-8")
    assert not is_binary(path)


def test_is_binary_tolerates_a_multibyte_char_cut_at_the_8kb_boundary(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("a" * 8191 + "é" * 10, encoding="utf-8")  # é straddles byte 8192
    assert not is_binary(path)


def test_is_binary_accepts_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert not is_binary(path)


def test_unified_diff_marks_removed_and_added_lines() -> None:
    diff = unified_diff("a\nold\nc\n", "a\nnew\nc\n", "src/x.py")
    assert "--- a/src/x.py" in diff
    assert "+++ b/src/x.py" in diff
    assert "\n-old\n" in diff
    assert "\n+new\n" in diff


def test_unified_diff_of_identical_text_is_empty() -> None:
    assert unified_diff("same\n", "same\n", "x") == ""


def test_keep_sgr_keeps_colors_and_removes_everything_else() -> None:
    assert keep_sgr("\x1b[1;32mok\x1b[0m\x1b[2K\x1b[1A\x1b]0;title\x07!") == "\x1b[1;32mok\x1b[0m!"


def test_keep_sgr_drops_a_dangling_partial_sequence_at_the_end() -> None:
    assert keep_sgr("ok\x1b[3") == "ok"
    assert keep_sgr("ok\x1b[") == "ok"
    assert keep_sgr("ok\x1b") == "ok"
    assert keep_sgr("ok\x1b]0;unfinished title") == "ok"


def test_keep_sgr_drops_a_lone_escape_inside_the_text() -> None:
    assert keep_sgr("a\x1bb") == "ab"


def test_keep_sgr_leaves_plain_text_alone() -> None:
    assert keep_sgr("plain [1m text\n") == "plain [1m text\n"


def test_truncate_ansi_of_text_without_escapes_equals_truncate() -> None:
    for text in ("short", "x" * 100, "x" * 30_000, "x" * 30_001):
        assert truncate_ansi(text, 50) == truncate(text, 50)
    assert truncate_ansi("x" * 100_000) == truncate("x" * 100_000)


def test_truncate_ansi_does_not_count_sequences_and_resets_the_style() -> None:
    colored = "\x1b[32m" + "x" * 10 + "\x1b[0m"
    assert truncate_ansi(colored, 10) == colored  # 10 visible characters: untouched
    cut = truncate_ansi("\x1b[32m" + "x" * 20 + "\x1b[0m", 5)
    assert cut == "\x1b[32mxxxxx\x1b[0m\n[15 characters truncated]"


def test_truncate_ansi_never_cuts_inside_a_sequence() -> None:
    assert truncate_ansi("abc\x1b[1;31mdef", 3) == "abc\n[3 characters truncated]"
    assert truncate_ansi("\x1b[1;31mabcdef", 3) == "\x1b[1;31mabc\x1b[0m\n[3 characters truncated]"


@pytest.mark.parametrize("seed", range(20))
def test_the_visible_text_of_a_cut_matches_truncate_of_the_plain_text(seed: int) -> None:
    rng = random.Random(seed)
    pieces = ["a", "bc", "\n", "é", "\x1b[0m", "\x1b[1;31m", "\x1b[38;5;200m", "\x1b[m"]
    text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 200)))
    limit = rng.randint(1, 80)
    assert strip_ansi(truncate_ansi(text, limit)) == truncate(strip_ansi(text), limit)


def test_an_osc_ended_by_st_is_removed_and_the_text_after_it_is_kept() -> None:
    text = "x\x1b]8;;http://a\x1b\\link\x1b]8;;\x1b\\ end"
    assert keep_sgr(text) == "xlink end"
    assert strip_ansi(text) == "xlink end"


def test_an_unterminated_osc_at_the_end_is_dropped_with_either_terminator_style() -> None:
    assert keep_sgr("ok\x1b]8;;http://a") == "ok"
    assert keep_sgr("ok\x1b]8;;http://a\x1b") == "ok"


def test_many_unterminated_oscs_are_handled_in_linear_time() -> None:
    import time

    text = "\x1b]8;;" * 50_000 + "tail"
    start = time.perf_counter()
    keep_sgr(text)
    strip_ansi(text)
    assert time.perf_counter() - start < 2.0


def test_colon_form_sgr_is_kept_by_keep_sgr_and_removed_by_strip_ansi() -> None:
    text = "\x1b[38:2:1:2:3mx\x1b[0m"
    assert keep_sgr(text) == text
    assert strip_ansi(text) == "x"
