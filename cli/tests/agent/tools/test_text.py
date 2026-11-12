from __future__ import annotations

from pathlib import Path

from src.agent.tools.text import (
    TOOL_OUTPUT_LIMIT,
    is_binary,
    strip_ansi,
    truncate,
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
