from __future__ import annotations

from pathlib import Path

from core.engine.tools.text import (
    is_binary,
    truncate,
    truncate_ansi,
)


def test_is_binary_detects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    assert is_binary(path)


def test_truncate_ansi_of_text_without_escapes_equals_truncate() -> None:
    for text in ("short", "x" * 100, "x" * 30_000, "x" * 30_001):
        assert truncate_ansi(text, 50) == truncate(text, 50)
    assert truncate_ansi("x" * 100_000) == truncate("x" * 100_000)


def test_truncate_ansi_does_not_count_sequences_and_resets_the_style() -> None:
    colored = "\x1b[32m" + "x" * 10 + "\x1b[0m"
    assert truncate_ansi(colored, 10) == colored  # 10 visible characters: untouched
    cut = truncate_ansi("\x1b[32m" + "x" * 20 + "\x1b[0m", 5)
    assert cut == "\x1b[32mxxxxx\x1b[0m\n[15 characters truncated]"
