"""Turning Textual key names into the bytes a terminal program expects."""

from __future__ import annotations

import pytest
from src.tui.keys import encode


@pytest.mark.parametrize(
    ("key", "character", "expected"),
    [
        ("home", None, b"\x1b[H"),
        ("f4", None, b"\x1bOS"),
        ("ctrl+underscore", None, b"\x1f"),
        ("shift+left", None, b"\x1b[1;2D"),
        ("ctrl+pageup", None, b"\x1b[5;5~"),
        ("alt+x", None, b"\x1bx"),
    ],
)
def test_a_key_is_encoded_as_a_terminal_would_send_it(key: str, character: str | None, expected: bytes) -> None:
    assert encode(key, character, app_cursor=False) == expected


def test_application_cursor_mode_does_not_change_other_keys() -> None:
    assert encode("pageup", None, app_cursor=True) == b"\x1b[5~"
    assert encode("a", "a", app_cursor=True) == b"a"


@pytest.mark.parametrize("key", ["ctrl+shift+a"])
def test_a_key_with_no_encoding_gives_none(key: str) -> None:
    assert encode(key, None, app_cursor=False) is None
