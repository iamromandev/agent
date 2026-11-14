"""Turning Textual key names into the bytes a terminal program expects."""

from __future__ import annotations

import pytest
from src.tui.keys import encode


@pytest.mark.parametrize(
    ("key", "character", "expected"),
    [
        ("a", "a", b"a"),
        ("A", "A", b"A"),
        ("space", " ", b" "),
        ("é", "é", "é".encode()),
        ("question_mark", "?", b"?"),
        ("enter", "\r", b"\r"),
        ("enter", None, b"\r"),
        ("tab", "\t", b"\t"),
        ("shift+tab", None, b"\x1b[Z"),
        ("escape", "\x1b", b"\x1b"),
        ("backspace", "\x08", b"\x7f"),
        ("delete", "\x7f", b"\x1b[3~"),
        ("insert", None, b"\x1b[2~"),
        ("up", None, b"\x1b[A"),
        ("down", None, b"\x1b[B"),
        ("right", None, b"\x1b[C"),
        ("left", None, b"\x1b[D"),
        ("home", None, b"\x1b[H"),
        ("end", None, b"\x1b[F"),
        ("pageup", None, b"\x1b[5~"),
        ("pagedown", None, b"\x1b[6~"),
        ("f1", None, b"\x1bOP"),
        ("f4", None, b"\x1bOS"),
        ("f5", None, b"\x1b[15~"),
        ("f12", None, b"\x1b[24~"),
        ("ctrl+a", None, b"\x01"),
        ("ctrl+c", None, b"\x03"),
        ("ctrl+d", None, b"\x04"),
        ("ctrl+z", None, b"\x1a"),
        ("ctrl+t", None, b"\x14"),
        ("ctrl+j", None, b"\n"),
        ("ctrl+h", None, b"\x08"),
        ("ctrl+space", None, b"\x00"),
        ("ctrl+@", None, b"\x00"),
        ("ctrl+backslash", None, b"\x1c"),
        ("ctrl+right_square_bracket", None, b"\x1d"),
        ("ctrl+circumflex_accent", None, b"\x1e"),
        ("ctrl+underscore", None, b"\x1f"),
        ("ctrl+up", None, b"\x1b[1;5A"),
        ("shift+left", None, b"\x1b[1;2D"),
        ("alt+right", None, b"\x1b[1;3C"),
        ("ctrl+home", None, b"\x1b[1;5H"),
        ("ctrl+delete", None, b"\x1b[3;5~"),
        ("ctrl+pageup", None, b"\x1b[5;5~"),
        ("alt+x", None, b"\x1bx"),
        ("alt+enter", None, b"\x1b\r"),
        ("alt+backspace", None, b"\x1b\x7f"),
    ],
)
def test_a_key_is_encoded_as_a_terminal_would_send_it(key: str, character: str | None, expected: bytes) -> None:
    assert encode(key, character, app_cursor=False) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("up", b"\x1bOA"),
        ("down", b"\x1bOB"),
        ("right", b"\x1bOC"),
        ("left", b"\x1bOD"),
        ("home", b"\x1bOH"),
        ("end", b"\x1bOF"),
    ],
)
def test_application_cursor_mode_uses_ss3_for_arrows_home_and_end(key: str, expected: bytes) -> None:
    assert encode(key, None, app_cursor=True) == expected


def test_modified_arrows_keep_the_csi_form_in_application_cursor_mode() -> None:
    assert encode("ctrl+up", None, app_cursor=True) == b"\x1b[1;5A"


def test_application_cursor_mode_does_not_change_other_keys() -> None:
    assert encode("pageup", None, app_cursor=True) == b"\x1b[5~"
    assert encode("a", "a", app_cursor=True) == b"a"


@pytest.mark.parametrize("key", ["volume_up", "ctrl+shift+a", "f13", "ctrl+f1", "nonsense"])
def test_a_key_with_no_encoding_gives_none(key: str) -> None:
    assert encode(key, None, app_cursor=False) is None
