"""Text from a file in the workspace, made safe to show or return."""

from __future__ import annotations

import unicodedata


def escape(text: str) -> str:
    """`text` with everything but printable ASCII written out (`\\x1b`, `\\u202e`, `\\U0001f600`).

    A project file is written by someone else: it must not be able to hide or rewrite what is shown or sent
    with escape sequences, carriage returns, newlines, or characters that reorder the text.
    """
    return "".join(char if char.isascii() and char.isprintable() else _written(char) for char in text)


def _written(char: str) -> str:
    code = ord(char)
    if code < 0x100:
        return f"\\x{code:02x}"
    if code < 0x10000:
        return f"\\u{code:04x}"
    return f"\\U{code:08x}"


def escape_display(text: str) -> str:
    """Like `escape`, but only control, format (bidi, zero-width, tags), surrogate, private, unassigned
    characters and the line/paragraph separators are written out; letters and punctuation of any script stay."""
    return "".join(
        _written(char) if unicodedata.category(char).startswith("C") or char in "\u2028\u2029" else char
        for char in text
    )
