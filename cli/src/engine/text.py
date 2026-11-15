"""Text from a file in the workspace, made safe to show or return."""

from __future__ import annotations


def escape(text: str) -> str:
    """`text` with everything but printable ASCII written out (`\\x1b`, `\\u202e`, `\\U0001f600`).

    A project file is written by someone else: it must not be able to hide or rewrite what is shown or sent
    with escape sequences, carriage returns, newlines, or characters that reorder the text.
    """
    out: list[str] = []
    for char in text:
        code = ord(char)
        if char.isascii() and char.isprintable():
            out.append(char)
        elif code < 0x100:
            out.append(f"\\x{code:02x}")
        elif code < 0x10000:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    return "".join(out)
