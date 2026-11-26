"""Encode Textual key events as the bytes a terminal program reads from its tty."""

from __future__ import annotations

_CTRL = {  # control characters of keys Textual names instead of `ctrl+<letter>`
    "space": 0x00,
    "@": 0x00,
    "backslash": 0x1C,
    "right_square_bracket": 0x1D,
    "circumflex_accent": 0x1E,
    "underscore": 0x1F,
}
_PLAIN = {
    "enter": b"\r",
    "tab": b"\t",
    "shift+tab": b"\x1b[Z",
    "escape": b"\x1b",
    "backspace": b"\x7f",
}
_CURSOR = {"up": "A", "down": "B", "right": "C", "left": "D", "home": "H", "end": "F"}  # CSI <final>, or SS3 <final>
_TILDE = {"insert": 2, "delete": 3, "pageup": 5, "pagedown": 6}  # CSI <n> ~
_FUNCTION = {  # f1-f4 are SS3 P-S, the others CSI <n> ~
    **{f"f{n}": f"\x1bO{c}".encode() for n, c in zip(range(1, 5), "PQRS", strict=True)},
    **{f"f{n}": f"\x1b[{code}~".encode() for n, code in zip(range(5, 13), (15, 17, 18, 19, 20, 21, 23, 24), strict=True)},
}
_MODIFIER = {"shift": 2, "alt": 3, "ctrl": 5}  # xterm's modifier parameter: 1 + bit mask


def encode(key: str, character: str | None, *, app_cursor: bool) -> bytes | None:
    """The bytes for a key press, or None when a terminal has no encoding for it.

    `key` is Textual's key name (`ctrl+up`, `f5`, `a`), `character` what it types (None for non-printing keys),
    `app_cursor` whether the program asked for application cursor keys (`ESC O A` arrows).
    """
    if character is not None and len(character) == 1 and character.isprintable():
        return character.encode()
    if key in _PLAIN:
        return _PLAIN[key]
    if key in _FUNCTION:
        return _FUNCTION[key]
    if key in _CURSOR:
        return f"\x1b{'O' if app_cursor else '['}{_CURSOR[key]}".encode()
    if key in _TILDE:
        return f"\x1b[{_TILDE[key]}~".encode()
    modifier, _, rest = key.rpartition("+")
    if not modifier:
        return None
    if modifier == "ctrl":
        if rest in _CTRL:
            return bytes([_CTRL[rest]])
        if len(rest) == 1 and "a" <= rest <= "z":
            return bytes([ord(rest) & 0x1F])
    if modifier in _MODIFIER:
        parameter = _MODIFIER[modifier]
        if rest in _CURSOR:
            return f"\x1b[1;{parameter}{_CURSOR[rest]}".encode()
        if rest in _TILDE:
            return f"\x1b[{_TILDE[rest]};{parameter}~".encode()
    # Alt+letter is limited by Textual's input parser, which reads `ESC <letter>` itself: `ESC b` arrives as
    # ctrl+left, `ESC DEL` as ctrl+w and `ESC x` as no key event at all, so those never get here as alt+<key>.
    if modifier == "alt":  # Meta sends the key with an Escape in front
        inner = encode(rest, rest if len(rest) == 1 else None, app_cursor=app_cursor)
        return None if inner is None else b"\x1b" + inner
    return None
