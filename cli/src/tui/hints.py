"""The wording of the key-hints line under the input. Pure: no Textual, no IO."""

from __future__ import annotations

MIN_ROWS = 20  # the line hides below this window height

IDLE = ("enter send", "ctrl+j newline", "tab mode", "/ commands", "@ files", "ctrl+t terminal")
BUSY = ("esc interrupt", "enter queue message", "tab mode", "ctrl+t terminal")
SUGGESTIONS = ("↑↓ select", "tab complete", "esc close")
TERMINAL = ("ctrl+t chat", "ctrl+pageup/pagedown tabs")
SEP = " · "


def hint_text(*, busy: bool, popup_open: bool, terminal_focused: bool, has_terminals: bool) -> str:
    """Precedence: terminal focused, suggestion list open, busy, idle. Without terminals the `ctrl+t` entries are
    left out and a focused terminal (impossible then) falls through to the next state."""
    if terminal_focused and has_terminals:
        entries = TERMINAL
    elif popup_open:
        entries = SUGGESTIONS
    else:
        entries = BUSY if busy else IDLE
    if not has_terminals:
        entries = tuple(e for e in entries if not e.startswith("ctrl+t"))
    return SEP.join(entries)
