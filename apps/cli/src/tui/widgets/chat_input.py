"""The message box: Enter sends, ctrl+j or shift+enter adds a line."""

from __future__ import annotations

from typing import Any, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static, TextArea

from src.engine.mentions_index import Suggestion, current_token
from src.tui.widgets.command_suggestions import CommandSuggestions
from src.tui.widgets.path_suggestions import PathSuggestions

POPUP_KEYS = frozenset({"up", "down", "tab", "enter", "escape"})
PLACEHOLDER = "ask anything, @ for files, / for commands"
PROMPT = chr(0x203A) + " "  # the marker: a single right-pointing angle quotation mark


class InputRow(Horizontal):
    """The prompt marker beside the chat input, between a line above and a line below (accent while focus is inside)."""

    DEFAULT_CSS = """
    InputRow {
        height: auto;
        border-top: solid $panel;
        border-bottom: solid $panel;
    }
    InputRow:focus-within {
        border-top: solid $accent;
        border-bottom: solid $accent;
    }
    InputRow > #prompt { width: 2; height: 1; color: $accent; }
    """

    def __init__(self, chat: ChatInput, *, id: str = "input-row") -> None:
        super().__init__(id=id)
        self._chat = chat

    def compose(self) -> ComposeResult:
        yield Static(PROMPT, id="prompt", markup=False)
        yield self._chat


class ChatInput(TextArea):
    DEFAULT_CSS = """
    ChatInput {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 8;
        border: none;
        padding: 0;
        background: transparent;
    }
    ChatInput:focus { border: none; }
    ChatInput .text-area--cursor-line { background: transparent; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+j", "newline", "New line", show=False),
        Binding("shift+enter", "newline", "New line", show=False),
        Binding("ctrl+v", "paste_image", "Paste", show=False),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class PasteImage(Message):
        """ctrl+v: the app looks for an image on the clipboard and falls back to `action_paste`."""

    def __init__(self, text: str = "", **kwargs: Any) -> None:
        super().__init__(text, **kwargs)
        self.suggestions: CommandSuggestions | None = None  # set by the app; its keys are ours while it is open
        self.paths: PathSuggestions | None = None  # same for the `@path` list

    @property
    def popup_open(self) -> bool:
        return (self.suggestions is not None and self.suggestions.is_open) or (
            self.paths is not None and self.paths.is_open
        )

    def token_before_cursor(self) -> str | None:
        row, col = self.cursor_location
        return current_token(self.document.get_line(row)[:col])

    def complete_path(self, choice: Suggestion) -> None:
        """Replace the `@token` at the cursor by the chosen path. A file ends with a space; a folder ends with `/` and
        the list stays open for the next segment. A path with whitespace is quoted (a folder leaves the quote open)."""
        token = self.token_before_cursor()
        if token is None:
            return
        row, col = self.cursor_location
        quoted = any(c.isspace() for c in choice.path)
        if choice.folder:
            text = f'@"{choice.path}' if quoted else f"@{choice.path}"
        else:
            text = f'@"{choice.path}" ' if quoted else f"@{choice.path} "
        self.replace(text, (row, col - len(token) - 1), (row, col))
        if not choice.folder and self.paths is not None:
            self.paths.hide()

    def _typed_file(self, key: str) -> bool:
        """Enter on a file whose whole path is typed: nothing is left to complete, so it sends (the list closes)."""
        paths = self.paths
        if key != "enter" or paths is None or (choice := paths.chosen()) is None or choice.folder:
            return False
        token = self.token_before_cursor()
        if token is None or token.removeprefix('"') != choice.path:
            return False
        paths.hide()
        return True

    def complete(self, name: str) -> None:
        """Put `/name ` in the box, ready for arguments."""
        self.load_text(f"/{name} ")
        self.move_cursor(self.document.end)
        if self.suggestions is not None:
            self.suggestions.hide()

    def action_paste_image(self) -> None:
        self.post_message(self.PasteImage())

    def action_newline(self) -> None:
        self.insert("\n")

    async def _on_key(self, event: events.Key) -> None:
        paths = self.paths
        if paths is not None and paths.is_open and event.key in POPUP_KEYS and not self._typed_file(event.key):
            event.stop()
            event.prevent_default()
            match event.key:
                case "up" | "down":
                    paths.move(-1 if event.key == "up" else 1)
                case "tab" | "enter":
                    if (choice := paths.chosen()) is not None:
                        self.complete_path(choice)
                    else:
                        paths.hide()
                case _:  # escape
                    paths.hide()
            return
        popup = self.suggestions
        if popup is not None and popup.is_open and event.key == "enter" and self.text == f"/{popup.chosen()}":
            popup.hide()  # the whole name is typed: Enter sends it, as it always did
        elif popup is not None and popup.is_open and event.key in POPUP_KEYS:
            event.stop()
            event.prevent_default()
            match event.key:
                case "up" | "down":
                    popup.move(-1 if event.key == "up" else 1)
                case "tab" | "enter":
                    if (name := popup.chosen()) is not None:
                        self.complete(name)
                    popup.hide()
                case _:  # escape
                    popup.hide()
            return
        if event.key == "enter":  # TextArea would insert a newline
            event.stop()
            event.prevent_default()
            text = self.text
            if text.strip():
                self.clear()
                self.post_message(self.Submitted(text))
            return
        await super()._on_key(event)
