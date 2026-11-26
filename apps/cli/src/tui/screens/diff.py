"""The viewer of the last turn's changes: a file list and the coloured unified diff of the selected file.

Read-only and fail-safe: Escape or `q` dismisses it, nothing is written to the transcript or the session. Every string
shown comes from the workspace, so it is a `rich.text.Text` built from `escape_display`ed pieces, never markup."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import ClassVar

from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Label, OptionList, Static
from textual.widgets.option_list import Option

from src.engine.text import escape, escape_display
from src.engine.turn_diff import BANNER, DiffItem, FileDiff, LineKind, TurnDiff, render_lines
from src.tui.styles import DIM, ERROR, MODIFIED, OK, PLAIN, TYPE_CHANGED

Loader = Callable[[TurnDiff, DiffItem], Awaitable[FileDiff]]
STYLES: dict[LineKind, Style] = {"add": OK, "del": ERROR, "hunk": DIM, "meta": DIM, "ctx": PLAIN}
STATUS_STYLES = {"A": OK, "D": ERROR, "M": MODIFIED, "T": TYPE_CHANGED}
MORE_ID = "more"
MINUS = "\u2212"  # the minus sign, not a hyphen: the counts line up with `+`
OFF_LOOP_LINES = 200  # a diff with more lines is turned into text in a thread


def describe(item: DiffItem) -> Text:
    """One list row: status letter, escaped path, the counts that are not zero (`?` when git could not count and
    nothing else says why) and the note."""
    row = Text()
    row.append(item.status, STATUS_STYLES[item.status])
    row.append(f" {escape_display(item.path)}")
    counts = []
    if item.added:
        counts.append(f"+{item.added}")
    if item.removed:
        counts.append(f"{MINUS}{item.removed}")
    if item.added is None and item.removed is None and not item.note:
        counts.append("?")
    if counts:
        row.append(f" {' '.join(counts)}", DIM)
    if item.note:
        row.append(f" ({escape_display(item.note)})", DIM)
    return row


def body(file_diff: FileDiff) -> Text:
    """The diff pane's text."""
    text = Text(no_wrap=False, overflow="fold")
    for line in render_lines(file_diff):
        text.append(line.text + "\n", STYLES[line.kind])
    if not text.plain:
        text.append("(no content changes)", DIM)
    return text


class DiffScreen(Screen[None]):
    DEFAULT_CSS = """
    DiffScreen { layout: vertical; }
    DiffScreen #banner { height: auto; color: $warning; padding: 0 1; }
    DiffScreen #hint { height: 1; color: $text-muted; padding: 0 1; }
    DiffScreen Horizontal { height: 1fr; }
    DiffScreen #files { width: 40%; border-right: solid $accent; }
    DiffScreen #pane { width: 60%; padding: 0 1; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss", "Close", show=False),
        Binding("q", "dismiss", "Close", show=False),
        Binding("pageup", "scroll_diff(-1)", "Scroll up", show=False, priority=True),
        Binding("pagedown", "scroll_diff(1)", "Scroll down", show=False, priority=True),
    ]
    AUTO_FOCUS = "#files"

    def __init__(self, diff: TurnDiff, loader: Loader) -> None:
        super().__init__()
        self.diff = diff
        self._loader = loader

    def compose(self) -> ComposeResult:
        if self.diff.fallback:
            yield Label(Text(BANNER), id="banner")
        options = [Option(describe(item), id=str(n)) for n, item in enumerate(self.diff.items)]
        if self.diff.more:
            options.append(Option(Text(f"and {self.diff.more} more files", style=DIM), id=MORE_ID, disabled=True))
        with Horizontal():
            yield OptionList(*options, id="files")
            with VerticalScroll(id="pane"):
                yield Static(Text(""), id="diff")
        yield Label(Text("Up/Down file · PageUp/PageDown scroll · Tab switch focus · Esc or q close"), id="hint")

    def on_mount(self) -> None:
        self.query_one(OptionList).highlighted = 0 if self.diff.items else None
        self.query_one("#pane").can_focus = True

    @on(OptionList.OptionHighlighted)
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        if event.option.id is None or event.option.id == MORE_ID:
            return
        item = self.diff.items[int(event.option.id)]
        self._show(Text("loading…", style=DIM))
        self.run_worker(self._load(item), group="diff-file", exclusive=True, exit_on_error=False)

    async def _load(self, item: DiffItem) -> None:
        try:
            loaded = await self._loader(self.diff, item)
            # Up to 5,000 lines of up to 2,000 characters: turned into text away from the event loop when it is big.
            content = await asyncio.to_thread(body, loaded) if len(loaded.lines) > OFF_LOOP_LINES else body(loaded)
        except Exception as error:  # git failed, the store is gone: shown in the pane, never raised into the app
            content = Text(f"Could not load this file: {escape(str(error)[:200])}", style=ERROR)
        self._show(content)

    def _show(self, content: Text) -> None:
        try:
            self.query_one("#diff", Static).update(content)
        except NoMatches:  # the screen is being removed
            return

    def action_scroll_diff(self, direction: int) -> None:
        pane = self.query_one("#pane", VerticalScroll)
        pane.scroll_page_down() if direction > 0 else pane.scroll_page_up()
