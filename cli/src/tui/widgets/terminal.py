"""The terminal pane: a tab per terminal, each a `TerminalView` that draws the pty's screen and sends it keys."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import TabbedContent, TabPane

from src.agent.terminal.manager import Event, Terminal, TerminalManager
from src.agent.terminal.pty import WriteOverflow
from src.tui.keys import encode
from src.tui.terminal_render import line_strip
from src.tui.terminal_screen import TerminalScreen, TerminalStream

MIN_COLS, MIN_ROWS = 10, 3  # a terminal smaller than this is of no use; a tiny pane still gets this
REFRESH = 1 / 30  # the screen is redrawn at most this often, however fast the process prints
FEED_SLICE = 4_096  # bytes given to the emulator at once (a few ms of pyte): the app never waits longer
FEED_BUDGET = 0.008  # seconds of emulator work per tick (a quarter of it), however much output is waiting
# Bytes waiting for the emulator; beyond that the oldest are skipped (the screen is behind). Limits of skipping:
# the bytes after the cut are trimmed to the first Escape or line end, the parser and the colors/charset start
# afresh, but modes the skipped bytes set (alternate screen, cursor keys, bracketed paste, scroll region) stay as
# they were, so a full-screen program that switched in or out of the alternate screen right there is drawn
# wrongly until it redraws. Answers to the program's questions (cursor position, device attributes) wait behind
# the backlog like everything else, and one that was skipped is lost: a program waiting for it may stall.
MAX_PENDING = 256 * 1024
_START = (80, 24)  # the size a terminal starts with (the manager's default) until the view is laid out
_PASTE_START, _PASTE_END = b"\x1b[200~", b"\x1b[201~"
_RESTART = frozenset(b"\x1b\r\n")  # where output can be taken up again after a gap: a new sequence or a new line


class TerminalView(Widget, can_focus=True):
    """Draws one terminal's screen and forwards keys and pastes to it.

    It listens to the terminal from its creation, not from its mount, so no output is lost while it is
    being mounted. It stops listening when it is removed.
    """

    DEFAULT_CSS = """
    TerminalView { width: 1fr; height: 1fr; }
    """

    def __init__(self, terminal: Terminal) -> None:
        super().__init__()
        self.terminal = terminal
        self.emulator = TerminalScreen(*_START)
        self.emulator.reply = self._answer
        self._stream = TerminalStream(self.emulator)
        self._dirty = True
        self._pending: deque[bytes] = deque()  # output the emulator has not seen yet: fed a slice per tick
        self._waiting = 0  # the bytes in `_pending`
        self._gap = False  # output was skipped and no restart point has come since
        self._sent: tuple[int, int] | None = None  # the size the process was told last
        self._unsubscribe = terminal.subscribe(self._feed)

    @property
    def pending(self) -> int:
        """Bytes of output the screen has not caught up with."""
        return self._waiting

    def _feed(self, data: bytes) -> None:
        """Queue output for the emulator. Parsing it here would hold the event loop for as long as a flood
        lasts (pyte is pure Python: 64 KiB take ~50 ms), so the tick feeds it within a time budget."""
        self._pending.append(data)
        self._waiting += len(data)
        if self._gap:  # still inside the output after a gap: nothing is drawn before a restart point
            self._skip_to_restart()
        skipped = False
        while self._waiting > MAX_PENDING and len(self._pending) > 1:  # far behind: show the newest, not all
            self._waiting -= len(self._pending.popleft())
            skipped = True
        if skipped:
            self._resync()
        self._dirty = True

    def _resync(self) -> None:
        """Make what follows a gap in the output safe to draw: the cut may have left the parser inside a sequence,
        the output after it the tail of one, and colors or a charset the skipped bytes changed."""
        self._stream = TerminalStream(self.emulator)
        self.emulator.restart_rendition()
        self._gap = True
        self._skip_to_restart()

    def _skip_to_restart(self) -> None:
        """Drop queued bytes up to the first Escape or line end (they may be the end of a cut sequence); until
        one arrives, `_gap` stays set and the output that comes is dropped the same way."""
        while self._pending:
            head = self._pending[0]
            cut = next((i for i, byte in enumerate(head) if byte in _RESTART), None)
            if cut is None:
                self._waiting -= len(self._pending.popleft())
                continue
            if cut:
                self._pending[0] = head[cut:]
                self._waiting -= cut
            self._gap = False
            return

    def _drain(self, budget: float = FEED_BUDGET) -> None:
        """Give the emulator waiting output until `budget` seconds are used; at least one slice."""
        deadline = time.perf_counter() + budget
        while self._pending:
            chunk = self._pending.popleft()
            piece, rest = chunk[:FEED_SLICE], chunk[FEED_SLICE:]
            if rest:
                self._pending.appendleft(rest)
            self._waiting -= len(piece)
            try:
                self._stream.feed(piece)  # skips a sequence the emulator cannot take, and goes on
            finally:
                self._dirty = True
            if time.perf_counter() >= deadline:
                break

    def release(self) -> None:
        """Stop listening to the terminal (also done when the view is removed)."""
        self._unsubscribe()
        self._pending.clear()
        self._waiting = 0

    def _send(self, data: bytes) -> None:
        try:
            self.terminal.write(data)
        except WriteOverflow:  # the process reads nothing: the input is dropped
            self.app.bell()

    def _answer(self, text: str) -> None:
        """The emulator answers a question the program asked (where is the cursor, ...)."""
        with contextlib.suppress(WriteOverflow):
            self.terminal.write(text.encode())

    def on_mount(self) -> None:
        self.set_interval(REFRESH, self._tick)

    def on_unmount(self) -> None:
        self.release()

    def _tick(self) -> None:
        # A view mounted while the pane is hidden gets no resize when the pane is shown: catch up here.
        if self.size.width > 0 and self.size.height > 0:
            self._fit(self.size.width, self.size.height)
        self._drain()
        if self._dirty:
            self._dirty = False
            self.refresh()

    def render_line(self, y: int) -> Strip:
        return line_strip(self.emulator, y, cursor=self.has_focus).adjust_cell_length(self.size.width)

    def on_focus(self) -> None:
        self.refresh()  # the cursor shows only while the view has focus

    def on_blur(self) -> None:
        self.refresh()

    def on_resize(self, event: events.Resize) -> None:
        if event.size.width > 0 and event.size.height > 0:  # not shown otherwise
            self._fit(event.size.width, event.size.height)

    def _fit(self, width: int, height: int) -> None:
        """Give the emulator and the process the size of the view (at least the minimum), if it changed."""
        cols, rows = max(MIN_COLS, width), max(MIN_ROWS, height)
        if (cols, rows) == self._sent:
            return
        self._sent = (cols, rows)
        self.emulator.resize(rows, cols)
        self.terminal.resize(cols, rows)
        self._dirty = True

    def on_key(self, event: events.Key) -> None:
        data = encode(event.key, event.character, app_cursor=self.emulator.application_cursor)
        if data is None:  # no terminal has an encoding for it: the app may have a use
            return
        event.stop()
        event.prevent_default()
        self._send(data)

    def on_paste(self, event: events.Paste) -> None:
        event.stop()
        text = _clean(event.text).encode()
        if self.emulator.bracketed_paste:
            self._send(_PASTE_START + text + _PASTE_END)
        else:
            self._send(text.replace(b"\r\n", b"\n").replace(b"\n", b"\r"))  # Enter is a carriage return


def _clean(text: str) -> str:
    """Pasted text without Escape (it could end a bracketed paste early and run the rest as typed) and other
    control characters; Tab and newlines stay."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(c for c in text if c in "\t\n" or not (ord(c) < 0x20 or 0x7F <= ord(c) <= 0x9F))


@dataclass
class _Tab:
    view: TerminalView
    pane_id: str
    announce: bool = True  # tell the app when it is added while the pane is hidden


def _title(name: str, terminal: Terminal) -> str:
    code = terminal.status[1]
    if code is None:
        return name
    if code >= 0:
        return f"{name} ✗ exited {code}"
    try:
        return f"{name} ✗ {signal.Signals(-code).name}"
    except ValueError:  # a signal without a name
        return f"{name} ✗ signal {-code}"


class TerminalPane(Vertical):
    """One tab per terminal of the manager, hidden until shown. Follows the manager's events."""

    DEFAULT_CSS = """
    TerminalPane { height: 40%; display: none; }
    TerminalPane TabbedContent { height: 1fr; }
    TerminalPane ContentSwitcher { height: 1fr; }
    TerminalPane TabPane { height: 1fr; padding: 0; }
    """

    class StartedHidden(Message):
        """A terminal got its tab while the pane was hidden (the model started it): the app may say so."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    def __init__(self, manager: TerminalManager) -> None:
        super().__init__()
        self._manager = manager
        self._tabs: dict[str, _Tab] = {}
        self._count = 0
        self._tail: asyncio.Task[None] | None = None  # the last queued change of the tabs: they apply in order
        self._closing = False

    def compose(self) -> ComposeResult:
        yield TabbedContent()

    def on_mount(self) -> None:
        for name in self._manager.names():
            self._started(name, announce=False)  # there before the pane: not news
        self._manager.listen(self._on_event)

    def on_unmount(self) -> None:
        self._closing = True  # the manager may still report terminals closing: nothing is left to show them in
        for tab in self._tabs.values():
            tab.view.release()  # a view that was never mounted would otherwise listen for ever

    def view(self, name: str) -> TerminalView | None:
        tab = self._tabs.get(name)
        return tab.view if tab else None

    def title(self, name: str) -> str | None:
        """The label of the tab of terminal `name` as it is shown (None: it has no tab, or not yet)."""
        tab = self._tabs.get(name)
        if tab is None:
            return None
        try:
            return self.query_one(TabbedContent).get_tab(tab.pane_id).label.plain
        except Exception:  # the tab is not mounted yet
            return None

    @property
    def active_name(self) -> str | None:
        active = self.query_one(TabbedContent).active
        return next((name for name, tab in self._tabs.items() if tab.pane_id == active), None)

    def active_view(self) -> TerminalView | None:
        return self.view(name) if (name := self.active_name) else None

    async def settled(self) -> None:
        """Wait until every change of the tabs reported so far is shown."""
        while (tail := self._tail) is not None and not tail.done():
            await asyncio.wait([tail])

    async def focus_active(self) -> None:
        await self.settled()
        if (view := self.active_view()) is not None:
            # Now, not `view.focus()` (deferred): the caller may rely on where focus is when this returns, e.g. a
            # tab close that drops its guard while focus would still be on the tab bar, where ctrl+w closes any tab.
            self.screen.set_focus(view)

    def _on_event(self, name: str, event: Event) -> None:
        if self._closing:
            return
        if event == "started":
            self._started(name, announce=True)
        elif event == "exited":
            if (tab := self._tabs.get(name)) is not None:
                self._enqueue(self._retitle, name, tab)
        elif (tab := self._tabs.pop(name, None)) is not None:  # closed
            self._enqueue(self._remove, tab)

    def _started(self, name: str, *, announce: bool) -> None:
        terminal = self._manager.get(name)
        if terminal is not None:
            self._count += 1
            tab = _Tab(TerminalView(terminal), f"terminal-{self._count}", announce)  # subscribed from here on
            self._tabs[name] = tab
            self._enqueue(self._add, name, tab, undo=tab.view.release)

    def _enqueue(self, change, *args: object, undo: Callable[[], None] | None = None) -> None:
        """Queue a change of the tabs; `undo` runs if it is not made (the pane is closing, or it failed)."""
        previous = self._tail

        async def run() -> None:
            if previous is not None:
                await asyncio.wait([previous])
            if self._closing:
                if undo is not None:
                    undo()
                return
            try:
                await change(*args)
            except Exception as exc:
                if undo is not None:
                    undo()
                if not self._closing:
                    self.app.notify(f"terminal pane: {exc}", severity="error")

        self._tail = asyncio.get_running_loop().create_task(run())

    async def _add(self, name: str, tab: _Tab) -> None:
        tabs = self.query_one(TabbedContent)
        typing = isinstance(self.app.focused, TerminalView)
        await tabs.add_pane(TabPane(_title(name, tab.view.terminal), tab.view, id=tab.pane_id))
        if not typing:  # a tab does not take the screen from the terminal the user is typing in
            tabs.active = tab.pane_id
        if tab.announce and not self.display:  # opening the pane for a shell shows it before this runs
            self.post_message(self.StartedHidden(name))

    @property
    def tab_count(self) -> int:
        return len(self._tabs)

    async def switch(self, step: int) -> None:
        """Select the tab `step` places on (negative: back), wrapping around. Focus follows when the user was
        typing in a terminal, so what they type goes to the tab they see."""
        await self.settled()
        names = list(self._tabs)
        active = self.active_name
        if len(names) < 2 or active is None:
            return
        typing = isinstance(self.app.focused, TerminalView)
        target = self._tabs[names[(names.index(active) + step) % len(names)]]
        self.query_one(TabbedContent).active = target.pane_id
        if typing:
            self.app.call_after_refresh(target.view.focus)

    async def _retitle(self, name: str, tab: _Tab) -> None:
        if self._tabs.get(name) is tab:
            self.query_one(TabbedContent).get_tab(tab.pane_id).label = _title(name, tab.view.terminal)

    async def _remove(self, tab: _Tab) -> None:
        await self.query_one(TabbedContent).remove_pane(tab.pane_id)
