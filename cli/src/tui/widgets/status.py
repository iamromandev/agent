"""One-line status bar: mode, model, context use, workspace, activity."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from textual.reactive import reactive
from textual.widgets import Static

from src.engine.compaction import CHARS_PER_TOKEN
from src.engine.events import Phase
from src.engine.mode import Mode
from src.engine.text import escape_display

WARN_AT = 0.8
PHASE_TICK = 0.25  # seconds between timer redraws
MAX_ACTIVITY = 80  # characters of the activity part
RATE_AFTER = 0.5  # seconds of streaming before a tokens-per-second rate is shown


def display_path(path: Path) -> str:
    """The path with the home folder shown as `~`."""
    home = Path.home()
    if path == home:
        return "~"
    if path.is_relative_to(home):
        return f"~/{path.relative_to(home)}"
    return str(path)


def format_elapsed(seconds: float) -> str:
    """`0s`..`59s`, then `m:ss`."""
    whole = int(seconds)
    return f"{whole}s" if whole < 60 else f"{whole // 60}:{whole % 60:02d}"


def format_phase_text(kind: str, detail: str) -> str:
    """The words for a phase kind; `detail` (a tool or hook name) is escaped."""
    shown = escape_display(detail)
    match kind:
        case "waiting":
            return "waiting for model"
        case "writing":
            return "writing reply"
        case "approval":
            return f"awaiting approval: {shown}"
        case "hook":
            return f"running hook {shown}"
        case _:
            return f"running {shown}"


def format_phase(
    kind: str, detail: str, step: int, max_steps: int, elapsed: float, tokens: int | None, rate: int | None
) -> str:
    """`● running bash · step 2/50 · 12s`, plus ` · 48 tok · 21 tok/s` when `tokens` is given (no rate when None)."""
    text = f"● {format_phase_text(kind, detail)} · step {step}/{max_steps} · {format_elapsed(elapsed)}"
    if tokens is not None:
        text += f" · {tokens} tok"
        if rate is not None:
            text += f" · {rate} tok/s"
    return text[:MAX_ACTIVITY]


class StatusBar(Static):
    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    """

    mode: reactive[Mode] = reactive(Mode.BUILD)
    pending_mode: reactive[bool] = reactive(False)  # a toggle that applies from the next request
    model: reactive[str] = reactive("")
    used: reactive[int] = reactive(0)
    estimated: reactive[bool] = reactive(False)  # `used` is a guess (resumed or compacted), shown with a "~"
    window: reactive[int] = reactive(0)
    workspace: reactive[str] = reactive("")
    busy: reactive[bool] = reactive(False)
    compacting: reactive[bool] = reactive(False)  # a summary is being written: slow, and Esc cancels it
    error: reactive[str | None] = reactive(None)
    progress: reactive[str] = reactive("")  # a running tool's live line (generate_image); cleared when it ends

    line: str = ""

    def __init__(self) -> None:
        super().__init__(markup=False)
        self.clock: Callable[[], float] = time.monotonic  # tests replace it
        self._phase: Phase | None = None
        self._since = 0.0
        self._first_text_at = 0.0
        self._chars = 0
        self._tokens: int | None = None

    def on_mount(self) -> None:
        self._redraw()
        self.set_interval(PHASE_TICK, self._tick)

    def set_phase(self, phase: Phase) -> None:
        now = self.clock()
        if phase.kind == "waiting":
            self._chars, self._tokens = 0, None
        if phase.kind == "writing":
            self._first_text_at = now
        self._phase, self._since = phase, now
        self._redraw()

    def add_reply_chars(self, n: int) -> None:
        self._chars += n  # no redraw: the timer picks it up

    def set_reply_tokens(self, n: int) -> None:
        self._tokens = n  # the model's own count, from Done

    def clear_phase(self) -> None:
        self._reset_phase()
        self._redraw()

    def _reset_phase(self) -> None:
        self._phase, self._chars, self._tokens = None, 0, None

    def _tick(self) -> None:
        if self._phase is not None and self._build_line() != self.line:
            self._redraw()

    def _phase_text(self, phase: Phase) -> str:
        now = self.clock()
        tokens = rate = None
        if phase.kind == "writing":
            tokens = self._tokens if self._tokens is not None else self._chars // CHARS_PER_TOKEN
            streamed = now - self._first_text_at
            rate = round(tokens / streamed) if streamed >= RATE_AFTER else None
        return format_phase(phase.kind, phase.detail, phase.step, phase.max_steps, now - self._since, tokens, rate)

    def _redraw(self) -> None:
        self.line = self._build_line()
        self.update(self.line)

    def _build_line(self) -> str:
        mode = self.mode.value.upper() + (" (next request)" if self.pending_mode else "")
        parts = [mode, self.model, f"ctx {'~' if self.estimated else ''}{self.used} / {self.window}", self.workspace]
        if self.compacting:
            parts.append("◐ compacting…")
        elif self.progress:
            parts.append(f"◐ {self.progress[:80]}")
        elif self.busy and self._phase is not None:
            parts.append(self._phase_text(self._phase))
        elif self.busy:
            parts.append("● working")
        if self.error:
            parts.append(f"✗ {self.error}")
        if self.window and self.used / self.window >= WARN_AT:
            parts.append(f"⚠ context {round(100 * self.used / self.window)}% — /clear")
        return " · ".join(parts)

    def _watch_all(self, *_: object) -> None:
        self._redraw()

    def watch_busy(self, busy: bool) -> None:
        if not busy:
            self.progress = ""  # nothing runs any more: a line left by an ended call must not stay
            self._reset_phase()
        self._redraw()

    watch_mode = watch_pending_mode = watch_model = watch_used = _watch_all
    watch_window = watch_workspace = watch_error = watch_estimated = watch_compacting = watch_progress = _watch_all
