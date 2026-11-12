"""One-line status bar: mode, model, context use, workspace, activity."""

from __future__ import annotations

from pathlib import Path

from textual.reactive import reactive
from textual.widgets import Static

from src.agent.mode import Mode

WARN_AT = 0.8


def display_path(path: Path) -> str:
    """The path with the home folder shown as `~`."""
    home = Path.home()
    if path == home:
        return "~"
    if path.is_relative_to(home):
        return f"~/{path.relative_to(home)}"
    return str(path)


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

    line: str = ""

    def __init__(self) -> None:
        super().__init__(markup=False)

    def on_mount(self) -> None:
        self._redraw()

    def _redraw(self) -> None:
        mode = self.mode.value.upper() + (" (next request)" if self.pending_mode else "")
        parts = [mode, self.model, f"ctx {'~' if self.estimated else ''}{self.used} / {self.window}", self.workspace]
        if self.compacting:
            parts.append("◐ compacting…")
        elif self.busy:
            parts.append("● working")
        if self.error:
            parts.append(f"✗ {self.error}")
        if self.window and self.used / self.window >= WARN_AT:
            parts.append(f"⚠ context {round(100 * self.used / self.window)}% — /clear")
        self.line = " · ".join(parts)
        self.update(self.line)

    def _watch_all(self, *_: object) -> None:
        self._redraw()

    watch_mode = watch_pending_mode = watch_model = watch_used = _watch_all
    watch_window = watch_workspace = watch_busy = watch_error = watch_estimated = watch_compacting = _watch_all
