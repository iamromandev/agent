"""The session picker: this folder's saved sessions, newest first. Fail-safe: Escape changes nothing."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from src.agent.session import SessionInfo

TITLE_WIDTH = 40
RECENT_DAYS = 30


def relative(moment: datetime, now: datetime | None = None) -> str:
    """How long ago `moment` was, in words; an ISO date beyond 30 days. Read in the clock of `now`."""
    now = now or datetime.now().astimezone()
    moment = moment.astimezone(now.tzinfo)
    seconds = (now - moment).total_seconds()
    if seconds < 60:  # includes a clock a little ahead
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    days = (now.date() - moment.date()).days
    if days == 0:
        return f"today {moment:%H:%M}"
    if days == 1:
        return "yesterday"
    if days <= RECENT_DAYS:
        return f"{days} days ago"
    return f"{moment:%Y-%m-%d}"


def describe(info: SessionInfo, now: datetime | None = None) -> Text:
    """One row. A `Text`, never markup: titles come from what the user typed."""
    return Text(f"{info.title:<{TITLE_WIDTH}.{TITLE_WIDTH}}  {relative(info.updated_at, now)}  {info.message_count:>3}")


class SessionPicker(ModalScreen[str | None]):
    DEFAULT_CSS = """
    SessionPicker { align: center middle; }
    SessionPicker > Vertical {
        width: 90%;
        max-height: 90%;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 0 1;
    }
    SessionPicker OptionList { height: auto; max-height: 20; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "dismiss_none", "Cancel", show=False)]
    AUTO_FOCUS = "OptionList"

    def __init__(self, infos: Sequence[SessionInfo]) -> None:
        super().__init__()
        self.infos = list(infos)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Resume a session (Enter to resume, Esc to cancel)")
            yield OptionList(*(Option(describe(info), id=info.id) for info in self.infos))

    def on_mount(self) -> None:
        self.query_one(OptionList).highlighted = 0

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
