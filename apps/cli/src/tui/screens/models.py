"""The model picker: the installed models with size, parameters and context length. Fail-safe: Escape changes nothing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from core.engine.provider.models import NAME_SHOWN, ModelInfo
from core.engine.text import escape_display
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

GIB = 1024**3
NAME_WIDTH = 40
PARAMS_SHOWN = 12


def _cut(text: str, limit: int) -> str:
    """`text` cut to `limit` characters, then written out with `escape_display`, so a cut never splits an escape."""
    return escape_display(text[:limit]) + ("…" if len(text) > limit else "")


def describe(info: ModelInfo, current: bool) -> Text:
    """One row. A `Text`, never markup: names come from Ollama."""
    name = _cut(info.name, NAME_SHOWN)
    size = f"{info.size / GIB:.1f} GiB" if info.size is not None else "?"
    params = _cut(info.parameters, PARAMS_SHOWN) if info.parameters else "?"
    ctx = f"ctx {info.context_length:,}" if info.context_length else "ctx ?"
    mark = "* " if current else "  "
    return Text(f"{mark}{name:<{NAME_WIDTH}}  {size:>8}  {params:>7}  {ctx}")


class ModelPicker(ModalScreen[str | None]):
    DEFAULT_CSS = """
    ModelPicker { align: center middle; }
    ModelPicker > Vertical {
        width: 90%;
        max-height: 90%;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 0 1;
    }
    ModelPicker OptionList { height: auto; max-height: 20; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "dismiss_none", "Cancel", show=False)]
    AUTO_FOCUS = "OptionList"

    def __init__(self, infos: Sequence[ModelInfo], current: str | None) -> None:
        super().__init__()
        self.infos = list(infos)
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Switch model (Enter to switch, Esc to cancel; * is the current one)")
            yield OptionList(*(Option(describe(i, i.name == self.current), id=str(n)) for n, i in enumerate(self.infos)))

    def on_mount(self) -> None:
        picker = self.query_one(OptionList)
        picker.highlighted = next((n for n, i in enumerate(self.infos) if i.name == self.current), 0)

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self.infos[int(str(event.option.id))].name)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
