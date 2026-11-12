"""The approval dialog. Fail-closed: only an explicit choice allows a call; Escape yields None."""

from __future__ import annotations

from typing import ClassVar

from rich.syntax import Syntax
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from src.agent.approval import ApprovalRequest, Choice, Decision


class ApprovalScreen(ModalScreen[Decision | None]):
    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; }
    ApprovalScreen > Vertical {
        width: 90%;
        max-height: 90%;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 0 1;
    }
    ApprovalScreen #preview { max-height: 20; overflow-y: auto; }
    ApprovalScreen OptionList { height: auto; }
    ApprovalScreen Input { display: none; }
    ApprovalScreen Input.-visible { display: block; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "dismiss_none", "Deny and interrupt", show=False)]
    AUTO_FOCUS = "OptionList"

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request
        preview = request.preview
        self.preview_text = preview.body if preview is not None else ""

    def compose(self) -> ComposeResult:
        request, preview = self.request, self.request.preview
        with Vertical():
            yield Label(Text(f"Allow {preview.title if preview else request.label}?"), id="title")
            if preview is not None:
                body = Syntax(preview.body, "diff", word_wrap=True) if preview.kind == "diff" else Text(preview.body)
                yield Static(body, id="preview")
            options = [Option("Allow once", id="once")]
            if request.allow_always:
                options.append(Option("Always", id="always"))
            options.append(Option("Deny…", id="deny"))
            yield OptionList(*options)
            yield Input(placeholder="Note for the model (optional), Enter to deny", id="note")

    def on_mount(self) -> None:
        # Default to Deny: a stray Enter (or two) must never approve a call.
        options = self.query_one(OptionList)
        options.highlighted = options.option_count - 1

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        match event.option.id:
            case "once":
                self.dismiss(Decision(Choice.ONCE))
            case "always" if self.request.allow_always:
                self.dismiss(Decision(Choice.ALWAYS))
            case "deny":
                note = self.query_one(Input)
                note.add_class("-visible")
                note.focus()

    @on(Input.Submitted)
    def _note_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(Decision(Choice.DENY, event.value.strip()))

    def action_dismiss_none(self) -> None:
        self.dismiss(None)
