"""The Textual app: layout, keys, the turn worker and the slash commands."""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches

from src.agent import (
    Agent,
    AgentBusy,
    AgentEvent,
    AssistantText,
    Notice,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)
from src.agent.approval import Approver
from src.agent.mode import Mode
from src.agent.provider.base import Provider
from src.tui.approver import TuiApprover
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar, display_path
from src.tui.widgets.transcript import Transcript


async def _uninterrupted(work: Awaitable[None]) -> None:
    """Run `work` to the end even if this task is cancelled meanwhile; the cancel is re-raised after."""
    task = asyncio.ensure_future(work)
    cancel: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancel = cancel or exc  # a repeated cancel must not abandon the work
        except Exception:
            break  # the task holds it; read below
    task.result()
    if cancel is not None:
        raise cancel


HELP = (
    "Enter send · ctrl+j / shift+enter new line · Tab switch BUILD/PLAN · Esc interrupt · "
    "ctrl+c interrupt (quit when idle) · /clear new conversation · /help this help · /quit exit"
)


class AgentApp(App[None]):
    CSS = """
    #transcript { height: 1fr; }
    ChatInput { height: auto; max-height: 10; min-height: 3; }
    """
    ENABLE_COMMAND_PALETTE = False  # it would open over a pending approval dialog
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("tab", "toggle_mode", "Mode", priority=True),
        Binding("escape", "interrupt_turn", "Interrupt", priority=True),
        Binding("ctrl+c", "interrupt_or_quit", "Interrupt / quit", priority=True, show=False),
    ]

    def __init__(
        self,
        agent_factory: Callable[[Approver], Agent],
        provider: Provider,
        workspace: Path,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self._on_close = on_close
        self.provider = provider
        self.workspace = workspace
        self.agent = agent_factory(TuiApprover(self))
        self._shutting_down = False
        self._turn_active = False
        self._turn_entered = False  # the turn worker has run its first line
        self._draft: str | None = None  # the submitted message until the turn really starts sending it
        self._finished_since_toggle = False  # a tool finished after a mid-turn mode toggle

    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")
        yield ChatInput(id="input")
        yield StatusBar()

    def on_mount(self) -> None:
        status = self.query_one(StatusBar)
        status.model = self.provider.model
        status.window = self.provider.context_window
        status.workspace = display_path(self.workspace)
        status.mode = self.agent.mode
        self.query_one(ChatInput).focus()
        self.run_worker(self._run_check(), group="check")

    async def _run_check(self) -> bool:
        """Ask the provider whether it is ready. Shows the problem; returns True when ready."""
        status = self.query_one(StatusBar)
        try:
            problem = await self.provider.check()
        except Exception as exc:  # the contract says it returns a message; a bug must not kill the app
            problem = f"Provider check failed: {exc}"
        if problem is None:
            status.error = None
            return True
        status.error = "not ready"
        await self.query_one(Transcript).add_notice("error", problem)
        return False

    def exit(self, *args: Any, **kwargs: Any) -> None:
        self._shutting_down = True  # from here on the widgets are going away: stop applying agent events
        super().exit(*args, **kwargs)

    async def on_unmount(self) -> None:
        self._shutting_down = True
        if self._on_close is not None:
            await self._on_close()

    # keys

    def action_toggle_mode(self) -> None:
        self.agent.set_mode(Mode.PLAN if self.agent.mode is Mode.BUILD else Mode.BUILD)
        status = self.query_one(StatusBar)
        status.mode = self.agent.mode
        if self._turn_active:  # a step already running keeps the mode it started with
            self._finished_since_toggle = False
            status.pending_mode = True
        else:
            status.pending_mode = False

    def action_interrupt_turn(self) -> None:
        self.interrupt_turn()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        stack = self.screen_stack  # empty while shutting down; `self.screen` would raise then
        if action == "interrupt_turn":  # with the approval dialog open Escape belongs to it: it denies, then interrupts
            return not (stack and isinstance(stack[-1], ApprovalScreen))
        if action == "toggle_mode":  # the mode is not changed under a pending approval, nor Tab taken from the dialog
            return not any(isinstance(screen, ApprovalScreen) for screen in stack)
        return True

    def action_interrupt_or_quit(self) -> None:
        if self._turn_active:
            self.interrupt_turn()
        else:
            self.exit()

    def interrupt_turn(self) -> None:
        self.workers.cancel_group(self, "turn")
        if self._turn_active and not self._turn_entered:
            # Cancelled before its first line ran: the `finally` in `_turn` never will.
            self._end_turn()

    def _restore_draft(self, text: str) -> None:
        """Put a message that was not sent back in the input, ahead of anything typed since."""
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            chat = self.query_one(ChatInput)
            chat.load_text(f"{text}\n{chat.text}" if chat.text else text)
            chat.move_cursor(chat.document.end)

    def _end_turn(self) -> None:
        self._turn_active = False
        if self._draft is not None:  # the turn ended before it sent the message
            draft, self._draft = self._draft, None
            self._restore_draft(draft)
        with contextlib.suppress(NoMatches):  # the status bar is gone when the app is shutting down
            self.query_one(StatusBar).busy = False

    # turns

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text
        if text.strip().startswith("/"):
            await self._command(text.strip())
            return
        if self._turn_active:
            self._restore_draft(text)  # not accepted: the message stays in the box
            await self.query_one(Transcript).add_notice("warn", "busy — press Esc to interrupt")
            return
        self._turn_active = True
        self._draft = text
        self.query_one(StatusBar).busy = True
        self._turn_entered = False
        # A partial, not a coroutine: cancelled before it first runs, nothing is left un-awaited.
        self.run_worker(functools.partial(self._turn, event.text), group="turn")

    async def _command(self, command: str) -> None:
        transcript = self.query_one(Transcript)
        name = command.split()[0]
        match name:
            case "/help":
                await transcript.add_notice("info", HELP)
            case "/quit":
                self.exit()
            case "/clear":
                # Refused while a turn runs: the history belongs to the running turn until it ends.
                try:
                    if self._turn_active:
                        raise AgentBusy
                    self.agent.clear()
                except AgentBusy:
                    await transcript.add_notice("warn", "busy — press Esc to interrupt, then /clear")
                    return
                await transcript.clear_all()
                self.query_one(StatusBar).used = 0
            case _:
                await transcript.add_notice("warn", f"Unknown command: {name} (try /help)")

    async def _turn(self, text: str) -> None:
        self._turn_entered = True
        transcript = self.query_one(Transcript)
        status = self.query_one(StatusBar)
        try:
            if status.error and not await self._run_check():  # R3: re-check before giving up on the model
                return
            self._draft = None  # from here the message is part of the conversation
            await transcript.add_user(text)
            self._finished_since_toggle = False
            status.pending_mode = False
            # The loop's sink is synchronous; a consumer task applies the events to the widgets in order.
            events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
            consumer = asyncio.create_task(self._consume(events))
            try:
                await self.agent.run_turn(text, events.put_nowait)
            except AgentBusy:
                events.put_nowait(Notice("warn", "busy — a turn is already running"))
            finally:
                # Whatever ended the turn, the queued events still reach the transcript, and a
                # repeated Esc must not cut that short.
                await _uninterrupted(self._finish_turn(events, consumer))
        finally:
            self._end_turn()

    async def _finish_turn(self, events: asyncio.Queue[AgentEvent | None], consumer: asyncio.Task[None]) -> None:
        events.put_nowait(None)
        await consumer
        if self._shutting_down:
            return
        transcript = self.query_one(Transcript)
        await transcript.end_assistant()
        await transcript.close_open_tools("ended")

    async def _consume(self, events: asyncio.Queue[AgentEvent | None]) -> None:
        while (event := await events.get()) is not None:
            if self._shutting_down:
                continue  # shutting down: drain the queue without touching the torn-down widgets
            try:
                await self._on_agent_event(event)
            except Exception as exc:  # one bad event must not drop the ones behind it
                with contextlib.suppress(Exception):
                    await self.query_one(Transcript).add_notice("error", f"Display error: {exc}")

    async def _on_agent_event(self, event: AgentEvent) -> None:
        transcript = self.query_one(Transcript)
        status = self.query_one(StatusBar)
        match event:
            case AssistantText(delta):
                if self._finished_since_toggle:  # text after a tool result: a new request has started
                    status.pending_mode = False
                await transcript.write_assistant(delta)
            case ToolStarted(call_id, label):
                await transcript.add_tool(call_id, label)
            case ToolFinished(call_id, result, preview, applied):
                self._finished_since_toggle = True
                if (entry := transcript.tool(call_id)) is not None:
                    await entry.finish(result, preview, applied)
            case Notice(level, text):
                await transcript.add_notice(level, text)
            case UsageUpdate(used, window):
                status.used = used
                status.window = window
            case TurnEnded(turn_status):
                status.pending_mode = False
                await transcript.end_assistant()
                await transcript.close_open_tools(turn_status.replace("_", " "))
                if turn_status == "interrupted":
                    await transcript.add_notice("info", "interrupted", icon="⏹")
