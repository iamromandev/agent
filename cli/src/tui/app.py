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
    Compacting,
    EventSink,
    Notice,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)
from src.agent.approval import Approver
from src.agent.compaction import CHARS_PER_TOKEN
from src.agent.facts import FACTS_SHARE, init_check, render_facts, scan
from src.agent.instructions import fingerprint, levels
from src.agent.loop import close_recorder
from src.agent.mode import Mode
from src.agent.prompt import build_init_prompt
from src.agent.provider.base import Provider
from src.agent.session import (
    FileRecorder,
    NullRecorder,
    Recorder,
    SessionError,
    SessionInfo,
    SessionStore,
    SessionWriteError,
)
from src.tui import NEW, StartChoice
from src.tui.approver import TuiApprover
from src.tui.screens.approval import ApprovalScreen
from src.tui.screens.sessions import SessionPicker
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
    "ctrl+c interrupt (quit when idle) · /new or /clear new session · /sessions switch session · "
    "/compact shrink the context · /init write AGENTS.md · /help this help · /quit exit"
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
        *,
        store: SessionStore | None = None,
        start: StartChoice = NEW,
        recorder_factory: Callable[[], Recorder] | None = None,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self._on_close = on_close
        self.store = store
        self.start = start
        self.recorder_factory: Callable[[], Recorder] = recorder_factory or NullRecorder
        self.provider = provider
        self.workspace = workspace
        self.agent = agent_factory(TuiApprover(self))
        self._shutting_down = False
        self._turn_active = False
        self._turn_entered = False  # the turn worker has run its first line
        self._draft: str | None = None  # the submitted message until the turn really starts sending it
        self._finished_since_toggle = False  # a tool finished after a mid-turn mode toggle
        self._switching = False  # a session is being loaded and drawn: no turn may start meanwhile

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
        self._switching = True  # claimed before any worker runs: `_start` releases it after the announcement
        self.run_worker(self._run_check(), group="check")
        self.run_worker(self._start(), group="start")

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

    async def _start(self) -> None:
        try:  # the switch claimed in `on_mount` is held until the announcement is done: no turn can overlap it
            await self._open_start_session()
            await self._announce_instructions()
        finally:
            self._switching = False

    async def _announce_instructions(self) -> None:
        """Show which AGENTS.md files are in effect; never raises. The caller holds the switch claim."""
        try:
            notice = await asyncio.to_thread(self.agent.instructions_notice)
            if notice is not None:
                await self.query_one(Transcript).add_notice(notice.level, notice.text)
        except Exception as exc:  # a failure here must not take the app down
            self.agent.forget_instructions_notice()  # not shown: the next turn announces it again
            with contextlib.suppress(Exception):
                await self.query_one(Transcript).add_notice("error", f"Could not read AGENTS.md: {exc}")

    async def _open_start_session(self) -> None:
        """Begin as the command line asked: the latest session, one by id prefix, or a new one."""
        store, kind = self.store, self.start.kind
        if store is None or kind == "new":
            return
        # `on_mount` claimed the switch; `_start` releases it
        if kind == "pick":
            await self._pick_session()
            return
        transcript = self.query_one(Transcript)
        info: SessionInfo | None = None
        try:
            if kind == "continue":
                info = await asyncio.to_thread(store.latest)
                if info is None:
                    await transcript.add_notice("info", "No previous session for this folder")
            else:
                info = await asyncio.to_thread(store.find, self.start.session_id or "")
        except SessionError as exc:
            await transcript.add_notice("error", str(exc))
        if info is not None:
            await self._show_session(info)  # redraws the transcript, so the warning comes after it
        await self._warn_skipped(store)

    async def _warn_skipped(self, store: SessionStore) -> None:
        if store.skipped:
            await self.query_one(Transcript).add_notice("warn", f"Skipped {len(store.skipped)} unreadable session file(s)")

    def _begin_switch(self) -> bool:
        """Claim the right to switch sessions: False while a turn runs or another switch is under way."""
        if self._turn_active or self._switching:
            return False
        self._switching = True  # no message is accepted until the switch ends
        return True

    async def _close_old(self, old: Recorder | None) -> None:
        """Close the replaced recorder in a thread (a final write must not block the UI), holding the
        switch so no message is accepted meanwhile. The new recorder is already in place: nothing more
        is written to the old session."""
        if old is None:
            return
        held = not self._switching
        self._switching = True
        try:
            await _uninterrupted(asyncio.to_thread(close_recorder, old))
        finally:
            if held:
                self._switching = False

    def _busy_text(self, command: str) -> str:
        hint = "a session is loading" if self._switching and not self._turn_active else f"press Esc to interrupt, then {command}"
        return f"busy — {hint}"

    async def _refuse_switch(self) -> None:
        await self.query_one(Transcript).add_notice("warn", self._busy_text("/sessions"))

    async def _pick_session(self) -> None:
        """List this folder's sessions in a picker (a worker: it waits for the screen), then switch to the choice.

        The caller has claimed the switch (`_begin_switch`, or `on_mount` for the start) and releases it.
        """
        store, transcript = self.store, self.query_one(Transcript)
        if store is None:
            return
        infos = await asyncio.to_thread(store.list)
        await self._warn_skipped(store)
        if not infos:
            await transcript.add_notice("info", "No saved sessions for this folder")
            return
        chosen = await self.push_screen_wait(SessionPicker(infos))
        if chosen is not None:
            await self._show_session(next(info for info in infos if info.id == chosen))

    async def _sessions_command(self) -> None:
        """The `/sessions` worker: pick and switch, then release the claim `_begin_switch` took."""
        try:
            await self._pick_session()
        finally:
            self._switching = False

    async def _show_session(self, info: SessionInfo) -> None:
        """Load `info`, make it the live session and redraw its transcript. The switching flag is held by the caller."""
        store, transcript = self.store, self.query_one(Transcript)
        if store is None:
            return
        try:
            loaded = await asyncio.to_thread(store.load, info.id)
            recorder = FileRecorder.resume(loaded.info.path, self.workspace)
        except (SessionError, SessionWriteError) as exc:  # stay on the session we have
            await transcript.add_notice("error", str(exc))
            return
        try:
            if self._turn_active:  # a turn started while the file was read
                raise AgentBusy
            old = self.agent.reset(recorder=recorder, history=loaded.history, tool_meta=loaded.tool_meta)
        except AgentBusy:
            await self._refuse_switch()
            return
        await self._close_old(old)
        status = self.query_one(StatusBar)
        status.mode = self.agent.mode
        status.pending_mode = False
        status.used = self.agent.estimate()
        status.estimated = True
        await transcript.clear_all()
        await transcript.replay(loaded.items)
        await transcript.add_notice("info", f'Resumed "{info.title}" ({info.message_count} messages)')
        self.agent.forget_instructions_notice()  # the transcript was redrawn: show the notice in it
        await self._announce_instructions()

    def exit(self, *args: Any, **kwargs: Any) -> None:
        self._shutting_down = True  # from here on the widgets are going away: stop applying agent events
        super().exit(*args, **kwargs)

    async def on_unmount(self) -> None:
        self._shutting_down = True
        self.agent.close()
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
        if action == "interrupt_turn":  # with a dialog open Escape belongs to it (the approval denies, then interrupts)
            return not (stack and isinstance(stack[-1], ApprovalScreen | SessionPicker))
        if action == "toggle_mode":  # the mode is not changed under a dialog, nor Tab taken from it
            return not any(isinstance(screen, ApprovalScreen | SessionPicker) for screen in stack)
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
            status = self.query_one(StatusBar)
            status.busy = False
            status.compacting = False

    # turns

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text
        if text.strip().startswith("/"):
            await self._command(text.strip())
            return
        if self._turn_active or self._switching:
            self._restore_draft(text)  # not accepted: the message stays in the box
            hint = "press Esc to interrupt" if self._turn_active else "a session is loading"
            await self.query_one(Transcript).add_notice("warn", f"busy — {hint}")
            return
        self._begin_turn(text, text)

    def _begin_turn(self, text: str, draft: str, init: bool = False) -> None:
        """Start the turn worker for `text`; `draft` is what the box gets back if it never sends."""
        self._turn_active = True
        self._draft = draft
        self.query_one(StatusBar).busy = True
        self._turn_entered = False
        # A partial, not a coroutine: cancelled before it first runs, nothing is left un-awaited.
        self.run_worker(functools.partial(self._turn, text, init), group="turn")

    async def _command(self, command: str) -> None:
        transcript = self.query_one(Transcript)
        name = command.split()[0]
        match name:
            case "/help":
                await transcript.add_notice("info", HELP)
            case "/quit":
                self.exit()
            case "/sessions":
                if not self._begin_switch():
                    await self._refuse_switch()
                    return
                self.run_worker(self._sessions_command, group="sessions")
            case "/new" | "/clear":  # the old session's file stays as it is; the next message starts a new one
                try:
                    # Refused while a turn runs or a session loads: the history belongs to them until they end.
                    if self._turn_active or self._switching:
                        raise AgentBusy
                    old = self.agent.reset(recorder=self.recorder_factory())
                except AgentBusy:
                    await transcript.add_notice("warn", self._busy_text(name))
                    return
                self._switching = True  # no message is accepted until the notice is back in the new transcript
                try:
                    await self._close_old(old)
                    await transcript.clear_all()
                    status = self.query_one(StatusBar)
                    status.used = 0
                    status.estimated = False
                    status.mode = self.agent.mode
                    status.pending_mode = False
                    self.agent.forget_instructions_notice()
                    await self._announce_instructions()
                finally:
                    self._switching = False
            case "/init":
                if self._turn_active or self._switching:
                    await transcript.add_notice("warn", self._busy_text("/init"))
                    return
                if self.agent.mode is Mode.PLAN:
                    await transcript.add_notice("warn", "/init needs BUILD mode — press Tab to switch")
                    return
                try:
                    prompt = await asyncio.to_thread(self._init_prompt)
                except Exception as error:
                    await transcript.add_notice("error", f"Could not start /init: {error}")
                    return
                # Keys and messages were handled while the prompt was built: check again, with no await before the start.
                if self._turn_active or self._switching:
                    await transcript.add_notice("warn", self._busy_text("/init"))
                    return
                if self.agent.mode is Mode.PLAN:
                    await transcript.add_notice("warn", "/init needs BUILD mode — press Tab to switch")
                    return
                self._begin_turn(prompt, draft="/init", init=True)
            case "/compact":
                if self._turn_active or self._switching:
                    await transcript.add_notice("warn", self._busy_text("/compact"))
                    return
                self._turn_active = True
                self._turn_entered = False
                self.query_one(StatusBar).busy = True
                # Like a turn: Esc cancels the "turn" group, and the same events reach the transcript.
                self.run_worker(self._compact, group="turn")
            case _:
                await transcript.add_notice("warn", f"Unknown command: {name} (try /help)")

    def _init_prompt(self) -> str:
        """The `/init` message: the file's state and the parent files already loaded (reads the disk)."""
        own = levels(self.workspace)[-1][0]
        parents = [f.label for f in self.agent.instructions().files if f.label != own and f.state != "omitted"]
        block = render_facts(scan(self.workspace), int(FACTS_SHARE * self.provider.context_window) * CHARS_PER_TOKEN)
        return build_init_prompt(self.workspace, (self.workspace / "AGENTS.md").is_file(), parents, facts=block)

    async def _turn(self, text: str, init: bool = False) -> None:
        self._turn_entered = True
        transcript = self.query_one(Transcript)
        status = self.query_one(StatusBar)
        agents_md = self.workspace / "AGENTS.md"
        before: tuple[int, int] | None = None
        sent = False
        try:
            if status.error and not await self._run_check():  # R3: re-check before giving up on the model
                return
            self._draft = None  # from here the message is part of the conversation
            sent = True
            before = fingerprint(agents_md)
            await transcript.add_user(text)
            self._finished_since_toggle = False
            status.pending_mode = False
            await self._run_with_events(lambda emit: self.agent.run_turn(text, emit))
        finally:
            try:
                if init and sent and not self._shutting_down:
                    if fingerprint(agents_md) == before:
                        with contextlib.suppress(NoMatches):
                            await transcript.add_notice("info", "/init finished without changing AGENTS.md")
                    else:
                        await self._check_init()
            finally:
                self._end_turn()

    async def _check_init(self) -> None:
        """After an `/init` turn that changed AGENTS.md: warn about commands the scan did not find. Never raises."""
        try:
            notice = await asyncio.to_thread(init_check, self.workspace)
            if notice is not None:
                await self.query_one(Transcript).add_notice(*notice)
        except NoMatches:
            pass
        except Exception as error:
            with contextlib.suppress(Exception):
                await self.query_one(Transcript).add_notice("error", f"/init check failed: {error}")

    async def _compact(self) -> None:
        self._turn_entered = True
        try:
            await self._run_with_events(self.agent.compact)
        finally:
            self._end_turn()

    async def _run_with_events(self, work: Callable[[EventSink], Awaitable[None]]) -> None:
        """Run an agent job (a turn, a compaction) with its events applied to the widgets in order."""
        # The loop's sink is synchronous; a consumer task applies the events to the widgets in order.
        events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        consumer = asyncio.create_task(self._consume(events))
        try:
            await work(events.put_nowait)
        except AgentBusy:
            events.put_nowait(Notice("warn", "busy — a turn is already running"))
        finally:
            # Whatever ended the job, the queued events still reach the transcript, and a
            # repeated Esc must not cut that short.
            await _uninterrupted(self._finish_turn(events, consumer))

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
            case Compacting(active):
                status.compacting = active
            case UsageUpdate(used, window, estimated):
                status.used = used
                status.window = window
                status.estimated = estimated
            case TurnEnded(turn_status):
                status.pending_mode = False
                await transcript.end_assistant()
                await transcript.close_open_tools(turn_status.replace("_", " "))
                if turn_status == "interrupted":
                    await transcript.add_notice("info", "interrupted", icon="⏹")
