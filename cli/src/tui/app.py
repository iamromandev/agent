"""The Textual app: layout, keys, the turn worker and the slash commands."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import signal
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.widgets import Tabs

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
from src.agent.terminal.manager import TerminalError, TerminalManager
from src.tui import NEW, StartChoice
from src.tui.approver import TuiApprover
from src.tui.screens.approval import ApprovalScreen
from src.tui.screens.sessions import SessionPicker
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar, display_path
from src.tui.widgets.terminal import TerminalPane, TerminalView
from src.tui.widgets.transcript import Transcript

KEY_GUARD = 0.5  # seconds an approval dialog that opens over a terminal ignores keys: they were meant for it


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
    "ctrl+c interrupt (quit when idle) · ctrl+t terminal pane (show, hide, back to chat) · "
    "ctrl+\\ back to chat · ctrl+pageup / ctrl+pagedown switch terminal tab (these four work inside a terminal too) · "
    "ctrl+w close an exited terminal's tab (or the clicked one) · /new or /clear new session · "
    "/sessions switch session · "
    "/compact shrink the context · /init write AGENTS.md · /undo undo the last turn · /help this help · /quit exit"
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
        Binding("ctrl+t", "toggle_terminal", "Terminal", priority=True),
        Binding("ctrl+backslash", "focus_input", "Chat", priority=True, show=False),
        Binding("ctrl+pageup", "terminal_tab(-1)", "Previous terminal", priority=True, show=False),
        Binding("ctrl+pagedown", "terminal_tab(1)", "Next terminal", priority=True, show=False),
        Binding("ctrl+w", "close_terminal_tab", "Close terminal tab", priority=True, show=False),
    ]
    # Tab, Esc and Ctrl+C are the app's, except while a terminal has focus: they belong to its process then.
    TERMINAL_KEYS: ClassVar[frozenset[str]] = frozenset({"toggle_mode", "interrupt_turn", "interrupt_or_quit"})

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
        terminals: TerminalManager | None = None,
    ) -> None:
        super().__init__()
        self._on_close = on_close
        self.terminals = terminals
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
        self._undoing = False  # /undo is running: keys that act on a turn leave it alone
        self._switching = False  # a session is being loaded and drawn: no turn may start meanwhile
        self._signals_before: dict[signal.Signals, Any] = {}  # what SIGTERM and SIGHUP did before the app
        self._keys_ignored_until = 0.0  # time.monotonic() until which an approval dialog takes no keys
        self._closing_tabs: set[str] = set()  # terminals whose tab is being closed (they are being stopped)

    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")
        if self.terminals is not None:
            yield TerminalPane(self.terminals)
        yield ChatInput(id="input")
        yield StatusBar()

    def on_mount(self) -> None:
        status = self.query_one(StatusBar)
        status.model = self.provider.model
        status.window = self.provider.context_window
        status.workspace = display_path(self.workspace)
        status.mode = self.agent.mode
        self.query_one(ChatInput).focus()
        self._stop_on_signals()
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

    def guard_keys(self) -> None:
        """Called as an approval dialog opens: keys the user is typing into a terminal must not answer it.

        With a terminal focused, an Up and an Enter typed for the shell would otherwise pick "Allow once".
        """
        if isinstance(self.focused, TerminalView):
            self._keys_ignored_until = time.monotonic() + KEY_GUARD

    async def on_event(self, event: events.Event) -> None:
        if (
            isinstance(event, events.Key)
            and not event.is_forwarded
            and event.key != "ctrl+c"  # it interrupts: it never approves
            and time.monotonic() < self._keys_ignored_until
            and isinstance(next(iter(reversed(self.screen_stack)), None), ApprovalScreen)  # empty on shutdown
        ):
            event.stop()
            return
        await super().on_event(event)

    def exit(self, *args: Any, **kwargs: Any) -> None:
        self._shutting_down = True  # from here on the widgets are going away: stop applying agent events
        super().exit(*args, **kwargs)

    def _stop_on_signals(self) -> None:
        """Quit properly on SIGTERM and SIGHUP (closing the window), which stops the terminals' processes too.

        Without a handler the process would die at once and leave processes that ignore SIGHUP behind. The exit
        code is 128 + the signal, as for a process the signal kills.
        """
        if self.terminals is None:
            return
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):  # not the main thread, or no signals
                previous = signal.getsignal(sig)
                loop.add_signal_handler(sig, functools.partial(self.exit, return_code=128 + sig))
                self._signals_before[sig] = previous

    def _restore_signals(self) -> None:
        """Put back what the signals did before the app (`nohup agent` ignores SIGHUP; the default would kill it)."""
        loop = asyncio.get_running_loop()
        for sig, previous in self._signals_before.items():
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)  # leaves the default action behind
                if previous is not None:  # None: it was not installed from Python; the default stays
                    signal.signal(sig, previous)
        self._signals_before.clear()

    async def on_unmount(self) -> None:
        self._shutting_down = True
        self.agent.close()
        try:
            if self.terminals is not None:
                await self.terminals.close_all()  # no terminal process outlives the app, whatever ended it
        finally:
            try:
                if self._on_close is not None:
                    await self._on_close()
            finally:
                self._restore_signals()  # until here a second signal is absorbed too

    # keys

    async def on_terminal_pane_started_hidden(self, message: TerminalPane.StartedHidden) -> None:
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            await self.query_one(Transcript).add_notice("info", f"terminal {message.name} started · ctrl+t to show")

    def action_toggle_mode(self) -> None:
        self.agent.set_mode(Mode.PLAN if self.agent.mode is Mode.BUILD else Mode.BUILD)
        status = self.query_one(StatusBar)
        status.mode = self.agent.mode
        if self._turn_active and not self._undoing:  # a step already running keeps the mode it started with
            self._finished_since_toggle = False
            status.pending_mode = True
        else:
            status.pending_mode = False

    def action_interrupt_turn(self) -> None:
        self.interrupt_turn()

    async def action_toggle_terminal(self) -> None:
        """Show the pane (starting a shell if it is empty) and focus it; from a terminal go back to the input,
        and from the input hide the pane. Hidden terminals keep running."""
        if self.terminals is None:
            return
        pane = self.query_one(TerminalPane)
        focused = self.focused
        if not pane.display:
            if not self.terminals.names():
                try:
                    self.terminals.start("shell", None)
                except TerminalError as exc:
                    await self.query_one(Transcript).add_notice("error", f"Could not start a shell: {exc}")
                    return
            pane.display = True
            await pane.focus_active()
        elif isinstance(focused, TerminalView):
            self.query_one(ChatInput).focus()
        elif isinstance(focused, ChatInput):
            pane.display = False
        else:
            await pane.focus_active()

    def action_focus_input(self) -> None:
        self.query_one(ChatInput).focus()

    async def action_terminal_tab(self, step: int) -> None:
        await self.query_one(TerminalPane).switch(step)

    def action_close_terminal_tab(self) -> None:
        """Stop the active terminal if it runs, close its tab, and go on in the pane (the input if it was the last).

        A worker: stopping a process that ignores SIGTERM takes the grace period, and the app must stay responsive.
        """
        name = self.query_one(TerminalPane).active_name
        if self.terminals is None or name is None or self._closing_tabs:
            return  # one close at a time: while it runs, the focus (and so what ctrl+w may close) is not settled
        self._closing_tabs.add(name)
        self.run_worker(self._close_terminal_tab(name), group="terminal-close")

    async def _close_terminal_tab(self, name: str) -> None:
        pane = self.query_one(TerminalPane)
        try:
            if self.terminals is not None:
                await self.terminals.close(name)
            await pane.settled()
            focused = self.focused
            if pane.tab_count:
                # Focus goes on in the pane only if the user is still there (or nowhere): a slow close must not
                # pull them back from the chat input they moved to meanwhile.
                if focused is None or pane in focused.ancestors_with_self:
                    await pane.focus_active()
            else:  # nothing left to show: back to the chat (Ctrl+T starts a fresh shell)
                pane.display = False
                self.query_one(ChatInput).focus()
        finally:
            self._closing_tabs.discard(name)  # only now: until the focus has settled, no other tab may be closed

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        stack = self.screen_stack  # empty while shutting down; `self.screen` would raise then
        if action in ("toggle_terminal", "focus_input"):
            return self.terminals is not None and not any(isinstance(s, ApprovalScreen | SessionPicker) for s in stack)
        if action in ("terminal_tab", "close_terminal_tab"):
            return self._tab_key_applies(action, stack)
        if action in self.TERMINAL_KEYS and stack and isinstance(stack[-1].focused, TerminalView):
            return False  # the key goes on to the terminal
        if action == "interrupt_turn":  # with a dialog open Escape belongs to it (the approval denies, then interrupts)
            return not (stack and isinstance(stack[-1], ApprovalScreen | SessionPicker))
        if action == "toggle_mode":  # the mode is not changed under a dialog, nor Tab taken from it
            return not any(isinstance(screen, ApprovalScreen | SessionPicker) for screen in stack)
        return True

    def _tab_key_applies(self, action: str, stack: list[Any]) -> bool:
        """Ctrl+PageUp/PageDown and Ctrl+W work while the pane is shown. Ctrl+W is also readline's and the
        input's "delete word", so it closes only a tab nobody types into: that of an exited terminal that has
        focus, or any tab while the tab bar has it (not the input, the transcript, a running terminal or nothing)."""
        if self.terminals is None or not stack or any(isinstance(s, ApprovalScreen | SessionPicker) for s in stack):
            return False
        pane = self.query_one(TerminalPane)
        if not pane.display or not pane.tab_count:
            return False
        if action == "terminal_tab":
            return True
        focused = stack[-1].focused
        if isinstance(focused, TerminalView):
            return focused.terminal.status[0] == "exited"
        return isinstance(focused, Tabs)  # a running tab closes only from the tab bar, never from the input or transcript

    def action_interrupt_or_quit(self) -> None:
        if self._turn_active:
            self.interrupt_turn()
        else:
            self.exit()

    def interrupt_turn(self) -> None:
        if self._undoing:  # an undo is not a turn: there is nothing to cancel, and its guard must stay up
            return
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
        self.run_worker(functools.partial(self._turn, text, init, draft), group="turn")

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
            case "/undo":
                await self._undo_command()
            case _:
                await transcript.add_notice("warn", f"Unknown command: {name} (try /help)")

    async def _undo_command(self) -> None:
        transcript = self.query_one(Transcript)
        if self._turn_active or self._switching:
            await transcript.add_notice("warn", self._busy_text("/undo"))
            return
        status = self.query_one(StatusBar)
        self._turn_active = True
        self._undoing = True
        status.busy = True
        events: list[AgentEvent] = []
        prompt: str | None = None
        try:
            prompt = await self.agent.undo(events.append)
        except AgentBusy:
            events.append(Notice("warn", self._busy_text("/undo")))
        except Exception as error:
            events.append(Notice("error", f"Undo failed: {error}"))
        finally:
            self._turn_active = False
            self._undoing = False
            status.busy = False
        if prompt is not None:
            await transcript.remove_last_turn()
            self._restore_draft(prompt)
            status.used = self.agent.estimate()
            status.estimated = True
        for event in events:
            if isinstance(event, Notice):
                await transcript.add_notice(event.level, event.text)

    def _init_prompt(self) -> str:
        """The `/init` message: the file's state and the parent files already loaded (reads the disk)."""
        own = levels(self.workspace)[-1][0]
        parents = [f.label for f in self.agent.instructions().files if f.label != own and f.state != "omitted"]
        block = render_facts(scan(self.workspace), int(FACTS_SHARE * self.provider.context_window) * CHARS_PER_TOKEN)
        return build_init_prompt(self.workspace, (self.workspace / "AGENTS.md").is_file(), parents, facts=block)

    async def _turn(self, text: str, init: bool = False, draft: str | None = None) -> None:
        self._turn_entered = True
        transcript = self.query_one(Transcript)
        status = self.query_one(StatusBar)
        agents_md = self.workspace / "AGENTS.md"
        before: tuple[int, int] | None = None
        sent = False
        shown = False  # the user entry was drawn (or was being drawn)
        started = False  # the agent was handed the message

        async def work(emit: EventSink) -> None:
            nonlocal started
            started = True
            await self.agent.run_turn(text, emit, draft=draft)

        try:
            if status.error and not await self._run_check():  # R3: re-check before giving up on the model
                return
            self._draft = None  # from here the message is part of the conversation
            sent = True
            before = fingerprint(agents_md)
            shown = True
            await transcript.add_user(text)
            self._finished_since_toggle = False
            status.pending_mode = False
            await self._run_with_events(work)
        finally:
            try:
                if shown and not self._shutting_down:
                    with contextlib.suppress(Exception):
                        if started:
                            transcript.end_turn(transcript.latest_turn)  # what follows is not part of the turn
                        else:  # Esc landed before the agent got the message: no turn, so no entry, and the text is kept
                            self._draft = draft if draft is not None else text
                            await _uninterrupted(self._discard_turn(transcript))
                if init and sent and started and not self._shutting_down:
                    if fingerprint(agents_md) == before:
                        with contextlib.suppress(NoMatches):
                            await transcript.add_notice("info", "/init finished without changing AGENTS.md")
                    else:
                        await self._check_init()
            finally:
                self._end_turn()

    @staticmethod
    async def _discard_turn(transcript: Transcript) -> None:
        await transcript.remove_turn(transcript.latest_turn)

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
