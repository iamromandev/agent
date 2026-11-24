"""The Textual app: layout, keys, the turn worker and the slash commands."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import re
import signal
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Literal

from textual import events
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import OptionList, Tabs, TextArea
from textual.worker import WorkerCancelled

from src.client import clipboard
from src.client.clipboard import ClipboardError, ClipboardUnsupported, PasteFolder
from src.engine import (
    Agent,
    AgentBusy,
    AgentEvent,
    AssistantText,
    Compacting,
    EventSink,
    Notice,
    Phase,
    TaskProgress,
    TodosChanged,
    ToolFinished,
    ToolProgress,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)
from src.engine.agents.loader import READ_ONLY_TOOLS, Agents
from src.engine.approval import ApprovalRequest, Approver, Choice
from src.engine.commands.builtins import BUILTINS
from src.engine.commands.loader import Commands
from src.engine.compaction import CHARS_PER_TOKEN, turn_starts
from src.engine.configguard import NEXT_LAUNCH, ConfigGuard, HeldGlobal
from src.engine.configwatch import Cause, TrustRevoked
from src.engine.events import WorkflowLine, WorkflowProgress
from src.engine.facts import FACTS_SHARE, init_check, render_facts, scan
from src.engine.guardrails.paths import (
    BUILTIN_HOME,
    BUILTIN_NAMES,
    MAX_TRACKED_DEPTH,
    MAX_TRACKED_DIRS,
    MAX_TRACKED_FILES,
)
from src.engine.guardrails.setup import Guardrails
from src.engine.hooks.setup import Hooks, hook_failed
from src.engine.image_pull import NOT_INSTALLED, ROLES, ImagePuller, Level, Role, format_status
from src.engine.instructions import fingerprint, levels
from src.engine.loop import NOT_SUMMARIZED, close_recorder
from src.engine.mcp.hub import HubEvent, McpHub
from src.engine.mcp.names import tool_name
from src.engine.mentions import MentionNote, apply_vision, expand_mentions, note_text
from src.engine.mentions_index import FileIndex
from src.engine.mode import Mode
from src.engine.permissions.setup import Rules, RuleSavedNotTrusted
from src.engine.prompt import build_init_prompt
from src.engine.provider.base import Image, SwitchableProvider
from src.engine.provider.images import MAX_IMAGES, ImageError, describe, read_image_fd, size_text, sniff
from src.engine.provider.models import (
    SHOW_TIMEOUT,
    CatalogError,
    ModelCatalog,
    ModelInfo,
    installed_name,
    resolve_window,
    shown_name,
)
from src.engine.provider.vision import VisionCheck
from src.engine.session import (
    FileRecorder,
    NullRecorder,
    Recorder,
    SessionError,
    SessionInfo,
    SessionStore,
    SessionWriteError,
)
from src.engine.session.records import model_notice
from src.engine.skills.loader import Skills
from src.engine.terminal.manager import TerminalError, TerminalManager
from src.engine.text import escape, escape_display, one_line
from src.engine.tools.base import Preview
from src.engine.turn_diff import BANNER, NO_FILES, NOTHING
from src.engine.usage import summary
from src.engine.workflows.loader import Workflows
from src.engine.workflows.model import PlanSummary, RunResult
from src.engine.workflows.runner import WorkflowRunner
from src.engine.workflows.runs import RunIndex
from src.engine.workflows.timings import TimingStore
from src.tui import NEW, StartChoice
from src.tui.approver import TuiApprover
from src.tui.hints import MIN_ROWS, hint_text
from src.tui.queue import MessageQueue, Queued, merge
from src.tui.screens.approval import ApprovalScreen
from src.tui.screens.diff import DiffScreen
from src.tui.screens.global_trust import GlobalTrustScreen
from src.tui.screens.mcp_trust import McpTrustScreen
from src.tui.screens.models import ModelPicker
from src.tui.screens.permission_trust import PermissionTrustScreen
from src.tui.screens.sessions import SessionPicker
from src.tui.screens.usage import UsageScreen
from src.tui.widgets.attachment_panel import AttachmentPanel
from src.tui.widgets.chat_input import PLACEHOLDER, ChatInput, InputRow
from src.tui.widgets.command_suggestions import CommandSuggestions
from src.tui.widgets.key_hints import KeyHints
from src.tui.widgets.path_suggestions import PathSuggestions
from src.tui.widgets.queue_panel import QueuePanel
from src.tui.widgets.status import StatusBar, display_path
from src.tui.widgets.terminal import TerminalPane, TerminalView
from src.tui.widgets.todo_panel import TodoPanel
from src.tui.widgets.transcript import Transcript

DIALOGS = (ApprovalScreen, SessionPicker, ModelPicker, McpTrustScreen, PermissionTrustScreen, GlobalTrustScreen, DiffScreen, UsageScreen)  # modal screens (and the diff viewer): the app's own keys leave them alone
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
    "Enter send · ctrl+j / shift+enter new line · ctrl+v paste an image (macOS) · Esc on an empty input clears pasted images · Tab switch BUILD/PLAN · Enter during a turn queues the message (sent when it ends; Esc puts it back) · Esc interrupt · "
    "ctrl+c interrupt (quit when idle) · ctrl+t terminal pane (show, hide, back to chat) · "
    "ctrl+\\ back to chat · ctrl+pageup / ctrl+pagedown switch terminal tab (these four work inside a terminal too) · "
    "ctrl+w close an exited terminal's tab (or the clicked one) · / suggests commands · /new or /clear new session · "
    "/sessions switch session · /mcp MCP servers and their tools · /permissions saved permission rules · "
    "/guardrails protected paths, environment filter and redaction · /hooks configured hooks · /skills loaded skills · /agents loaded subagents · /compact shrink the context · /model switch model · /init write AGENTS.md · /undo undo the last turn · /diff show the last turn's changes · /usage session token totals · /images image models (status, pull, cancel) · /workflow run a workflow (list, run, resume, cancel) · /help this help · /quit exit"
)

WORKFLOW_RUNNING = "a workflow is running (Esc or /workflow cancel stops it)"
WORKFLOW_USAGE = "usage: /workflow [list | run <name> [input] | resume [id] [--redo <step>] | cancel]"
PLAN_ARG_SHOWN = 60  # characters of a call's key argument (prompt, text) in the plan dialog
FAILED_SHOWN = 200
PLAN_NOTE_SHOWN = 200
PLAN_LINES = 40 # calls listed in the plan dialog; the counts above them cover the rest

_COMMAND = re.compile(r"/(\S+)(\s.*)?\Z", re.DOTALL)


class AgentApp(App[None]):
    CSS = """
    #transcript { height: 1fr; }
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
        provider: SwitchableProvider,
        workspace: Path,
        *,
        store: SessionStore | None = None,
        start: StartChoice = NEW,
        recorder_factory: Callable[[], Recorder] | None = None,
        on_close: Callable[[], Awaitable[None]] | None = None,
        terminals: TerminalManager | None = None,
        mcp: McpHub | None = None,
        mcp_trust: Callable[[], None] | None = None,
        permissions: Rules | None = None,
        permission_warnings: Sequence[str] = (),
        commands: Commands | None = None,
        skills: Skills | None = None,
        agents: Agents | None = None,
        guard: Guardrails | None = None,
        hooks: Hooks | None = None,
        catalog: ModelCatalog | None = None,
        vision: VisionCheck | None = None,
        window_cap: int,
        launch_notices: Sequence[str] = (),
        paste: PasteFolder | None = None,
        images: ImagePuller | None = None,
        workflows: Workflows | None = None,
        run_index: RunIndex | None = None,
        timings: TimingStore | None = None,
        config_guard: ConfigGuard | None = None,
        held: HeldGlobal | None = None,
    ) -> None:
        super().__init__()
        # What the marker holds until the next launch besides the allow rules, hooks and servers (shown and asked about
        # with them): the global keepEnv and the global commands, skills, agents and workflows.
        self._held = held if held is not None else HeldGlobal()
        # The config watch's reaction object: its warnings go to the transcript, `/guardrails` lists its changes, and
        # its marker (an earlier run's change to the global config) opens the global trust dialog at launch.
        self.config_guard = config_guard
        self._loop_thread: int | None = None  # the thread the app's event loop runs on, once mounted
        self.workflows = workflows if workflows is not None else Workflows()
        self._run_index = run_index
        self._timings = timings  # None: the machine's own file, made on the first workflow command
        self._workflow_active = False  # a `/workflow` run (or resume) is going: it holds the turn claim
        self.images = images  # `/images`: the models' state and the background warm-up (None: no image tools in this run)
        self._on_close = on_close
        self.terminals = terminals
        self.mcp = mcp
        self._mcp_trust = mcp_trust  # remembers that the user trusted the project's servers
        self.permissions = permissions if permissions is not None else Rules.empty()
        self._permission_warnings = tuple(permission_warnings)
        self.commands = commands if commands is not None else Commands({}, ())
        self.skills = skills if skills is not None else Skills({}, ())
        self.agents = agents if agents is not None else Agents.empty()
        self.guard = guard if guard is not None else Guardrails.off(workspace)
        self.hooks = hooks if hooks is not None else Hooks.empty()
        self.store = store
        self.start = start
        self.recorder_factory: Callable[[], Recorder] = recorder_factory or NullRecorder
        self.provider = provider
        self.catalog = catalog
        self.vision = vision if vision is not None else VisionCheck(catalog)  # one answer per model per run
        self.window_cap = window_cap  # AGENT_NUM_CTX: caps every window
        self._launch_notices = tuple(launch_notices)  # info lines for the transcript, already escaped
        self.workspace = workspace
        self.file_index = FileIndex(workspace, self.guard)
        self._indexing = False
        self._reindex_pending = False
        self.agent = agent_factory(TuiApprover(self))
        self._shutting_down = False
        self._turn_active = False
        self._turn_entered = False  # the turn worker has run its first line
        self.paste = paste if paste is not None else PasteFolder(workspace)  # removed when the app quits
        self._pending: list[tuple[Path, str, tuple[int, int, int, int]]] = []  # pasted images waiting: (file, chip label, identity)
        self._draft: str | None = None  # the submitted message until the turn really starts sending it
        self._queue = MessageQueue()  # messages typed while a turn runs, sent together when it ends `done`
        self._interrupted = False  # Esc or ctrl+c landed while the turn or compaction was active: the queue is not sent
        self._job_status: str | None = None  # how the running turn or compaction ended, once known
        self._progress_call: str | None = None  # the call whose live line the status bar shows
        self._finished_since_toggle = False  # a tool finished after a mid-turn mode toggle
        self._undoing = False  # /undo is running: keys that act on a turn leave it alone
        self._switching = False  # a session is being loaded and drawn: no turn may start meanwhile
        self._signals_before: dict[signal.Signals, Any] = {}  # what SIGTERM and SIGHUP did before the app
        self._keys_ignored_until = 0.0  # time.monotonic() until which an approval dialog takes no keys
        self._closing_tabs: set[str] = set()  # terminals whose tab is being closed (they are being stopped)

    class McpEvent(Message):
        """An MCP server became ready, failed or exited, or the hub has a warning."""

        def __init__(self, event: HubEvent) -> None:
            super().__init__()
            self.event = event

    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")
        if self.terminals is not None:
            yield TerminalPane(self.terminals)
        yield TodoPanel(id="todos")
        yield QueuePanel(id="queue")
        yield AttachmentPanel(id="attachments")
        yield CommandSuggestions(id="suggestions")  # above the input
        yield PathSuggestions(id="path-suggestions")
        yield InputRow(ChatInput(id="input", placeholder=PLACEHOLDER))
        yield KeyHints(id="hints")
        yield StatusBar()

    def on_mount(self) -> None:
        status = self.query_one(StatusBar)
        status.model = escape_display(self.provider.model)
        status.window = self.provider.context_window
        status.workspace = display_path(self.workspace)
        status.mode = self.agent.mode
        self.query_one(ChatInput).suggestions = self.query_one(CommandSuggestions)
        self.query_one(ChatInput).paths = self.query_one(PathSuggestions)
        self.query_one(ChatInput).focus()
        self.watch(status, "busy", lambda _: self._refresh_hints(), init=False)
        self._refresh_hints()
        self._apply_height(self.size.height)
        self._reindex()
        self._stop_on_signals()
        self._loop_thread = threading.get_ident()
        if self.config_guard is not None:  # a reaction on another thread tells its listeners on this loop
            self.config_guard.attach_loop(asyncio.get_running_loop())
        self._switching = True  # claimed before any worker runs: `_start` releases it after the announcement
        self.run_worker(self._run_check(), group="check")
        self.run_worker(self._start(), group="start")

    def _refresh_hints(self) -> None:
        """Recompute the key-hints line; called wherever one of its inputs changes."""
        try:
            chat = self.query_one(ChatInput)
            hints = self.query_one(KeyHints)
            self.query_one(StatusBar)
        except NoMatches:  # an early event, before `compose` finished
            return
        hints.show(
            hint_text(
                busy=self._queueing(),  # exactly when Esc interrupts and Enter queues
                popup_open=chat.popup_open,
                terminal_focused=isinstance(self.focused, TerminalView),
                has_terminals=self.terminals is not None,
            )
        )

    def _apply_height(self, rows: int) -> None:
        try:
            self.query_one(KeyHints).display = rows >= MIN_ROWS
        except NoMatches:
            return

    def on_resize(self, event: events.Resize) -> None:
        self._apply_height(event.size.height)

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
        source: Literal["startup", "resume"] = "startup"
        try:  # the switch claimed in `on_mount` is held until the hook is done: no turn can overlap it
            try:
                source = await self._open_start_session()
                await self._announce_instructions()
            finally:
                self._start_mcp()  # after the session is drawn (a redraw would wipe the servers' notices), before the hook
                self._listen_config()  # also after the draw: a change seen during the launch is shown now
                self._start_trust()
            await self._session_started(source)
        finally:
            self._switching = False

    def _start_mcp(self) -> None:
        """Start the configured MCP servers in the background; their events become notices."""
        if self.mcp is None:
            return
        self.mcp.listen(self._post_mcp_event)
        self.mcp.start()  # never the untrusted ones

    def _listen_config(self) -> None:
        if self.config_guard is not None:
            self.config_guard.listen(self._config_changed)

    def _config_changed(self, text: str) -> None:
        """A config watch warning (already escaped) for the transcript; from any thread."""
        if self._shutting_down:
            raise RuntimeError("the app is closing")  # the guard keeps the text: it is printed after the app
        if self._loop_thread is None or threading.get_ident() == self._loop_thread:
            self.call_later(self._show_config_change, text)
        else:
            self.call_from_thread(self._show_config_change, text)  # raises when the app is not running: kept

    async def _show_config_change(self, text: str) -> None:
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            await self.query_one(Transcript).add_notice("warn", text)

    def _global_waiting(self) -> bool:
        """An earlier run saw the global config change (the marker), and its rules, hooks and servers wait for a yes."""
        return (
            self.config_guard is not None
            and self.config_guard.marker_set
            and not self.permissions.global_active
            and not self.permissions.global_revoked
        )

    def _start_trust(self) -> None:
        """One worker for everything to show or ask at launch: the permission warnings, then the global config
        prompt, then the MCP prompt, then the permission prompt. One after the other, so two prompts never overlap."""
        if (
            self._permission_warnings
            or self._launch_notices
            or self.commands.warnings
            or self.skills.warnings
            or self.permissions.untrusted_allow
            or self.hooks.untrusted()
            or (self.mcp is not None and self.mcp.untrusted("project"))
            or self._global_waiting()
        ):
            self.run_worker(self._ask_trust(), group="trust")

    async def _ask_trust(self) -> None:
        transcript = self.query_one(Transcript)
        for warning in (*self._permission_warnings, *self.commands.warnings, *self.skills.warnings):
            await transcript.add_notice("warn", warning)
        for line in self._launch_notices:
            await transcript.add_notice("info", line)
        if self._global_waiting():
            await self._ask_global_trust()
        if self.mcp is not None and self.mcp.untrusted("project"):
            await self._ask_mcp_trust()
        await self._ask_permission_trust()

    async def _ask_global_trust(self) -> None:
        """Ask once whether to use the global allow rules, hooks and MCP servers again; only a yes clears the marker
        (the agent's own write) and turns them on. No, or Esc, keeps them off for this run and keeps the marker."""
        guard = self.config_guard
        if guard is None:
            return
        servers = self.mcp.untrusted("global") if self.mcp is not None else []
        hooks = [entry for entry in self.hooks.entries if entry.source == "global"]
        marker = await asyncio.to_thread(guard.marker_info)
        transcript = self.query_one(Transcript)
        for line in self._held.notices():  # one per kind, with its count (names come from files: counts only here)
            await transcript.add_notice("warn", line)
        self._keys_ignored_until = time.monotonic() + KEY_GUARD  # keys typed for the chat must not answer it
        screen = GlobalTrustScreen(
            self.permissions.global_allow, hooks, servers, marker, deny=self.permissions.global_deny, held=self._held
        )
        trusted = await self.push_screen_wait(screen)
        if not trusted:
            await transcript.add_notice(
                "info", "global config not trusted: global allow rules, hooks and MCP servers are off for this run"
            )
            return
        try:
            await asyncio.to_thread(guard.trust_global)  # removes the marker: not on the event loop
        except TrustRevoked as exc:  # a change seen in this run: everything stays off, the marker stays
            await transcript.add_notice("warn", escape(f"global config: {exc}"))
            return
        except OSError as exc:  # the marker is still there: nothing was turned on
            await transcript.add_notice("warn", escape(f"global config: {exc}"))
            return
        if self.mcp is not None and servers:
            self.mcp.start([s.name for s in servers])
        later = f"; {NEXT_LAUNCH}" if self._held.any else ""
        await transcript.add_notice("info", f"global config trusted: global allow rules, hooks and MCP servers are on{later}")

    async def _ask_permission_trust(self) -> None:
        """Ask once whether to trust the project's allow rules and hooks (one decision for the file's bytes); only a
        yes turns them on."""
        rules = self.permissions.untrusted_allow
        hooks = self.hooks.untrusted()
        if not rules and not hooks:
            return
        self._keys_ignored_until = time.monotonic() + KEY_GUARD  # keys typed for the chat must not answer it
        trusted = await self.push_screen_wait(PermissionTrustScreen(rules, hooks))
        transcript = self.query_one(Transcript)
        if not trusted:
            if rules and hooks:
                text = "project permission rules and hooks not trusted: allow rules and hooks are off for this run"
            elif hooks:
                text = "project hooks not trusted: hooks are off for this run"
            else:
                text = "project permission rules not trusted: allow rules are off for this run"
            await transcript.add_notice("info", text)
            return
        try:
            await asyncio.to_thread(self.permissions.trust_project)  # writes the trust store: not on the event loop
        except TrustRevoked as exc:  # the config watch revoked project trust meanwhile: nothing was trusted
            await transcript.add_notice("warn", escape(f"permissions: {exc}"))
        except OSError as exc:  # the rules are on for this run; it will ask again next time
            await transcript.add_notice("warn", escape(f"permissions: could not remember the trust: {exc}"))

    async def _ask_mcp_trust(self) -> None:
        """Ask once whether to start the project's MCP servers; only a yes starts them."""
        if self.mcp is None:
            return
        servers = self.mcp.untrusted("project")  # the global ones held by the marker have their own prompt
        names = ", ".join(s.name for s in servers)
        self._keys_ignored_until = time.monotonic() + KEY_GUARD  # keys typed for the chat must not answer it
        trusted = await self.push_screen_wait(McpTrustScreen(servers))
        transcript = self.query_one(Transcript)
        if not trusted:
            await transcript.add_notice("info", f"mcp: project servers not trusted, so not started: {names}")
            return
        if self._mcp_trust is not None:
            try:
                self._mcp_trust()
            except TrustRevoked as exc:  # the config watch revoked project trust meanwhile: nothing starts
                await transcript.add_notice("warn", escape(f"mcp: {exc}"))
                return
            except OSError as exc:  # the servers still start for this run; it will ask again next time
                await transcript.add_notice("warn", f"mcp: could not remember the trust: {exc}")
        self.mcp.start([s.name for s in servers])

    def _post_mcp_event(self, event: HubEvent) -> None:
        self.post_message(self.McpEvent(event))

    async def on_agent_app_mcp_event(self, message: McpEvent) -> None:
        kind, server, text = message.event
        match kind:
            case "ready":
                level, line = "info", f"mcp {server} ready · {text}"
            case "failed":
                level, line = "warn", f"mcp {server} failed: {text}"
            case "exited":
                level, line = "warn", f"mcp {server} exited: {text}"
            case "warn":
                level, line = "warn", f"mcp {server}: {text}" if server else text
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            await self.query_one(Transcript).add_notice(level, line)

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

    async def _session_started(self, source: Literal["startup", "resume", "new", "switch"]) -> None:
        """Run the SessionStart hooks (the session is drawn and its notices are in); never raises. The caller holds the
        switch claim, so no message is accepted while a hook runs. The hook runs in its own worker (group
        `session-start`), which Esc cancels: the runner then kills the hook's process group."""
        if not self.hooks.has("SessionStart"):
            return  # nothing to run, so nothing for Esc or ctrl+c to interrupt
        events: list[AgentEvent] = []

        async def run() -> None:
            try:
                await self.agent.session_started(source, events.append)
            except Exception as exc:  # the agent does not raise; a bug here must not take the app down
                events.append(hook_failed("SessionStart", exc))

        worker = self.run_worker(run(), group="session-start", exit_on_error=False)
        try:
            await worker.wait()
        except WorkerCancelled:
            events.append(Notice("info", "hook SessionStart interrupted"))
        with contextlib.suppress(Exception):
            transcript = self.query_one(Transcript)
            for event in events:
                if isinstance(event, Notice):
                    await transcript.add_notice(event.level, event.text)

    async def _finish_new_session(self) -> None:
        """The end of `/new` and `/clear`: the SessionStart hook, then release the claim the command took."""
        try:
            await self._session_started("new")
        finally:
            self._switching = False

    async def _open_start_session(self) -> Literal["startup", "resume"]:
        """Begin as the command line asked: the latest session, one by id prefix, or a new one. Returns the source
        for the SessionStart hooks: `resume` when a saved session was opened, else `startup`."""
        store, kind = self.store, self.start.kind
        if store is None or kind == "new":
            return "startup"
        # `on_mount` claimed the switch; `_start` releases it
        if kind == "pick":
            return "resume" if await self._pick_session() else "startup"
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
        shown = False
        if info is not None:
            shown = await self._show_session(info)  # redraws the transcript, so the warning comes after it
        await self._warn_skipped(store)
        return "resume" if shown else "startup"

    async def _warn_skipped(self, store: SessionStore) -> None:
        if store.skipped:
            await self.query_one(Transcript).add_notice("warn", f"Skipped {len(store.skipped)} unreadable session file(s)")

    def _cancel_diff(self) -> None:
        """A turn, an undo or a switch is starting: a diff still being built would describe the turn before it."""
        self.workers.cancel_group(self, "diff")

    def _begin_switch(self) -> bool:
        """Claim the right to switch sessions: False while a turn runs or another switch is under way."""
        if self._turn_active or self._switching:
            return False
        self._switching = True  # no message is accepted until the switch ends
        self._cancel_diff()
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

    def _skills_summary(self) -> str:
        """`<name> (<source>): <description>` per skill, then the launch warnings (file text, so escaped)."""
        lines = [
            f"{escape(s.name)} ({s.source}): {escape_display(' '.join(s.description.split()))}" for s in self.skills.sorted()
        ]
        lines.extend(self.skills.warnings)
        return "\n".join(lines) if lines else "No skills."

    def _agents_summary(self) -> str:
        """Per agent `<name> (<source>)[ model <model>]: <description>` and its tools, then the launch warnings."""
        lines: list[str] = []
        for a in self.agents.sorted():
            shown = a.resolved_model or a.model
            model = f" model {escape_display(shown)}" if shown else ""
            if a.missing:
                model += " (not installed)"
            lines.append(f"{escape(a.name)} ({a.source}){model}: {escape_display(' '.join(a.description.split()))}")
            tools = READ_ONLY_TOOLS if a.tools is None else a.tools
            lines.append(f"  tools: {', '.join(tools) or 'none'}")
        lines.extend(self.agents.warnings)
        return "\n".join(lines) if lines else "No agents."

    def _guardrails_summary(self) -> str:
        """What the guardrails do in this run. Names and paths only, never a value; all of it escaped."""
        guard = self.guard
        lines: list[str] = []
        # `.env` and `.env.*` are shown as one `.env*` (matching is unchanged: the template files stay unprotected)
        names = [".env*" if name == ".env" else name for name in BUILTIN_NAMES if name != ".env.*"]
        builtin = [pattern.removesuffix("/**") for pattern in BUILTIN_HOME] + names
        extra = guard.paths.extra_count
        lines.append("Protected: " + ", ".join(builtin) + (f" (+{extra} from settings)" if extra else ""))
        homes = guard.paths.home_folders
        if len(homes) > 1:  # `$HOME` and the account's recorded home differ: `~` in a pattern means both
            lines.append("Home folders: " + ", ".join(escape(str(home)) for home in homes))
        if guard.paths.tracking_truncated:
            lines.append(
                f"Hard links to protected files are tracked for the first {MAX_TRACKED_FILES:,} files only "
                f"(and {MAX_TRACKED_DEPTH} folder levels, {MAX_TRACKED_DIRS:,} folders) under protected folders "
                "and, for protected names, in the workspace"
            )
        removed = ", ".join(escape(name) for name in guard.removed_env) or "none"
        lines.append(f"Removed from child environment: {removed}")
        lines.append(f"Redacted this run: {guard.redactions}")
        if self.config_guard is not None:  # the changes the config watch saw (escaped there) and the global marker
            lines.extend(self.config_guard.summary())
        return "\n".join(lines)

    def _mcp_summary(self) -> str:
        """`<name> (<source>) <status>[: reason]` per server, the tool names of a ready one indented below it."""
        states = self.mcp.states() if self.mcp is not None else []
        if not states:
            return "No MCP servers configured (global: ~/.config/agent/mcp.json, project: .agent/mcp.json)"
        lines: list[str] = []
        for state in states:
            config = state.config
            line = f"{config.name} ({config.source}) {state.status}"
            line = f"{line}: {state.reason}" if state.reason else line
            if config.overrides_global:
                line += f" · overrides the global '{config.name}'"
            lines.append(line)
            if state.status == "ready":
                lines.extend(f"  {name}" for name in dict.fromkeys(tool_name(config.name, i.name) for i in state.tools))
        return "\n".join(lines)

    async def _refuse_switch(self) -> None:
        await self.query_one(Transcript).add_notice("warn", self._busy_text("/sessions"))

    async def _pick_session(self) -> bool:
        """List this folder's sessions in a picker (a worker: it waits for the screen), then switch to the choice.
        True when a session was switched to.

        The caller has claimed the switch (`_begin_switch`, or `on_mount` for the start) and releases it.
        """
        store, transcript = self.store, self.query_one(Transcript)
        if store is None:
            return False
        infos = await asyncio.to_thread(store.list)
        await self._warn_skipped(store)
        if not infos:
            await transcript.add_notice("info", "No saved sessions for this folder")
            return False
        chosen = await self.push_screen_wait(SessionPicker(infos))
        if chosen is None:
            return False
        return await self._show_session(next(info for info in infos if info.id == chosen))

    async def _model_command(self, name: str) -> None:
        """The `/model <name>` worker: switch, then release the claim `_begin_switch` took."""
        try:
            await self._switch_model(name)
        except Exception as error:  # a bug must not leave the app claimed or kill it
            await self.query_one(Transcript).add_notice("error", f"Model switch failed: {escape(str(error))}")
        finally:
            self._switching = False

    def _picking(self) -> bool:
        """A bare `/model` is loading or its picker is open."""
        return any(w.group == "model-pick" and not w.is_finished for w in self.workers) or any(
            isinstance(screen, ModelPicker) for screen in self.screen_stack
        )

    async def _pick_model(self) -> None:
        """The bare `/model` worker: list, read the lengths, pick, then switch. The input stays usable until the
        switch itself, which takes the claim."""
        transcript = self.query_one(Transcript)
        try:
            if self.catalog is None:
                await transcript.add_notice("error", "No model catalog is available")
                return
            try:
                infos = await self.catalog.list()
            except CatalogError as error:
                await transcript.add_notice("error", f"Could not list the installed models: {error}")
                return
            if not infos:
                await transcript.add_notice("info", "No models installed.")
                return
            catalog = self.catalog

            async def look(info: ModelInfo) -> tuple[ModelInfo, tuple[int | None, str | None]]:
                try:
                    async with asyncio.timeout(SHOW_TIMEOUT):
                        length = await catalog.context_length(info.name)
                    return replace(info, context_length=length), (length, None)
                except CatalogError as error:
                    return info, (None, str(error))
                except TimeoutError:
                    return info, (None, f"Ollama did not answer within {SHOW_TIMEOUT:g} s")

            looked = await asyncio.gather(*(look(info) for info in infos))
            known = {info.name: result for info, (_, result) in zip(infos, looked, strict=True)}
            if self._turn_active or self._switching:
                await transcript.add_notice("warn", self._busy_text("/model"))
                return
            if len(self.screen_stack) > 1:  # another dialog came up while the list loaded
                await transcript.add_notice("warn", "Another dialog is open; run /model again after closing it")
                return
            current = installed_name(self.provider.model, [i.name for i in infos])
            chosen = await self.push_screen_wait(ModelPicker([info for info, _ in looked], current))
            if chosen is None:
                return
            if not self._begin_switch():
                await transcript.add_notice("warn", self._busy_text("/model"))
                return
            try:
                await self._switch_model(chosen, known.get(chosen))
            finally:
                self._switching = False
        except Exception as error:  # a bug must not kill the app
            await transcript.add_notice("error", f"Model switch failed: {escape(str(error))}")

    async def _switch_model(self, name: str, looked: tuple[int | None, str | None] | None = None) -> None:
        """Make `name` (an installed model) the one every later request uses, with a window that fits it, and
        remember it for the project. Ollama failing leaves the old model and window in place."""
        transcript, status = self.query_one(Transcript), self.query_one(StatusBar)
        if self.catalog is None:
            await transcript.add_notice("error", "No model catalog is available")
            return
        try:
            installed = [info.name for info in await self.catalog.list()]
        except CatalogError as error:
            await transcript.add_notice("error", f"Could not list the installed models: {error}")
            return
        target = installed_name(name, installed)
        if target is None:
            listed = ", ".join(escape_display(n) for n in installed) or "none"
            await transcript.add_notice("error", f"Model {shown_name(name)} is not installed. Installed: {listed}")
            return
        if installed_name(self.provider.model, installed) == target:
            await transcript.add_notice("info", f"Already using {escape_display(target)}.")
            return
        length: int | None = None
        failed: str | None = None
        if looked is not None:  # the picker already asked
            length, failed = looked
        else:
            try:
                async with asyncio.timeout(SHOW_TIMEOUT):
                    length = await self.catalog.context_length(target)
            except CatalogError as error:
                failed = str(error)
            except TimeoutError:
                failed = f"Ollama did not answer within {SHOW_TIMEOUT:g} s"
        window = resolve_window(length, self.window_cap)
        self.provider.switch(target, window)
        status.model = escape_display(target)
        status.window = window
        await transcript.add_notice("info", model_notice(target, window))
        if failed is not None:
            await transcript.add_notice(
                "warn",
                f"Could not read the context length of {escape_display(target)}; using AGENT_NUM_CTX ({self.window_cap}): {failed}",
            )
        recorded: list[Any] = []
        self.agent.model_switched(target, window, recorded.append)
        for event in recorded:
            if isinstance(event, Notice):
                await transcript.add_notice(event.level, event.text)
        try:
            self.permissions.save_project_key("model", target)
        except RuleSavedNotTrusted as error:
            await transcript.add_notice("warn", escape(str(error)))
        except (OSError, ValueError) as error:
            await transcript.add_notice("warn", f"Model saved for the run only: {escape(str(error))}")

    async def _sessions_command(self) -> None:
        """The `/sessions` worker: pick and switch, then release the claim `_begin_switch` took."""
        try:
            if await self._pick_session():
                await self._session_started("switch")
        finally:
            self._switching = False

    async def _show_session(self, info: SessionInfo) -> bool:
        """Load `info`, make it the live session and redraw its transcript. The switching flag is held by the caller.
        False when the session could not be opened (the current one stays)."""
        store, transcript = self.store, self.query_one(Transcript)
        if store is None:
            return False
        try:
            loaded = await asyncio.to_thread(store.load, info.id)
            recorder = FileRecorder.resume(loaded.info.path, self.workspace)
        except (SessionError, SessionWriteError) as exc:  # stay on the session we have
            await transcript.add_notice("error", str(exc))
            return False
        try:
            if self._turn_active:  # a turn started while the file was read
                raise AgentBusy
            old = self.agent.reset(
                recorder=recorder,
                history=loaded.history,
                tool_meta=loaded.tool_meta,
                todos=loaded.todos,
                usage=loaded.usage,
                unrecorded=loaded.unrecorded,
            )
        except AgentBusy:
            await self._refuse_switch()
            return False
        self.query_one(TodoPanel).show(self.agent.todos)
        self._pending.clear()  # pasted images belong to the old session
        self._show_pending()
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
        return True

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
            and isinstance(next(iter(reversed(self.screen_stack)), None), ApprovalScreen | McpTrustScreen | PermissionTrustScreen | GlobalTrustScreen)  # empty on shutdown
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
        if self.terminals is None and self.mcp is None:
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
        if self.config_guard is not None:  # the transcript is going away: what is said now is printed after the app
            self.config_guard.unlisten(self._config_changed)
        with contextlib.suppress(Exception):  # a failing check must not keep the cleanup below from running
            self.agent.watch.check(Cause.background())  # first: the shell tab and background processes still run
        self.file_index.cancel()  # the build thread stops at its next check instead of walking on
        self.agent.close()
        self.paste.remove()
        try:
            if self.mcp is not None:
                await self.mcp.close_all()  # no server process outlives the app, whatever ended it
        finally:
            try:
                if self.terminals is not None:
                    await self.terminals.close_all()  # nor any terminal process
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
        self.agent.set_mode(self.agent.mode.next())
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
            return self.terminals is not None and not any(isinstance(s, DIALOGS) for s in stack)
        if action in ("terminal_tab", "close_terminal_tab"):
            return self._tab_key_applies(action, stack)
        if action in self.TERMINAL_KEYS and stack and isinstance(stack[-1].focused, TerminalView):
            return False  # the key goes on to the terminal
        if action in ("toggle_mode", "interrupt_turn") and stack:
            focused = stack[-1].focused
            if isinstance(focused, ChatInput) and focused.popup_open:
                return False  # Tab and Esc belong to the open suggestions: the input handles them
        if action == "interrupt_turn":  # with a dialog open Escape belongs to it (the approval denies, then interrupts)
            return not (stack and isinstance(stack[-1], DIALOGS))
        if action == "toggle_mode":  # the mode is not changed under a dialog, nor Tab taken from it
            return not any(isinstance(screen, DIALOGS) for screen in stack)
        return True

    def _tab_key_applies(self, action: str, stack: list[Any]) -> bool:
        """Ctrl+PageUp/PageDown and Ctrl+W work while the pane is shown. Ctrl+W is also readline's and the
        input's "delete word", so it closes only a tab nobody types into: that of an exited terminal that has
        focus, or any tab while the tab bar has it (not the input, the transcript, a running terminal or nothing)."""
        if self.terminals is None or not stack or any(isinstance(s, DIALOGS) for s in stack):
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
        if self._turn_active or any(w.group == "session-start" and not w.is_finished for w in self.workers):
            self.interrupt_turn()
        else:
            self.exit()

    def interrupt_turn(self) -> None:
        if self._undoing:  # an undo is not a turn: there is nothing to cancel, and its guard must stay up
            return
        if self._pending and not self._turn_active and not self._switching and self._input_empty():
            self._pending.clear()  # Esc on an empty input: only the pasted images go
            self._show_pending()
            return
        self.workers.cancel_group(self, "turn")
        self.workers.cancel_group(self, "session-start")
        self.workers.cancel_group(self, "model-pick")  # only while the list loads: an open picker takes Esc itself
        if self._turn_active:
            self._interrupted = True  # also after `done`, while the worker still finishes (`/init`'s file check)
        if self._turn_active and not self._turn_entered:
            # Cancelled before its first line ran: the `finally` in `_turn` never will.
            self._end_turn()

    def _input_empty(self) -> bool:
        with contextlib.suppress(NoMatches):
            return not self.query_one(ChatInput).text
        return False

    def _show_pending(self) -> None:
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            self.query_one(AttachmentPanel).show([label for _, label, _ in self._pending])

    async def on_chat_input_paste_image(self, event: ChatInput.PasteImage) -> None:
        self.run_worker(self._paste_image(), group="paste")

    async def _paste_image(self) -> None:
        """ctrl+v: an image on the clipboard becomes a chip; anything else is the input's ordinary text paste."""
        transcript = self.query_one(Transcript)
        chat = self.query_one(ChatInput)
        try:
            if len(self._pending) >= MAX_IMAGES:
                await transcript.add_notice("warn", f"not attached clipboard (image limit ({MAX_IMAGES} per message))")
                return
            try:
                path = await clipboard.read_clipboard_image(self.paste.path())
            except ClipboardUnsupported:
                await transcript.add_notice("info", "clipboard images need macOS")
            except ClipboardError as error:
                await transcript.add_notice("warn", f"could not read the clipboard: {escape(error.reason)}")
            else:
                if path is not None:
                    try:
                        label, ident = await asyncio.to_thread(_label_and_identity, path)
                    except OSError:
                        _unlink(path)
                        raise
                    if len(self._pending) >= MAX_IMAGES:  # another paste filled the list while this one read
                        _unlink(path)
                        await transcript.add_notice("warn", f"not attached clipboard (image limit ({MAX_IMAGES} per message))")
                        return
                    self._pending.append((path, label, ident))
                    self._show_pending()
                    return
            chat.action_paste()  # no image: today's text paste
        except OSError as error:  # the paste folder or the file could not be used
            with contextlib.suppress(NoMatches):
                await transcript.add_notice("warn", f"could not read the clipboard: {escape(str(error))[:200]}")
                chat.action_paste()
        except NoMatches:
            pass

    async def _take_pasted(self) -> tuple[list[Image], list[str]]:
        """The pending images, read now; the chips go. A file that is gone or no image is a warning."""
        taken, self._pending = self._pending, []
        self._show_pending()
        images: list[Image] = []
        warnings: list[str] = []
        for path, _, ident in taken:
            try:
                if await asyncio.to_thread(_identity, path) != ident:
                    warnings.append("not attached clipboard (file changed)")
                    continue
                image = await asyncio.to_thread(_read_pasted, path)
            except FileNotFoundError:
                warnings.append("not attached clipboard (file is gone)")
            except ImageError as error:
                warnings.append(f"not attached clipboard ({error.reason})")
            except OSError:
                warnings.append("not attached clipboard (unreadable)")
            else:
                if image is None:
                    warnings.append("not attached clipboard (not an image)")
                else:
                    images.append(image)
        return images, warnings

    def _restore_draft(self, text: str) -> None:
        """Put a message that was not sent back in the input, ahead of anything typed since."""
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            chat = self.query_one(ChatInput)
            chat.load_text(f"{text}\n{chat.text}" if chat.text else text)
            chat.move_cursor(chat.document.end)

    def _queueing(self) -> bool:
        """A message sent now waits for the running turn, `/init` or `/compact` (not for an undo or a session load)."""
        return self._turn_active and not self._undoing and not self._switching and not self._workflow_active

    def _show_queue(self) -> None:
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            self.query_one(QueuePanel).show(self._queue.entries)

    def _end_turn(self) -> None:
        """Every turn and compaction ends here: a `done` end sends the queue as one turn, any other puts it back."""
        self._turn_active = False
        if self._workflow_active:
            self._workflow_active = False
            with contextlib.suppress(NoMatches):
                self.query_one(StatusBar).workflow = ""
        self._refresh_hints()
        done = self._job_status == "done" and not self._interrupted
        self._job_status = None
        self._interrupted = False
        queued = self._queue.take()
        draft, self._draft = self._draft, None  # set when the turn ended before it sent the message
        with contextlib.suppress(NoMatches):  # the status bar is gone when the app is shutting down
            self.query_one(StatusBar).compacting = False
        if queued and done and draft is None and not self._shutting_down:
            self._show_queue()
            merged = merge(queued)
            self._begin_turn(merged.prompt, merged.draft, label=merged.label)  # busy stays on
            return
        back = [text for text in (draft, merge(queued).draft if queued and not self._shutting_down else None) if text]
        if back:
            self._restore_draft("\n".join(back))
        self._show_queue()
        with contextlib.suppress(NoMatches):
            self.query_one(StatusBar).busy = False

    # turns

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text
        if text.strip().startswith("/"):
            await self._command(text)
            return
        if self._queueing():
            if text.strip():
                self._queue.add(Queued(text, text, None))
                self._show_queue()
            return
        if await self._refuse_if_busy(text):
            return
        self._begin_turn(text, text)

    async def _refuse_if_busy(self, text: str) -> bool:
        """A message cannot start now: it goes back in the box, with a notice. True when it was refused."""
        if not (self._turn_active or self._switching):
            return False
        self._restore_draft(text)  # not accepted: the message stays in the box
        if self._workflow_active:
            await self.query_one(Transcript).add_notice("warn", WORKFLOW_RUNNING)
            return True
        hint = "press Esc to interrupt" if self._turn_active else "a session is loading"
        await self.query_one(Transcript).add_notice("warn", f"busy — {hint}")
        return True

    def _begin_turn(self, text: str, draft: str, init: bool = False, label: str | None = None) -> None:
        """Start the turn worker for `text`; `draft` is what the box gets back if it never sends."""
        self._turn_active = True
        self._refresh_hints()
        self._job_status = None
        self._interrupted = False
        self._cancel_diff()
        self._draft = draft
        self.query_one(StatusBar).busy = True
        self._turn_entered = False
        # A partial, not a coroutine: cancelled before it first runs, nothing is left un-awaited.
        self.run_worker(functools.partial(self._turn, text, init, draft, label), group="turn")

    async def _command(self, original: str) -> None:
        transcript = self.query_one(Transcript)
        found = _COMMAND.match(original.strip())
        name = f"/{found.group(1)}" if found else "/"
        rest = (found.group(2) or "") if found else ""
        match name:
            case "/help":
                await transcript.add_notice("info", self._help())
            case "/mcp":
                await transcript.add_notice("info", self._mcp_summary())
            case "/permissions":
                lines = self.permissions.listing()
                await transcript.add_notice("info", "\n".join(lines) if lines else "no permission rules")
            case "/guardrails":
                await transcript.add_notice("info", self._guardrails_summary())
            case "/hooks":
                lines = self.hooks.listing()
                await transcript.add_notice("info", "\n".join(lines) if lines else "No hooks.")
            case "/skills":
                await transcript.add_notice("info", self._skills_summary())
            case "/agents":
                await transcript.add_notice("info", self._agents_summary())
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
                self.query_one(TodoPanel).show(self.agent.todos)
                self._pending.clear()  # pasted images belong to the old session
                self._show_pending()
                self._switching = True  # no message is accepted until the notice is back in the new transcript
                self._cancel_diff()
                handed = False  # the worker that runs the SessionStart hook releases the claim
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
                    self.run_worker(self._finish_new_session(), group="new-session", exit_on_error=False)
                    handed = True
                finally:
                    if not handed:
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
                self._job_status = None
                self._interrupted = False
                self._cancel_diff()
                self._turn_entered = False
                self.query_one(StatusBar).busy = True
                # Like a turn: Esc cancels the "turn" group, and the same events reach the transcript.
                self.run_worker(self._compact, group="turn")
            case "/undo":
                await self._undo_command()
            case "/diff":
                await self._diff_command(rest)
            case "/usage":
                await self._usage_command(rest)
            case "/images":
                await self._images_command(rest)
            case "/workflow":
                await self._workflow_command(rest)
            case "/model":
                if not rest.strip():
                    if self._turn_active or self._switching:
                        await transcript.add_notice("warn", self._busy_text("/model"))
                        return
                    if self._picking():
                        await transcript.add_notice("warn", "The model list is already open or loading")
                        return
                    self.run_worker(self._pick_model, group="model-pick", exit_on_error=False)
                    return
                if not self._begin_switch():
                    await transcript.add_notice("warn", self._busy_text("/model"))
                    return
                self.run_worker(functools.partial(self._model_command, rest.strip()), group="model", exit_on_error=False)
            case _:
                custom = self.commands.get(name[1:])
                if custom is None:
                    await transcript.add_notice("warn", f"Unknown command: {name} (try /help)")
                    return
                queueing = self._queueing()
                if not queueing and await self._refuse_if_busy(original):
                    return
                prompt = custom.expand(rest)
                if not prompt.strip():
                    await transcript.add_notice("warn", f"{name} is empty")
                    return
                if queueing:
                    self._queue.add(Queued(original, prompt, f"{name} ({custom.source})"))
                    self._show_queue()
                    return
                self._begin_turn(prompt, draft=original, label=f"{name} ({custom.source})")

    def _suggestion_entries(self) -> list[tuple[str, str, str | None]]:
        entries: list[tuple[str, str, str | None]] = [(name, text, None) for name, text in BUILTINS.items()]
        entries.extend((c.name, c.description or "", c.source) for c in self.commands.sorted())
        return entries

    def _reindex(self) -> None:
        """Build the `@` index in a thread (once at mount, again after each turn). A request that arrives while a
        build runs makes it run once more when it ends, so files created meanwhile are not missed."""
        if self._shutting_down:
            return
        if self._indexing:
            self._reindex_pending = True
            return
        self._indexing = True
        self.run_worker(self._build_index(), group="file-index", exit_on_error=False)

    async def _build_index(self) -> None:
        self._indexing = True
        try:
            while True:
                self._reindex_pending = False
                await asyncio.to_thread(self.file_index.build)
                if self._shutting_down:
                    return
                with contextlib.suppress(NoMatches, ScreenStackError):  # the app may be closing
                    self._update_suggestions(self.query_one(ChatInput))
                if not self._reindex_pending:
                    return
        finally:
            self._indexing = False

    def _update_suggestions(self, box: ChatInput) -> None:
        """Command names for a bare `/prefix`, otherwise paths for the `@token` at the cursor; never both."""
        if box.suggestions is not None:
            box.suggestions.update_for(box.text, self._suggestion_entries())
        self._update_paths(box)

    def _update_paths(self, box: ChatInput) -> None:
        if box.paths is None:
            return
        token = None if box.suggestions is not None and box.suggestions.is_open else box.token_before_cursor()
        box.paths.update_for(self.file_index.suggest(token) if token is not None else [])

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if isinstance(event.text_area, ChatInput):
            self._update_suggestions(event.text_area)

    def on_text_area_selection_changed(self, event: TextArea.SelectionChanged) -> None:
        if isinstance(event.text_area, ChatInput) and event.text_area.paths is not None:
            self._update_paths(event.text_area)  # a cursor move only changes the token; the command list reads the text

    def on_command_suggestions_toggled(self, event: CommandSuggestions.Toggled) -> None:
        self._refresh_hints()

    def on_path_suggestions_toggled(self, event: PathSuggestions.Toggled) -> None:
        self._refresh_hints()

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        if isinstance(event.widget, ChatInput):
            if event.widget.suggestions is not None:
                event.widget.suggestions.hide()  # the popup belongs to the focused input
            if event.widget.paths is not None:
                event.widget.paths.hide()
        self._refresh_hints()

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if isinstance(event.widget, ChatInput):
            self._update_suggestions(event.widget)
        self._refresh_hints()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if isinstance(event.option_list, CommandSuggestions):
            event.stop()
            chat = self.query_one(ChatInput)
            if (name := event.option_list.name_at(event.option_index)) is not None:
                chat.complete(name)
            chat.focus()
        elif isinstance(event.option_list, PathSuggestions):
            event.stop()
            chat = self.query_one(ChatInput)
            if (choice := event.option_list.at(event.option_index)) is not None:
                chat.complete_path(choice)
            chat.focus()

    def _help(self) -> str:
        """The key and command help, plus the custom commands (names, descriptions and paths come from files: escaped)."""
        if not self.commands.by_name:
            return HELP
        lines = [
            f"/{escape(c.name)}" + (f"  {escape(c.description)}" if c.description else "") + f" ({c.source})"
            for c in self.commands.sorted()
        ]
        return f"{HELP}\nCustom commands:\n" + "\n".join(lines)

    # workflows

    def _workflow_runner(self) -> WorkflowRunner:
        if self._run_index is None:
            self._run_index = RunIndex.for_workspace(self.workspace)
        if self._timings is None:
            self._timings = TimingStore.default()
        return WorkflowRunner(
            self.agent, self.agent.all_tools, self.agent.ctx, self.agent.rules, self._run_index, timings=self._timings
        )

    def _workflow_listing(self) -> str:
        lines = [
            f"{escape(w.name)} ({w.source}): {escape_display(' '.join(w.description.split()))}"
            + (f" — input: {escape_display(' '.join(w.input_help.split()))}" if w.input_help else "")
            for _, w in sorted(self.workflows.items.items())
        ]
        lines.extend(self.workflows.warnings)
        return "\n".join(lines) if lines else "No workflows."

    async def _workflow_command(self, rest: str) -> None:
        """`/workflow` lists; `run <name> <input>`, `resume [id]` start a worker in the turn group; `cancel` stops it."""
        transcript = self.query_one(Transcript)
        words = rest.split(maxsplit=2)
        sub = words[0] if words else ""
        if not words or (sub == "list" and len(words) == 1):
            await transcript.add_notice("info", self._workflow_listing())
            return
        if sub == "cancel" and len(words) == 1:
            if self._workflow_active:
                self.interrupt_turn()
            else:
                await transcript.add_notice("info", "no workflow is running")
            return
        resume = _parse_resume(rest.split()[1:]) if sub == "resume" else None
        valid = (sub == "run" and len(words) in (2, 3)) or resume is not None
        if not valid:
            await transcript.add_notice("warn", WORKFLOW_USAGE)
            return
        if self._workflow_active:
            await transcript.add_notice("warn", WORKFLOW_RUNNING)
            return
        if self._turn_active or self._switching:
            await transcript.add_notice("warn", self._busy_text("/workflow"))
            return
        if self.agent.mode is Mode.PLAN:
            await transcript.add_notice("warn", "workflows need build or edits mode — press Tab to switch")
            return
        if self.images is not None and self.images.busy:
            await transcript.add_notice("warn", "an image pull is running (/images cancel stops it)")
            return
        runner = self._workflow_runner()
        if sub == "run":
            wf = self.workflows.items.get(words[1])
            if wf is None:
                await transcript.add_notice("warn", f"no workflow named '{escape_display(words[1])}' (/workflow lists them)")
                return
            text = words[2] if len(words) == 3 else ""

            def action(emit: EventSink) -> Awaitable[RunResult]:
                return runner.run(wf, text, emit, self._approve_plan)
        else:
            assert resume is not None
            run_id, redo = resume

            def action(emit: EventSink) -> Awaitable[RunResult]:
                return runner.resume(self.workflows.items.get, run_id, emit, self._approve_plan, redo)

        self._turn_active = True
        self._workflow_active = True
        self._job_status = None
        self._interrupted = False
        self._cancel_diff()
        self._turn_entered = False
        self.query_one(StatusBar).busy = True
        self._refresh_hints()
        self.run_worker(functools.partial(self._workflow, action, runner), group="turn", exit_on_error=False)

    async def _workflow(self, action: Callable[[EventSink], Awaitable[RunResult]], runner: WorkflowRunner) -> None:
        """The worker of a run: events reach the widgets in order, the outcome is one last line."""
        self._turn_entered = True
        try:

            async def work(emit: EventSink) -> None:
                try:
                    result = await action(emit)
                except asyncio.CancelledError:
                    if runner.saved:
                        emit(WorkflowLine("warn", "workflow interrupted (/workflow resume continues a saved run)"))
                    else:
                        emit(WorkflowLine("warn", "workflow cancelled; nothing was saved"))
                    raise
                except Exception as exc:
                    try:
                        text = self.guard.redactor.redact(str(exc))[0]
                    except Exception:
                        text = ""
                    emit(Notice("error", f"workflow failed: {one_line(text, FAILED_SHOWN)}"))
                    return
                emit(WorkflowLine("info" if result.ok else "warn", result.line))

            await self._run_with_events(work)
        finally:
            self._end_turn()

    async def _approve_plan(self, plan: PlanSummary) -> bool:
        """The one approval of a run: the plan as a content preview; Allow once or Deny, never Always."""
        body = self._plan_text(plan)
        title = self._plan_title(plan.workflow, plan.folder)
        request = ApprovalRequest("workflow", title, Preview("content", title, body), allow_always=False)
        decision = await self.agent.approver.ask(request)
        return decision.choice is Choice.ONCE

    @staticmethod
    def _plan_title(workflow: str, folder: str) -> str:
        return f"workflow {escape_display(workflow)} → {escape_display(folder)}"

    def _plan_text(self, plan: PlanSummary) -> str:
        counts = ", ".join(f"{n} {escape_display(tool)}" for tool, n in plan.counts.items())
        minutes = max(1, round(plan.estimate_s / 60))
        lines = [f"{len(plan.calls)} calls: {counts}"]
        if plan.reused:
            lines.append(f"reused {plan.reused} calls")
        lines.extend(one_line(note, PLAN_NOTE_SHOWN) for note in plan.notes)
        lines += [
            f"about {minutes} min ({escape_display(plan.estimate_source)})",
            f"files go to {escape_display(plan.folder)}",
            "",
        ]
        grant = {tool.name: tool.grant_arg for tool in self.agent.all_tools()}
        for call in plan.calls[:PLAN_LINES]:
            path = call.args.get("path")
            key = next(
                (
                    call.args[name]
                    for name in (grant.get(call.tool), "prompt", "text")
                    if name and name != "path" and isinstance(call.args.get(name), str)
                ),
                None,
            )
            line = f"{escape_display(call.key)}  {escape_display(call.tool)}  {escape_display(str(path)) if path else ''}"
            if key is not None:
                line += f"  {one_line(key, PLAN_ARG_SHOWN)}"
            lines.append(line)
        if len(plan.calls) > PLAN_LINES:
            lines.append(f"and {len(plan.calls) - PLAN_LINES} more calls")
        text = "\n".join(lines)
        try:
            return self.guard.redactor.redact(text)[0]
        except Exception:
            return "the plan could not be shown (redaction failed)"

    async def _usage_command(self, rest: str) -> None:
        """Open the totals now: they are in memory, so there is nothing to wait for and a running turn is no obstacle."""
        if rest.strip():
            await self.query_one(Transcript).add_notice("warn", "/usage takes no arguments")
            return
        status = self.query_one(StatusBar)
        recorder_id = getattr(self.agent.recorder, "session_id", "")
        data = summary(
            self.agent.usage,
            session_id=recorder_id if isinstance(recorder_id, str) else "",
            model=self.provider.model,
            context_used=status.used,
            window=status.window,
            turns=len(turn_starts(self.agent.history)),
            unrecorded=self.agent.usage_unrecorded,
            estimated=status.estimated,
        )
        await self.push_screen(UsageScreen(data))

    async def _images_command(self, rest: str) -> None:
        """Status, pull and cancel. Allowed at any time: a pull is its own background job and touches no history."""
        transcript = self.query_one(Transcript)
        images = self.images
        if images is None or not images.installed:
            await transcript.add_notice("info", NOT_INSTALLED)
            return
        words = rest.split()
        if not words:
            await self._image_notice("info", format_status(images.status(), images.pulling))
            return
        if words[0] == "cancel" and len(words) == 1:
            if not await images.cancel():
                await transcript.add_notice("info", "no image pull is running")
            return
        choices: dict[str, tuple[Role, ...]] = {
            "": ROLES,
            "gen": ("generate",),
            "edit": ("edit",),
            "upscale": ("upscale",),
        }
        roles = choices.get(words[1] if len(words) == 2 else "") if words[0] == "pull" and len(words) <= 2 else None
        if roles is None:
            await transcript.add_notice("warn", "usage: /images [pull [gen|edit|upscale] | cancel]")
            return
        if self._workflow_active:
            await transcript.add_notice("warn", "a workflow is running: /images pull waits until it ends (/images and /images cancel still work)")
            return
        if images.busy:
            await transcript.add_notice("warn", "an image pull is already running")
            return
        images.reserve()
        try:
            self.run_worker(self._image_pull(images, roles), group="image-pull", exit_on_error=False)
        except BaseException:
            images.release()
            raise

    async def _image_notice(self, level: Level, text: str) -> None:
        """A transcript notice with secrets out of it (fail closed); display only, never in the session."""
        if self._shutting_down:
            return
        try:
            shown = self.guard.redactor.redact(text)[0]
        except Exception:
            level, shown = "warn", "image pull failed: redaction failed"
        await self.query_one(Transcript).add_notice(level, shown)

    async def _image_pull(self, images: ImagePuller, roles: tuple[Role, ...]) -> None:
        def progress(text: str) -> None:
            if not self._shutting_down:
                self.query_one(StatusBar).background = text

        try:
            await images.pull(roles, self._image_notice, progress)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._image_notice("warn", f"image pull failed: {escape(str(exc)[:200])}")

    async def _diff_command(self, rest: str) -> None:
        transcript = self.query_one(Transcript)
        if rest.strip():
            await transcript.add_notice("warn", "/diff takes no arguments")
            return
        if self._turn_active or self._switching:
            await transcript.add_notice("warn", self._busy_text("/diff"))
            return
        self.run_worker(self._diff, group="diff", exclusive=True, exit_on_error=False)

    async def _diff(self) -> None:
        """Build the change set off the UI path, then open the viewer (or say why not)."""
        transcript = self.query_one(Transcript)
        try:
            diff = await self.agent.last_turn_diff()
        except AgentBusy:
            await transcript.add_notice("warn", self._busy_text("/diff"))
            return
        except Exception as error:
            await transcript.add_notice("error", f"Could not build the diff: {escape(str(error)[:200])}")
            return
        if diff is None:
            await transcript.add_notice("info", NOTHING)
            return
        if not diff.items:
            await transcript.add_notice("info", NO_FILES)
            if diff.fallback:
                await transcript.add_notice("warn", BANNER)
            return
        # Keys and messages were handled while the list was built: look again, with no await before the push.
        if self._turn_active or self._switching:
            await transcript.add_notice("warn", self._busy_text("/diff"))
            return
        if not self.agent.is_current_diff(diff):  # a newer turn, an undo or a reset: this is not the turn to show
            return
        await self.push_screen(DiffScreen(diff, self.agent.load_diff_file))

    async def _undo_command(self) -> None:
        transcript = self.query_one(Transcript)
        if self._turn_active or self._switching:
            await transcript.add_notice("warn", self._busy_text("/undo"))
            return
        status = self.query_one(StatusBar)
        self._turn_active = True
        self._cancel_diff()
        self._undoing = True
        status.busy = True
        self._refresh_hints()
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
            self._refresh_hints()
        if prompt is not None:
            await transcript.remove_last_turn()
            self.query_one(TodoPanel).show(self.agent.todos)
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

    async def _turn(
        self, text: str, init: bool = False, draft: str | None = None, label: str | None = None
    ) -> None:
        self._turn_entered = True
        transcript = self.query_one(Transcript)
        status = self.query_one(StatusBar)
        agents_md = self.workspace / "AGENTS.md"
        before: tuple[int, int] | None = None
        sent = False
        shown = False  # the user entry was drawn (or was being drawn)
        started = False  # the agent was handed the message
        hook_events: list[AgentEvent] = []  # what the UserPromptSubmit hooks reported, shown once the prompt is
        context: list[str] = []  # and what they added for the model
        sent_text = text  # what the hooks and the model get: `text` plus the files its @mentions attach
        notes: tuple[MentionNote, ...] = ()
        images: tuple[Image, ...] = ()
        image_warnings: list[str] = []  # images that were not attached, and why
        pasted_taken: list[tuple[Path, str, tuple[int, int, int, int]]] = []  # pending pasted images this turn took, given back if it never sends
        pasted: list[Image] = []

        async def work(emit: EventSink) -> None:
            nonlocal started
            started = True
            for hook_event in hook_events:
                emit(hook_event)
            await self.agent.run_turn(
                sent_text, emit, draft=draft, context=context, images=images, pasted=tuple(pasted)
            )

        try:
            if status.error and not await self._run_check():  # R3: re-check before giving up on the model
                return
            if not init:  # /init's prompt is the app's own, not something the user sent
                expansion = await asyncio.to_thread(expand_mentions, text, self.workspace, self.guard)
                pasted_taken = list(self._pending)
                pasted, pasted_warnings = await self._take_pasted()
                expansion = replace(expansion, images=(*expansion.images, *pasted))
                expansion, image_warnings = await apply_vision(expansion, self.vision, self.provider.model)
                image_warnings = [*pasted_warnings, *image_warnings]
                sent_text, notes, images = expansion.text, expansion.notes, expansion.images
                decision = await self.agent.prompt_hooks(sent_text, hook_events.append)
                if decision.blocked:  # the prompt is dropped: `_end_turn` puts it back in the box
                    for warning in image_warnings:  # still true when it is sent again, so say it now
                        await transcript.add_notice("warn", warning)
                    for hook_event in hook_events:
                        if isinstance(hook_event, Notice):
                            await transcript.add_notice(hook_event.level, hook_event.text)
                    return
                context = decision.context
            self._draft = None  # from here the message is part of the conversation
            sent = True
            before = fingerprint(agents_md)
            shown = True
            await (transcript.add_user(text, label=label) if label else transcript.add_user(text))
            for note in notes:
                await transcript.add_notice("info" if note.attached else "warn", note_text(note))
            for number, image in enumerate(pasted, 1):  # the numbers edit_image's `clipboard:N` uses
                if any(image is i for i in images):
                    await transcript.add_notice("info", f"image clipboard:{number} ({describe(image)})")
            for warning in image_warnings:
                await transcript.add_notice("warn", warning)
            self._finished_since_toggle = False
            status.pending_mode = False
            await self._run_with_events(work)
        finally:
            if pasted_taken and not started:
                self._pending[:0] = pasted_taken  # the agent never got the message: its images wait for the next one
                self._show_pending()
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
            failed = False

            async def work(emit: EventSink) -> None:
                def watch(event: AgentEvent) -> None:
                    nonlocal failed
                    failed = failed or event == Notice("warn", NOT_SUMMARIZED)
                    emit(event)

                await self.agent.compact(watch)

            if await self._run_with_events(work) and not failed:  # a failed summary is not a finished compaction
                self._job_status = "done"
        finally:
            self._end_turn()

    async def _run_with_events(self, work: Callable[[EventSink], Awaitable[None]]) -> bool:
        """Run an agent job (a turn, a compaction) with its events applied to the widgets in order.
        False when the agent was busy and the job did not run."""
        # The loop's sink is synchronous; a consumer task applies the events to the widgets in order.
        events: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        consumer = asyncio.create_task(self._consume(events))
        ran = True
        try:
            await work(events.put_nowait)
        except AgentBusy:
            ran = False
            events.put_nowait(Notice("warn", "busy — a turn is already running"))
        finally:
            # Whatever ended the job, the queued events still reach the transcript, and a
            # repeated Esc must not cut that short.
            await _uninterrupted(self._finish_turn(events, consumer))
        return ran

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
                status.add_reply_chars(len(delta))
                await transcript.write_assistant(delta)
            case Phase() as phase:
                status.set_phase(phase)
            case ToolStarted(call_id, label, task):
                await transcript.add_tool(call_id, label, task=task)
            case ToolFinished(call_id, result, preview, applied):
                self._finished_since_toggle = True
                if call_id == self._progress_call:
                    status.progress, self._progress_call = "", None
                if (entry := transcript.tool(call_id)) is not None:
                    await entry.finish(result, preview, applied)
            case Notice(level, text):
                await transcript.add_notice(level, text)
            case WorkflowProgress(text):
                status.workflow = text
            case WorkflowLine(level, text):
                await transcript.add_notice(level, text)
            case TaskProgress(call_id, inner):
                # The child's usage and compaction are not the parent's: the status bar keeps the parent's context.
                entry = transcript.tool(call_id)  # None: the block is gone (undone, cleared); its progress is dropped
                match inner:
                    case ToolStarted(child_id, label) if entry is not None:
                        await entry.add_child(child_id, label)
                    case ToolFinished(child_id, result, _, _) if entry is not None:
                        await entry.finish_child(child_id, result)
                    case ToolProgress(text) if entry is not None and not entry.finished:
                        status.progress, self._progress_call = text, call_id  # shown in the status bar
                    case Notice("warn" | "info" as level, text):  # an error ends the call: its result shows it
                        await transcript.add_notice(level, f"task: {text}")
            case TodosChanged(items):
                self.query_one(TodoPanel).show(items)
            case Compacting(active):
                status.compacting = active
            case UsageUpdate(used, window, estimated, _, completion_tokens):
                status.used = used
                status.window = window
                status.estimated = estimated
                if not estimated and completion_tokens:
                    status.set_reply_tokens(completion_tokens)
            case TurnEnded(turn_status):
                self._job_status = turn_status
                status.clear_phase()
                self._reindex()
                status.pending_mode = False
                await transcript.end_assistant()
                await transcript.close_open_tools(turn_status.replace("_", " "))
                if turn_status == "interrupted":
                    await transcript.add_notice("info", "interrupted", icon="⏹")


def _parse_resume(args: list[str]) -> tuple[str | None, str | None] | None:
    """`(run id, redo step)` of `resume [id] [--redo <step>]` (the flag before or after the id), None when malformed."""
    run_id: str | None = None
    redo: str | None = None
    i = 0
    while i < len(args):
        word = args[i]
        if word == "--redo":
            if redo is not None or i + 1 >= len(args) or args[i + 1].startswith("--"):
                return None
            redo = args[i + 1]
            i += 2
        elif word.startswith("--") or run_id is not None:
            return None
        else:
            run_id = word
            i += 1
    return run_id, redo


def _chip_label(path: Path) -> str:
    """`clipboard (PNG, 214 KB)` for a pasted file (its first bytes decide the format)."""
    size = path.stat().st_size
    with path.open("rb") as f:
        fmt = sniff(f.read(12)) or "png"
    return f"clipboard ({fmt.upper()}, {size_text(size)})"


def _identity(path: Path) -> tuple[int, int, int, int]:
    """What a pasted file was when it was added: a replaced or rewritten file differs."""
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _label_and_identity(path: Path) -> tuple[str, tuple[int, int, int, int]]:
    return _chip_label(path), _identity(path)


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _read_pasted(path: Path) -> Image | None:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        return read_image_fd(fd, "clipboard")
    finally:
        os.close(fd)
