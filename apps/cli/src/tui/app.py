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

from core.client import clipboard
from core.client.clipboard import ClipboardError, ClipboardUnsupported, PasteFolder
from core.client.daemon import DaemonError, DaemonUnavailable, ProtocolMismatch
from core.engine import (
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
from core.engine.agents.loader import Agents
from core.engine.approval import ApprovalRequest, Choice, Decision
from core.engine.commands.builtins import BUILTINS
from core.engine.commands.loader import Commands
from core.engine.compaction import turn_starts
from core.engine.configguard import ConfigGuard, HeldGlobal
from core.engine.configwatch import Cause
from core.engine.events import WorkflowLine, WorkflowProgress
from core.engine.guardrails.setup import Guardrails
from core.engine.hooks.setup import Hooks
from core.engine.image_pull import NOT_INSTALLED, ROLES, ImagePuller, Level, Role, format_status
from core.engine.mcp.hub import HubEvent, McpHub
from core.engine.mentions_index import FileIndex
from core.engine.mode import Mode
from core.engine.permissions.setup import Rules
from core.engine.provider.base import Image, SwitchableProvider
from core.engine.provider.images import MAX_IMAGES, ImageError, read_image_fd, size_text, sniff
from core.engine.provider.models import (
    SHOW_TIMEOUT,
    CatalogError,
    ModelCatalog,
    ModelInfo,
    installed_name,
)
from core.engine.session import (
    SessionError,
    SessionInfo,
    SessionStore,
)
from core.engine.skills.loader import Skills
from core.engine.terminal.manager import TerminalError, TerminalManager
from core.engine.text import escape, escape_display, one_line
from core.engine.tools.base import Preview
from core.engine.turn_diff import BANNER, NO_FILES, NOTHING
from core.engine.usage import summary
from core.engine.workflows.loader import Workflows
from core.engine.workflows.model import PlanSummary, RunResult
from core.engine.workflows.runner import WorkflowRunner
from core.engine.workflows.runs import RunIndex
from core.engine.workflows.timings import TimingStore
from core.host.asks import AnswerResult, AskEvent, AskResolved
from core.host.events import (
    Delivering,
    EntryEnd,
    JobEventsDone,
    ModeChanged,
    ModelChanged,
    ProviderState,
    QueueChanged,
    TurnOutcome,
    TurnUndone,
    UserEntry,
)
from core.host.port import LocalPort
from core.host.session import WORKFLOW_RUNNING, SendResult, SessionHost, uninterrupted
from core.trust import TrustAsk, TrustOutcome, apply_trust
from core.wire import models as wm
from textual import events
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import OptionList, Tabs, TextArea

from core import info
from src.tui import NEW, StartChoice
from src.tui.hints import MIN_ROWS, hint_text
from src.tui.remote import NOT_SHARED, DaemonGone, Reconnected, RemotePort
from src.tui.screens.approval import ApprovalScreen
from src.tui.screens.daemon_gone import DaemonGoneScreen
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

DIALOGS = (ApprovalScreen, DaemonGoneScreen, SessionPicker, ModelPicker, McpTrustScreen, PermissionTrustScreen, GlobalTrustScreen, DiffScreen, UsageScreen)  # modal screens (and the diff viewer): the app's own keys leave them alone
KEY_GUARD = 0.5  # seconds an approval dialog that opens over a terminal ignores keys: they were meant for it


HELP = (
    "Enter send · ctrl+j / shift+enter new line · ctrl+v paste an image (macOS) · Esc on an empty input clears pasted images · Tab switch BUILD/PLAN · Enter during a turn queues the message (sent when it ends; Esc puts it back) · Esc interrupt · "
    "ctrl+c interrupt (quit when idle) · ctrl+t terminal pane (show, hide, back to chat) · "
    "ctrl+\\ back to chat · ctrl+pageup / ctrl+pagedown switch terminal tab (these four work inside a terminal too) · "
    "ctrl+w close an exited terminal's tab (or the clicked one) · / suggests commands · /new or /clear new session · "
    "/sessions switch session · /mcp MCP servers and their tools · /permissions saved permission rules · "
    "/guardrails protected paths, environment filter and redaction · /hooks configured hooks · /skills loaded skills · /agents loaded subagents · /compact shrink the context · /model switch model · /init write AGENTS.md · /undo undo the last turn · /diff show the last turn's changes · /usage session token totals · /images image models (status, pull, cancel) · /workflow run a workflow (list, run, resume, cancel) · /help this help · /quit exit"
)

SHARED_INFO = frozenset({"/mcp", "/permissions", "/guardrails", "/hooks", "/skills", "/agents"})  # the daemon's
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
        port: LocalPort | RemotePort,
        *,
        start: StartChoice = NEW,
        on_close: Callable[[], Awaitable[None]] | None = None,
        terminals: TerminalManager | None = None,
        mcp: McpHub | None = None,
        mcp_trust: Callable[[], None] | None = None,
        permission_warnings: Sequence[str] = (),
        commands: Commands | None = None,
        skills: Skills | None = None,
        agents: Agents | None = None,
        launch_notices: Sequence[str] = (),
        paste: PasteFolder | None = None,
        images: ImagePuller | None = None,
        workflows: Workflows | None = None,
        run_index: RunIndex | None = None,
        timings: TimingStore | None = None,
        config_guard: ConfigGuard | None = None,
        held: HeldGlobal | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__()
        # The live session: its turn pipeline, claims and queue are the host's; this app draws them and keeps the
        # input box, the pasted images, the terminals, the workflows and the image pull.
        self.port = port
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
        self.images = images  # `/images`: the models' state and the background warm-up (None: no image tools in this run)
        self._on_close = on_close
        self.terminals = terminals
        self.mcp = mcp
        self._mcp_trust = mcp_trust  # remembers that the user trusted the project's servers
        self._permission_warnings = tuple(permission_warnings)
        self.commands = commands if commands is not None else Commands({}, ())
        self.skills = skills if skills is not None else Skills({}, ())
        self.agents = agents if agents is not None else Agents.empty()
        self.workspace = port.workspace
        if isinstance(port, LocalPort):  # the host's own, under the names the app has always used for them
            host = port.host
            self.permissions = host.permissions
            self.guard = host.guard
            self.hooks = host.hooks
            self.store: SessionStore | None = host.store
        else:  # `--shared`: the rules, hooks and guardrails are the daemon's (its `info` route lists them)
            self.permissions = Rules.empty()
            self.guard = Guardrails.off(self.workspace)
            self.hooks = Hooks.empty()
            self.store = None
            port.on_suggestions = self._paths_refreshed
        self.start = start
        self._wanted_model = model  # `--shared --model`: the daemon's session decides; a different one is said
        self._launch_notices = tuple(launch_notices)  # info lines for the transcript, already escaped
        self._indexing = False
        self._reindex_pending = False
        self._shutting_down = False
        self.paste = paste if paste is not None else PasteFolder(self.workspace)  # removed when the app quits
        self._pending: list[tuple[Path, str, tuple[int, int, int, int]]] = []  # pasted images waiting: (file, chip label, identity)
        self._taken: tuple[int, list[tuple[Path, str, tuple[int, int, int, int]]]] | None = None  # (turn, chips it took)
        self._ask_screens: dict[str, ApprovalScreen | PermissionTrustScreen] = {}  # open ask dialogs by ask id
        self._progress_call: str | None = None  # the call whose live line the status bar shows
        self._finished_since_toggle = False  # a tool finished after a mid-turn mode toggle
        self._signals_before: dict[signal.Signals, Any] = {}  # what SIGTERM and SIGHUP did before the app
        self._keys_ignored_until = 0.0  # time.monotonic() until which an approval dialog takes no keys
        self._closing_tabs: set[str] = set()  # terminals whose tab is being closed (they are being stopped)

    @property
    def shared(self) -> bool:
        """The session lives in the workspace's daemon (`agent --shared`)."""
        return not isinstance(self.port, LocalPort)

    def _local(self) -> LocalPort:
        """The in-process port; what only it has (the host, the agent, the file index) is never used in shared mode."""
        if not isinstance(self.port, LocalPort):
            raise RuntimeError("not available in --shared mode")
        return self.port

    @property
    def host(self) -> SessionHost:
        return self._local().host

    @property
    def agent(self) -> Agent:
        return self._local().host.agent

    @property
    def provider(self) -> SwitchableProvider:
        return self._local().host.provider

    @property
    def catalog(self) -> ModelCatalog | None:
        return self._local().host.catalog

    @property
    def file_index(self) -> FileIndex:
        return self._local().file_index

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
        status.model = escape_display(self.port.model)
        status.window = self.port.window
        status.workspace = display_path(self.workspace)
        status.mode = self.port.mode
        self.query_one(ChatInput).suggestions = self.query_one(CommandSuggestions)
        self.query_one(ChatInput).paths = self.query_one(PathSuggestions)
        self.query_one(ChatInput).focus()
        self.watch(status, "busy", lambda _: self._refresh_hints(), init=False)
        # A failed check shows `not ready` (a `ProviderState`); the host itself asks again before the next turn.
        self._refresh_hints()
        self._apply_height(self.size.height)
        self._reindex()
        self._stop_on_signals()
        self._loop_thread = threading.get_ident()
        if self.config_guard is not None:  # a reaction on another thread tells its listeners on this loop
            self.config_guard.attach_loop(asyncio.get_running_loop())
        self.port.begin_switch()  # claimed before any worker runs (nothing else can hold it yet): `_start` releases it
        if isinstance(self.port, RemotePort):  # the session is opened (and its stream subscribed) by the worker
            self.run_worker(self._start_remote(self.port), group="start")
            return
        # the host's events, drawn one after the other; a turn sent here takes the pasted images
        self.port.subscribe(self._on_port_event, self._on_ask, self._attachments)
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
        """Ask the provider whether it is ready (the host publishes the answer, shown by `_on_host_event`)."""
        return await self.port.check()

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
            self.port.end_switch()

    # shared mode (`agent --shared`): the session is the daemon's

    async def _start_remote(self, port: RemotePort) -> None:
        """Reach the daemon, open the session the command line asked for, draw it from its snapshot, then follow its
        stream. The switch claimed in `on_mount` is held until then, so no message is sent before the session is."""
        transcript = self.query_one(Transcript)
        try:
            while True:  # unreachable: ask whether to keep waiting (a daemon of another version: quit, naming it)
                try:
                    await port.connect()
                    opened = await self._open_remote_start(port)
                except ProtocolMismatch as exc:
                    self.exit(return_code=1, message=f"agent --shared: {exc}")
                    return
                except (DaemonUnavailable, DaemonError) as exc:
                    opened, reason = None, f"the agent daemon cannot be reached: {exc}"
                else:
                    reason = "the agent daemon cannot be reached"
                if opened is not None:
                    break
                if not await self.push_screen_wait(DaemonGoneScreen(reason)):
                    self.exit(return_code=1, message=f"agent --shared: {reason}")
                    return
            snapshot, title, count = opened
            await self._show_snapshot(snapshot)
            if title:
                await transcript.add_notice("info", f'Resumed "{title}" ({count} messages)')
            if self._wanted_model is not None and snapshot.model != self._wanted_model:
                await transcript.add_notice(
                    "info", f"this session runs {escape_display(snapshot.model)}; /model switches it for every client"
                )
            port.subscribe(self._on_port_event, self._on_ask)
            for ask in port.pending_asks():  # asked while no client was attached: shown now
                self._on_ask(ask)
        except Exception as exc:  # a bug or a bad answer must not take the app down: it says so and stays usable
            with contextlib.suppress(Exception):
                await transcript.add_notice("error", f"agent --shared: could not open the session: {escape(str(exc))[:200]}")
        finally:
            port.end_switch()

    async def _open_remote_start(self, port: RemotePort) -> tuple[wm.Snapshot, str, int] | None:
        """The start session as `--continue`, `--resume` or the picker chose it (a new one otherwise, also when the
        chosen one cannot be opened: the notice says why). `(snapshot, title, message count)`; None: unreachable."""
        transcript = self.query_one(Transcript)
        kind = self.start.kind
        resume: str | None = None
        title, count = "", 0
        if kind == "continue":
            resume = "latest"
        elif kind == "resume":
            resume = self.start.session_id or ""
        elif kind == "pick":
            try:
                infos = await port.list_sessions()
            except (DaemonError, DaemonUnavailable) as exc:
                await transcript.add_notice("warn", f"could not list the sessions: {escape(str(exc))}")
                infos = []
            if not infos:
                await transcript.add_notice("info", "No saved sessions for this folder")
            else:
                chosen = await self.push_screen_wait(SessionPicker(infos))
                info = next((i for i in infos if i.id == chosen), None)
                if info is not None:
                    resume, title, count = info.id, info.title, info.message_count
        try:
            try:
                return await port.open(resume, self._wanted_model), title, count
            except DaemonError as exc:
                if resume is None:
                    raise
                await transcript.add_notice("error", escape(exc.text))
                return await port.open(None, self._wanted_model), "", 0
        except (DaemonError, DaemonUnavailable) as exc:
            await transcript.add_notice("error", f"agent daemon: {escape(str(exc))}")
            return None

    async def _show_snapshot(self, snapshot: wm.Snapshot) -> None:
        """Redraw the transcript, the panels and the status bar from a snapshot of the daemon's session."""
        port = self._remote()
        transcript = self.query_one(Transcript)
        await transcript.clear_all()
        await transcript.replay(port.snapshot_session(snapshot).items)
        self.query_one(TodoPanel).show(port.todos)
        self._show_queue()
        status = self.query_one(StatusBar)
        status.model = escape_display(port.model)
        status.window = port.window
        status.mode = port.mode
        status.pending_mode = False
        status.busy = port.turn_active
        self._refresh_hints()

    def _remote(self) -> RemotePort:
        if not isinstance(self.port, RemotePort):
            raise RuntimeError("only in --shared mode")
        return self.port

    async def _daemon_gone(self, gone: DaemonGone) -> None:
        """The daemon did not come back: quit, or keep waiting (the typed text stays in the box either way)."""
        keep = await self.push_screen_wait(DaemonGoneScreen(gone.reason))
        if keep:
            self._remote().keep_waiting()
        else:
            self.exit()

    async def _refuse_shared(self) -> None:
        await self.query_one(Transcript).add_notice("warn", NOT_SHARED)

    def _paths_refreshed(self) -> None:
        """A `--shared` path lookup came back: show it if the input still asks for it."""
        with contextlib.suppress(NoMatches, ScreenStackError):
            self._update_paths(self.query_one(ChatInput))

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
        # the marker is removed (and the store written) off the event loop
        outcome = await asyncio.to_thread(
            apply_trust,
            "global_trust",
            bool(trusted),
            guard=guard,
            servers=[s.name for s in servers],
            held_any=self._held.any,
        )
        await self._show_trust(outcome)

    async def _ask_permission_trust(self) -> None:
        """Ask once whether to trust the project's allow rules and hooks (one decision for the file's bytes); only a
        yes turns them on."""
        rules = self.permissions.untrusted_allow
        hooks = self.hooks.untrusted()
        if not rules and not hooks:
            return
        self._keys_ignored_until = time.monotonic() + KEY_GUARD  # keys typed for the chat must not answer it
        trusted = await self.push_screen_wait(PermissionTrustScreen(rules, hooks))
        outcome = await asyncio.to_thread(  # writes the trust store: not on the event loop
            apply_trust,
            "permission_trust",
            bool(trusted),
            permissions=self.permissions,
            has_rules=bool(rules),
            has_hooks=bool(hooks),
        )
        await self._show_trust(outcome)

    async def _ask_mcp_trust(self) -> None:
        """Ask once whether to start the project's MCP servers; only a yes starts them."""
        if self.mcp is None:
            return
        servers = self.mcp.untrusted("project")  # the global ones held by the marker have their own prompt
        self._keys_ignored_until = time.monotonic() + KEY_GUARD  # keys typed for the chat must not answer it
        trusted = await self.push_screen_wait(McpTrustScreen(servers))
        outcome = await asyncio.to_thread(
            apply_trust, "mcp_trust", bool(trusted), mcp_trust=self._mcp_trust, servers=[s.name for s in servers]
        )
        await self._show_trust(outcome)

    async def _show_trust(self, outcome: TrustOutcome) -> None:
        """The notices of a trust decision, then the MCP servers it turned on (on the event loop)."""
        transcript = self.query_one(Transcript)
        for level, text in outcome.notices:
            await transcript.add_notice(level, text)
        if self.mcp is not None and outcome.start:
            self.mcp.start(outcome.start)

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
            notice = await asyncio.to_thread(self.port.instructions_notice)
            if notice is not None:
                await self.query_one(Transcript).add_notice(notice.level, notice.text)
        except Exception as exc:  # a failure here must not take the app down
            self.port.forget_instructions_notice()  # not shown: the next turn announces it again
            with contextlib.suppress(Exception):
                await self.query_one(Transcript).add_notice("error", f"Could not read AGENTS.md: {exc}")

    async def _session_started(self, source: Literal["startup", "resume", "new", "switch"]) -> None:
        """Run the SessionStart hooks (the session is drawn and its notices are in); never raises. The caller holds the
        switch claim, so no message is accepted while a hook runs. The hook runs as the host's own task, which Esc
        cancels: the runner then kills the hook's process group."""
        if not self.hooks.has("SessionStart"):
            return  # nothing to run, so nothing for Esc or ctrl+c to interrupt
        events = await self.port.session_started(source)
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
            self.port.end_switch()

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
        if not self.port.begin_switch():  # no message is accepted until the switch ends
            return False
        self._cancel_diff()
        return True

    def _busy_text(self, command: str) -> str:
        return self.port.busy_text(command)

    async def _refuse_switch(self) -> None:
        await self.query_one(Transcript).add_notice("warn", self._busy_text("/sessions"))

    async def _pick_session(self) -> bool:
        """List this folder's sessions in a picker (a worker: it waits for the screen), then switch to the choice.
        True when a session was switched to.

        The caller has claimed the switch (`_begin_switch`, or `on_mount` for the start) and releases it.
        """
        store, transcript = self.store, self.query_one(Transcript)
        if store is None and not self.shared:
            return False
        try:
            infos = await self.port.list_sessions()
        except (DaemonError, DaemonUnavailable) as exc:  # only the daemon's port raises
            await transcript.add_notice("warn", f"could not list the sessions: {escape(str(exc))}")
            return False
        if store is not None:
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
            self.port.end_switch()

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
            if not self.shared and self.catalog is None:
                await transcript.add_notice("error", "No model catalog is available")
                return
            try:
                infos = await self.port.list_models()
            except CatalogError as error:
                await transcript.add_notice("error", f"Could not list the installed models: {error}")
                return
            if not infos:
                await transcript.add_notice("info", "No models installed.")
                return
            port = self.port

            async def look(info: ModelInfo) -> tuple[ModelInfo, tuple[int | None, str | None]]:
                try:
                    async with asyncio.timeout(SHOW_TIMEOUT):
                        length = await port.model_length(info.name)
                    return replace(info, context_length=length), (length, None)
                except CatalogError as error:
                    return info, (None, str(error))
                except TimeoutError:
                    return info, (None, f"Ollama did not answer within {SHOW_TIMEOUT:g} s")

            looked = await asyncio.gather(*(look(info) for info in infos))
            known = {info.name: result for info, (_, result) in zip(infos, looked, strict=True)}
            if (reason := self.port.refusal("/model")) is not None:
                await transcript.add_notice("warn", reason)
                return
            if len(self.screen_stack) > 1:  # another dialog came up while the list loaded
                await transcript.add_notice("warn", "Another dialog is open; run /model again after closing it")
                return
            current = installed_name(self.port.model, [i.name for i in infos])
            chosen = await self.push_screen_wait(ModelPicker([info for info, _ in looked], current))
            if chosen is None:
                return
            if not self._begin_switch():
                await transcript.add_notice("warn", self._busy_text("/model"))
                return
            try:
                await self._switch_model(chosen, known.get(chosen))
            finally:
                self.port.end_switch()
        except Exception as error:  # a bug must not kill the app
            await transcript.add_notice("error", f"Model switch failed: {escape(str(error))}")

    async def _switch_model(self, name: str, looked: tuple[int | None, str | None] | None = None) -> None:
        """Switch through the host (which remembers the model for the project), then show the new model."""
        transcript, status = self.query_one(Transcript), self.query_one(StatusBar)
        result = await self.port.switch_model(name, looked)
        if result.model is not None and result.window is not None:
            status.model = escape_display(result.model)
            status.window = result.window
        for level, text in result.notices:
            await transcript.add_notice(level, text)

    async def _sessions_command(self) -> None:
        """The `/sessions` worker: pick and switch, then release the claim `_begin_switch` took."""
        try:
            if await self._pick_session():
                await self._session_started("switch")
        finally:
            self.port.end_switch()

    async def _show_session(self, info: SessionInfo) -> bool:
        """Load `info`, make it the live session and redraw its transcript. The switching flag is held by the caller.
        False when the session could not be opened (the current one stays)."""
        transcript = self.query_one(Transcript)
        if isinstance(self.port, RemotePort):  # the session as the daemon has it now: turn, queue and asks included
            try:
                snapshot = await self.port.switch_to(info.id)
            except (DaemonError, DaemonUnavailable) as exc:
                await transcript.add_notice("error", escape(exc.text if isinstance(exc, DaemonError) else str(exc)))
                return False
            self._pending.clear()
            self._show_pending()
            await self._close_stale_dialogs()
            await self._show_snapshot(snapshot)
            await transcript.add_notice("info", f'Resumed "{info.title}" ({info.message_count} messages)')
            for ask in self.port.pending_asks():
                self._on_ask(ask)
            return True
        opened = await self.port.open_session(info)
        for level, text in opened.notices:
            await transcript.add_notice(level, text)
        loaded = opened.loaded
        if loaded is None:
            return False
        self.query_one(TodoPanel).show(self.port.todos)
        self._pending.clear()  # pasted images belong to the old session
        self._show_pending()
        status = self.query_one(StatusBar)
        status.mode = self.port.mode
        status.pending_mode = False
        status.used = self.port.estimate()
        status.estimated = True
        await transcript.clear_all()
        await transcript.replay(loaded.items)
        await transcript.add_notice("info", f'Resumed "{info.title}" ({info.message_count} messages)')
        self.port.forget_instructions_notice()  # the transcript was redrawn: show the notice in it
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
        self.port.begin_close()  # and the host sends no queued message
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
        self.port.begin_close()
        if self.config_guard is not None:  # the transcript is going away: what is said now is printed after the app
            self.config_guard.unlisten(self._config_changed)
        if isinstance(self.port, RemotePort):  # the session goes on in the daemon: only this client's stream ends
            with contextlib.suppress(Exception):
                await self.port.close()
        else:
            with contextlib.suppress(Exception):  # a failing check must not keep the cleanup below from running
                self.agent.watch.check(Cause.background())  # first: the shell tab and background processes still run
            self.file_index.cancel()  # the build thread stops at its next check instead of walking on
            with contextlib.suppress(Exception):  # the host's job and asks go with the app, as its workers do
                await self.port.close()
                self.host.close()
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
        self.port.set_mode(self.port.mode.next())
        status = self.query_one(StatusBar)
        status.mode = self.port.mode
        if self.port.turn_active and not self.port.undoing:  # a step already running keeps the mode it started with
            self._finished_since_toggle = False
            status.pending_mode = True
        else:
            status.pending_mode = False

    def action_interrupt_turn(self) -> None:
        self.interrupt_turn()

    async def action_toggle_terminal(self) -> None:
        """Show the pane (starting a shell if it is empty) and focus it; from a terminal go back to the input,
        and from the input hide the pane. Hidden terminals keep running."""
        if self.shared:
            await self._refuse_shared()
            return
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
        if action == "toggle_terminal" and self.shared:  # said, not silently ignored
            return not any(isinstance(s, DIALOGS) for s in stack)
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
        if self.port.turn_active or self.port.session_starting:
            self.interrupt_turn()
        else:
            self.exit()

    def interrupt_turn(self) -> None:
        host = self.port
        if host.undoing:  # an undo is not a turn: there is nothing to cancel, and its guard must stay up
            return
        if self._pending and not host.turn_active and not host.switching and self._input_empty():
            self._pending.clear()  # Esc on an empty input: only the pasted images go
            self._show_pending()
            return
        self.workers.cancel_group(self, "turn")  # a workflow (the app runs those)
        self.workers.cancel_group(self, "model-pick")  # only while the list loads: an open picker takes Esc itself
        # The host's turn or compaction and its SessionStart hooks; a job cancelled before its first line is ended
        # at once (its `finally` never runs).
        self.port.interrupt()

    def _input_empty(self) -> bool:
        with contextlib.suppress(NoMatches):
            return not self.query_one(ChatInput).text
        return False

    def _show_pending(self) -> None:
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            self.query_one(AttachmentPanel).show([label for _, label, _ in self._pending])

    async def on_chat_input_paste_image(self, event: ChatInput.PasteImage) -> None:
        if self.shared:  # images never go to the daemon: the text paste still works
            await self.query_one(Transcript).add_notice("warn", f"pasting images is {NOT_SHARED}")
            self.query_one(ChatInput).action_paste()
            return
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

    async def _attachments(self, turn: int) -> tuple[list[Image], list[str]]:
        """The host's turn `turn` takes the pending pasted images; they come back if the agent never gets the message."""
        self._taken = (turn, list(self._pending))
        return await self._take_pasted()

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
        return self.port.queueing

    def _show_queue(self) -> None:
        with contextlib.suppress(NoMatches):  # gone when the app is shutting down
            self.query_one(QueuePanel).show(self.port.queue_entries)

    def _apply_outcome(self, outcome: TurnOutcome) -> None:
        """Every turn, compaction and workflow ends here: after a `done` end the host sent the queue as one turn (busy
        stays on); any other end puts the message that was not sent and the queue back in the box. The host holds its
        claim until this has run (`outcome_applied`), so a message sent before cannot start a turn the old text then
        lands on."""
        started = False
        if not outcome.continued:
            if (back := outcome.restore_for(self.port.client_id)) is not None:  # only what this client sent
                self._restore_draft(back)
            started = self.port.outcome_applied(outcome.ack)  # a message queued meanwhile is sent now
        if outcome.kind == "workflow":
            with contextlib.suppress(NoMatches):
                self.query_one(StatusBar).workflow = ""
        self._refresh_hints()
        if self._taken is not None and self._taken[0] == outcome.turn:
            taken, self._taken = self._taken[1], None
            if taken and not outcome.delivered:
                self._pending[:0] = taken  # the agent never got the message: its images wait for the next one
                self._show_pending()
        with contextlib.suppress(NoMatches):  # the status bar is gone when the app is shutting down
            self.query_one(StatusBar).compacting = False
        if outcome.continued or started:
            self._show_queue()
            self._cancel_diff()  # the merged turn has started
            return
        self._show_queue()
        with contextlib.suppress(NoMatches):
            self.query_one(StatusBar).busy = self.port.turn_active  # a turn sent meanwhile keeps it on

    # turns

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.text
        if text.strip().startswith("/"):
            await self._command(text)
            return
        await self._submit(text, text)

    async def _submit(self, prompt: str, typed: str, label: str | None = None) -> None:
        """Send, queue or refuse a message; a refused one goes back in the box, with a notice."""
        result = await self._send(prompt, typed, label=label)
        if result.status == "refused":
            self._restore_draft(typed)  # not accepted: the message stays in the box
            await self.query_one(Transcript).add_notice("warn", result.reason)

    async def _refuse_if_busy(self, text: str) -> bool:
        """A message cannot start now: it goes back in the box, with a notice. True when it was refused."""
        reason = self.port.message_refusal()
        if reason is None:
            return False
        self._restore_draft(text)  # not accepted: the message stays in the box
        await self.query_one(Transcript).add_notice("warn", reason)
        return True

    async def _send(self, text: str, draft: str, init: bool = False, label: str | None = None) -> SendResult:
        """`_begin_turn` for either port: the local one takes it at once (no await before it), the daemon's over the
        socket."""
        if isinstance(self.port, LocalPort):
            return self._begin_turn(text, draft, init=init, label=label)
        return self._sent(await self.port.send_message(text, label=label, init=init, draft=draft))

    def _begin_turn(self, text: str, draft: str, init: bool = False, label: str | None = None) -> SendResult:
        """Hand `text` to the host (a turn, or the queue while one runs); `draft` is what the box gets back if it
        never sends."""
        return self._sent(self._local().send(text, label=label, init=init, draft=draft))

    def _sent(self, result: SendResult) -> SendResult:
        if result.status == "started":
            self._refresh_hints()
            self._cancel_diff()
            self.query_one(StatusBar).busy = True
        elif result.status == "queued":
            self._show_queue()
        return result

    async def _command(self, original: str) -> None:
        transcript = self.query_one(Transcript)
        found = _COMMAND.match(original.strip())
        name = f"/{found.group(1)}" if found else "/"
        rest = (found.group(2) or "") if found else ""
        if self.shared and name in SHARED_INFO:  # the daemon's runtime has the rules, hooks, servers and skills
            for level, text in await self._remote().info(name[1:]):
                await transcript.add_notice(level, text)
            return
        match name:
            case "/help":
                await transcript.add_notice("info", self._help())
            case "/mcp":
                await transcript.add_notice("info", info.mcp_summary(self.mcp))
            case "/permissions":
                await transcript.add_notice("info", info.permissions_summary(self.permissions))
            case "/guardrails":
                await transcript.add_notice("info", info.guardrails_summary(self.guard, self.config_guard))
            case "/hooks":
                await transcript.add_notice("info", info.hooks_summary(self.hooks))
            case "/skills":
                await transcript.add_notice("info", info.skills_summary(self.skills))
            case "/agents":
                await transcript.add_notice("info", info.agents_summary(self.agents))
            case "/quit":
                self.exit()
            case "/sessions":
                if not self._begin_switch():
                    await self._refuse_switch()
                    return
                self.run_worker(self._sessions_command, group="sessions")
            case "/new" | "/clear":  # the old session's file stays as it is; the next message starts a new one
                # Refused while a turn runs or a session loads: the history belongs to them until they end. Otherwise
                # the host holds the switch claim until the notice is back in the new transcript.
                refused = await self.port.new_session(name)
                if refused is not None:
                    await transcript.add_notice("warn", refused)
                    return
                handed = False  # the worker that runs the SessionStart hook releases the claim
                try:
                    self.query_one(TodoPanel).show(self.port.todos)
                    self._pending.clear()  # pasted images belong to the old session
                    self._show_pending()
                    self._cancel_diff()
                    await transcript.clear_all()
                    status = self.query_one(StatusBar)
                    status.used = 0
                    status.estimated = False
                    status.mode = self.port.mode
                    status.pending_mode = False
                    self.port.forget_instructions_notice()
                    await self._announce_instructions()
                    self.run_worker(self._finish_new_session(), group="new-session", exit_on_error=False)
                    handed = True
                finally:
                    if not handed:
                        self.port.end_switch()
            case "/init":
                if (reason := self.port.refusal("/init")) is not None:
                    await transcript.add_notice("warn", reason)
                    return
                if self.port.mode is Mode.PLAN:
                    await transcript.add_notice("warn", "/init needs BUILD mode — press Tab to switch")
                    return
                if self.shared:  # the daemon builds the prompt from its own scan
                    result = await self._send("/init", draft="/init", init=True)
                    if result.status == "refused":
                        await transcript.add_notice("warn", result.reason)
                    return
                try:
                    prompt = await asyncio.to_thread(self._init_prompt)
                except Exception as error:
                    await transcript.add_notice("error", f"Could not start /init: {error}")
                    return
                # Keys and messages were handled while the prompt was built: check again, with no await before the start.
                if (reason := self.port.refusal("/init")) is not None:
                    await transcript.add_notice("warn", reason)
                    return
                if self.port.mode is Mode.PLAN:
                    await transcript.add_notice("warn", "/init needs BUILD mode — press Tab to switch")
                    return
                await self._send(prompt, draft="/init", init=True)
            case "/compact":
                # Like a turn: Esc cancels it, and the same events reach the transcript.
                result = await self.port.start_compact()
                if result.status == "refused":
                    await transcript.add_notice("warn", result.reason)
                    return
                self._cancel_diff()
                self.query_one(StatusBar).busy = True
            case "/undo":
                await self._undo_command()
            case "/diff":
                await self._diff_command(rest)
            case "/usage":
                await self._usage_command(rest)
            case "/images" | "/workflow" if self.shared:
                await self._refuse_shared()
            case "/images":
                await self._images_command(rest)
            case "/workflow":
                await self._workflow_command(rest)
            case "/model":
                if not rest.strip():
                    if (reason := self.port.refusal("/model")) is not None:
                        await transcript.add_notice("warn", reason)
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
                if custom is None and self.shared:
                    await transcript.add_notice("warn", f"{name}: unknown, or a custom command (custom commands are {NOT_SHARED})")
                    return
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
                await self._submit(prompt, original, label=f"{name} ({custom.source})")  # queued while a turn runs

    def _suggestion_entries(self) -> list[tuple[str, str, str | None]]:
        entries: list[tuple[str, str, str | None]] = [(name, text, None) for name, text in BUILTINS.items()]
        entries.extend((c.name, c.description or "", c.source) for c in self.commands.sorted())
        return entries

    def _reindex(self) -> None:
        """Build the `@` index in a thread (once at mount, again after each turn). A request that arrives while a
        build runs makes it run once more when it ends, so files created meanwhile are not missed."""
        if self._shutting_down or self.shared:  # the daemon keeps its own index
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
        box.paths.update_for(self.port.suggest_paths(token) if token is not None else [])

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
            if self.port.workflow_active:
                self.interrupt_turn()
            else:
                await transcript.add_notice("info", "no workflow is running")
            return
        resume = _parse_resume(rest.split()[1:]) if sub == "resume" else None
        valid = (sub == "run" and len(words) in (2, 3)) or resume is not None
        if not valid:
            await transcript.add_notice("warn", WORKFLOW_USAGE)
            return
        if self.port.workflow_active:
            await transcript.add_notice("warn", WORKFLOW_RUNNING)
            return
        if (reason := self.port.refusal("/workflow")) is not None:
            await transcript.add_notice("warn", reason)
            return
        if self.port.mode is Mode.PLAN:
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

        self.host.begin_workflow()  # the run holds the host's turn claim: messages are refused meanwhile
        self._cancel_diff()
        self.query_one(StatusBar).busy = True
        self._refresh_hints()
        self.run_worker(functools.partial(self._workflow, action, runner), group="turn", exit_on_error=False)

    async def _workflow(self, action: Callable[[EventSink], Awaitable[RunResult]], runner: WorkflowRunner) -> None:
        """The worker of a run: events reach the widgets in order, the outcome is one last line."""
        self.host.job_entered()
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
            self.host.end_turn()  # its outcome comes back on the bus, after everything the run published
            with contextlib.suppress(Exception):
                await uninterrupted(self.host.bus.drained())

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
        if isinstance(self.port, RemotePort):
            try:
                await self.push_screen(UsageScreen(await self.port.usage()))
            except Exception as error:
                await self.query_one(Transcript).add_notice("warn", f"/usage: {escape(str(error))[:200]}")
            return
        status = self.query_one(StatusBar)
        data = summary(
            self.agent.usage,
            session_id=self.port.session_id,
            model=self.port.model,
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
        if self.port.workflow_active:
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
        if (reason := self.port.refusal("/diff")) is not None:
            await transcript.add_notice("warn", reason)
            return
        self.run_worker(self._diff, group="diff", exclusive=True, exit_on_error=False)

    async def _diff(self) -> None:
        """Build the change set off the UI path, then open the viewer (or say why not)."""
        transcript = self.query_one(Transcript)
        try:
            diff = await self.port.diff_list()
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
        if (reason := self.port.refusal("/diff")) is not None:
            await transcript.add_notice("warn", reason)
            return
        if not self.port.is_current_diff(diff):  # a newer turn, an undo or a reset: this is not the turn to show
            return
        await self.push_screen(DiffScreen(diff, self.port.diff_file))

    async def _undo_command(self) -> None:
        transcript = self.query_one(Transcript)
        if (reason := self.port.refusal("/undo")) is not None:
            await transcript.add_notice("warn", reason)
            return
        status = self.query_one(StatusBar)
        self._cancel_diff()
        status.busy = True
        try:
            result = await self.port.undo()  # the host holds the turn claim (and the undo flag) until it is done
        finally:
            status.busy = False
            self._refresh_hints()
        prompt, events = result.prompt, result.events
        if prompt is not None:
            await transcript.remove_last_turn()
            self.query_one(TodoPanel).show(self.port.todos)
            self._restore_draft(prompt)
            status.used = self.port.estimate()
            status.estimated = True
        for event in events:
            if isinstance(event, Notice):
                await transcript.add_notice(event.level, event.text)

    def _init_prompt(self) -> str:
        """The `/init` message: the file's state and the parent files already loaded (reads the disk)."""
        return self.host.init_prompt()

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
            await uninterrupted(self._finish_turn(events, consumer))
        return ran

    async def _on_port_event(self, event: object) -> None:
        """One event of the host's bus (they come one at a time, in order); a bad one is a display error."""
        if self._shutting_down:
            return  # shutting down: the widgets are going away
        try:
            await self._on_host_event(event)
        except Exception as exc:  # one bad event must not drop the ones behind it
            with contextlib.suppress(Exception):
                await self.query_one(Transcript).add_notice("error", f"Display error: {exc}")

    async def _on_host_event(self, event: object) -> None:
        match event:
            case ProviderState(problem):
                status = self.query_one(StatusBar)
                status.error = None if problem is None else "not ready"
                if problem is not None:
                    await self.query_one(Transcript).add_notice("error", problem)
            case UserEntry(text, label):
                self.query_one(StatusBar).busy = True  # also a turn another client of the session sent
                self._refresh_hints()
                transcript = self.query_one(Transcript)
                try:
                    await (transcript.add_user(text, label=label) if label else transcript.add_user(text))
                except asyncio.CancelledError:
                    if (task := asyncio.current_task()) is not None and task.cancelling():
                        raise  # the app is closing
                    self.port.interrupt()  # the entry was not drawn: the turn stops before the agent gets the message
            case Delivering():
                self._finished_since_toggle = False
                self.query_one(StatusBar).pending_mode = False
            case JobEventsDone():
                transcript = self.query_one(Transcript)
                await transcript.end_assistant()
                await transcript.close_open_tools("ended")
            case EntryEnd(discard):
                transcript = self.query_one(Transcript)
                with contextlib.suppress(Exception):
                    if discard:  # the agent never got the message: no turn, so no entry (the text is kept)
                        await uninterrupted(transcript.remove_turn(transcript.latest_turn))
                    else:
                        transcript.end_turn(transcript.latest_turn)  # what follows is not part of the turn
            case TurnOutcome() as outcome:
                self._apply_outcome(outcome)
            case QueueChanged():
                self._show_queue()
                self._refresh_hints()
            case ModeChanged(mode):  # also from another client of the session
                self.query_one(StatusBar).mode = mode
            case ModelChanged(model, window):  # in a daemon the provider is every session's
                status = self.query_one(StatusBar)
                status.model = escape_display(model)
                status.window = window
            case DaemonGone() as gone:
                self.run_worker(self._daemon_gone(gone), group="daemon-gone", exclusive=True, exit_on_error=False)
            case TurnUndone(client) if client != self.port.client_id:  # its own undo already redrew this client
                transcript = self.query_one(Transcript)
                await transcript.remove_last_turn()
                self.query_one(StatusBar).used = self.port.estimate()
                await transcript.add_notice("info", "the last turn was undone in another client")
            case Reconnected(snapshot):
                if snapshot is not None:  # the gap was too old to replay (or the daemon started over)
                    await self._close_stale_dialogs()
                    await self._show_snapshot(snapshot)
                await self.query_one(Transcript).add_notice("info", "reconnected to the agent daemon")
                for ask in self.port.pending_asks():  # still open: shown again (one already shown is skipped)
                    self._on_ask(ask)
            case AskResolved(ask_id, by, _):
                screen = self._ask_screens.pop(ask_id, None)
                if screen is not None:
                    await self._close_dialog(screen)
                    if by and by != self.port.client_id:  # answered in another client of the same session
                        await self.query_one(Transcript).add_notice("info", "answered in another client")
            case _:
                await self._on_agent_event(event)  # ty: ignore[invalid-argument-type]

    def _on_ask(self, event: AskEvent) -> None:
        """An approval for the user: the dialog opens over whatever is shown, and its choice answers the ask. Esc denies
        and interrupts the turn."""
        request = event.request
        if event.ask_id in self._ask_screens:  # already shown (it was in the snapshot and in the stream)
            return
        if not isinstance(request, ApprovalRequest):
            if self.shared:  # the daemon's launch trust prompts; in-process they are asked by `_ask_trust`
                self._on_trust_ask(event.ask_id, request)
            return
        if self._shutting_down:  # the screens are going away: no dialog, and the ask must not wait for one
            if not self.shared:  # in a daemon another client (or the next one) answers it
                self.port.answer(event.ask_id, Decision(Choice.DENY, "shutting down"))
            return
        screen = ApprovalScreen(request)
        self.guard_keys()  # before the dialog is shown: keys typed from now on are not meant for it
        self._ask_screens[event.ask_id] = screen
        self.push_screen(screen, callback=functools.partial(self._approval_answered, event.ask_id))

    async def _close_stale_dialogs(self) -> None:
        """Close the ask dialogs whose ask the daemon no longer has (answered or withdrawn while this client was away,
        or of the session it left)."""
        pending = {ask.ask_id for ask in self.port.pending_asks()}
        for ask_id in [a for a in self._ask_screens if a not in pending]:
            screen = self._ask_screens.pop(ask_id)
            await self._close_dialog(screen)

    def _on_trust_ask(self, ask_id: str, ask: TrustAsk) -> None:
        """One of the daemon's trust prompts (its texts are built there); only a yes trusts. A stray key is ignored for
        a moment, as for the in-process prompts."""
        if self._shutting_down:
            return  # left open in the daemon: another client (or the next one) answers it
        screen = PermissionTrustScreen(ask=ask)
        self._keys_ignored_until = time.monotonic() + KEY_GUARD
        self._ask_screens[ask_id] = screen
        self.push_screen(screen, callback=functools.partial(self._trust_answered, ask_id))

    def _trust_answered(self, ask_id: str, trusted: bool | None) -> None:
        if self._ask_screens.pop(ask_id, None) is None:
            return  # answered in another client meanwhile
        self._answered(ask_id, self.port.answer(ask_id, bool(trusted)))

    def _approval_answered(self, ask_id: str, result: Decision | None) -> None:
        if self._ask_screens.pop(ask_id, None) is None:
            return  # the ask was withdrawn (its turn was cancelled) and the dialog closed for it: nothing to answer
        if result is None:  # Escape: deny, and stop the turn
            self.interrupt_turn()  # the cancel lands at once, so the answer below only counts if nothing was cancelled
            self._answered(ask_id, self.port.answer(ask_id, Decision(Choice.DENY, "interrupted")))
            return
        self._answered(ask_id, self.port.answer(ask_id, result))

    def _answered(self, ask_id: str, result: AnswerResult) -> None:
        """What the host said to an answer: a refused one is said, and an ask still open gets its dialog again."""
        if result in ("ok", "already_answered"):
            return
        self.call_later(self._report_answer, result)
        if result == "invalid":
            for event in self.port.pending_asks():
                if event.ask_id == ask_id:
                    self._on_ask(event)

    async def _report_answer(self, result: str) -> None:
        with contextlib.suppress(NoMatches):
            await self.query_one(Transcript).add_notice("warn", f"could not answer the approval ({result})")

    async def _close_dialog(self, screen: ApprovalScreen | PermissionTrustScreen) -> None:
        """Take a withdrawn dialog off the stack even when another screen sits above it."""
        try:
            while screen in (stack := self.screen_stack) and len(stack) > 1:
                if self.screen is screen:
                    screen.dismiss(None)
                else:
                    self.pop_screen()
        except Exception as exc:  # report it: a dialog that cannot be closed must not fail silently
            with contextlib.suppress(Exception):
                await self.query_one(Transcript).add_notice("error", f"Could not close the approval dialog: {exc}")
        if screen in self.screen_stack:
            with contextlib.suppress(Exception):
                await self.query_one(Transcript).add_notice("error", "The approval dialog is still open")

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
