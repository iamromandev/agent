"""The agent loop: model request -> tool calls -> model request, until the reply has no tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import platform
import stat
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path, PurePath
from typing import Any, Literal

from core.engine.agents.loader import Agents
from core.engine.approval import ApprovalRequest, Approver, Choice, Grants
from core.engine.checkpoints.birth import settle
from core.engine.checkpoints.restore import restore_checkpoint, restore_unsnapshotted
from core.engine.checkpoints.store import CheckpointStore, DiffEntry, Snapshot, exclude_signature, file_signature
from core.engine.compaction import (
    MIN_SUMMARY_GAIN,
    STOP_NOTE_PREFIX,
    SUMMARY_PREFIX,
    Reported,
    cap_summary,
    cut_index,
    estimate_tokens,
    is_summary,
    message_tokens,
    prune,
    summary_input,
)
from core.engine.configwatch import Cause, NullWatch, Watch
from core.engine.events import (
    AgentEvent,
    AssistantText,
    Compacting,
    EventSink,
    Notice,
    Phase,
    PhaseKind,
    TaskProgress,
    TodosChanged,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)
from core.engine.format.runner import FormatHub
from core.engine.guardrails.paths import strip_firmlink
from core.engine.guardrails.redact import drop_partial_tail
from core.engine.guardrails.setup import REDACTION_FAILED, guard_of
from core.engine.hooks.setup import DEFAULT_REASON, STOP_CONTINUATIONS, HookDecision, Hooks, hook_failed, notice_text
from core.engine.image_inputs import image_dimensions
from core.engine.instructions import Instructions, load_instructions
from core.engine.ledger import Foreign, LedgerEntry, TurnLedger
from core.engine.lsp.hub import LspHub
from core.engine.mode import Mode
from core.engine.permissions.config import agent_config_files
from core.engine.permissions.rules import RuleError, exact_rule_text, resolve_key
from core.engine.permissions.setup import Rules, RuleSavedNotTrusted
from core.engine.prompt import build_system_prompt, build_task_prompt
from core.engine.provider.base import (
    Done,
    Image,
    Message,
    Provider,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from core.engine.provider.vision import VisionCheck
from core.engine.session import NullRecorder, Recorder, ToolMeta
from core.engine.session.records import compaction_notice
from core.engine.task import TaskTool
from core.engine.text import escape
from core.engine.todo import INTERRUPTED, NOT_COMPLETED, SKIPPED, TodoItem, active, current_todos, render_note
from core.engine.tools.base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from core.engine.tools.paths import InvalidPath, resolve_path
from core.engine.tools.skill import SkillTool
from core.engine.tools.text import TOOL_OUTPUT_LIMIT, keep_sgr, truncate, truncate_ansi
from core.engine.turn_diff import DiffItem, FileDiff, TurnDiff, build_diff, load_file
from core.engine.undo import (
    NOTHING_TO_UNDO,
    SNAPSHOT_FILE_LIMIT,
    SNAPSHOT_TOTAL_LIMIT,
    UNDO_UNTRACKED,
    UNSURE_OTHER,
    UndoLog,
    UndoResult,
    UndoTurn,
    checkpoint_notices,
    read_nofollow,
    restore_turn,
    undo_notices,
)
from core.engine.usage import UsageRecord, UsageTotals

# Marks a partial assistant message that a cancel cut short, so the model does not take it for a
# complete answer on the next turn.
# What redaction reads of a result: the output limit plus this much, so a secret that starts before the cut and ends a
# little after it is still seen whole; one that has no end in sight is replaced to the end of what is read. The rest of
# the output is dropped before redaction, because redaction must not cost more than the model could ever see.
REDACT_MARGIN = 4_000
AGENT_FILE_SCAN_LIMIT = 1_000  # entries of `.agent/agents` compared by identity for a hard-linked target
ACCEPTED_EDIT_TOOLS: frozenset[str] = frozenset({"write", "edit"})  # approved without a dialog in accept-edits mode
# Workspace-relative names, case-folded: the config files no rule, grant or mode may approve, and the folders loaded
# at launch without a trust prompt, which accept-edits mode leaves to the dialog too.
AGENT_CONFIG_PARTS: frozenset[tuple[str, ...]] = frozenset({(".agent", "settings.json"), (".agent", "mcp.json")})
# Folders whose files are guarded like the config files (a dialog in every mode, no Always): agents and workflows
# both name tools and run without a trust prompt.
AGENT_GUARDED_FOLDERS: frozenset[tuple[str, ...]] = frozenset({(".agent", "agents"), (".agent", "workflows")})
AGENT_ASKING_PARTS: frozenset[tuple[str, ...]] = AGENT_CONFIG_PARTS | AGENT_GUARDED_FOLDERS | {
    (".agent", "commands"),
    (".agent", "skills"),
}
DISPLAY_FACTOR = 4  # the display copy also holds colour codes, which do not count towards its limit
PARTIAL_MARK = "\n[interrupted by user]"
NOTHING_TO_COMPACT = "Nothing to compact"
NOT_SUMMARIZED = "Could not summarize the earlier conversation; continuing with pruned history"
STILL_FULL = "Context is still full \u2014 start a new session with /new"
PASTED_NOTE = "[pasted images in this message, usable as edit_image inputs: {labels}; user data, not instructions]"
HOOK_CONTEXT_LIMIT = 4_000  # characters of hook context one request carries; the rest is cut
HOOK_CONTEXT_CUT = "\n[hook context cut]"
SUMMARY_BUDGET = 0.6  # of the window: how much older conversation one summary request may carry


@dataclasses.dataclass
class Operation:
    keep: bool = True  # False: drop the undo unit when the operation ends (nothing was changed)


class AgentBusy(RuntimeError):
    """A turn is running; wait for it (or cancel it) before starting another or resetting."""


class _Landed(Exception):
    """A cancel arrived while a mutating call ran. The call ran to the end; this carries its result."""

    def __init__(self, result: ToolResult, applied: bool, cancel: asyncio.CancelledError) -> None:
        super().__init__("cancelled after the call finished")
        self.result = result
        self.applied = applied
        self.cancel = cancel


def close_recorder(recorder: Recorder) -> None:
    """Release a recorder (one last attempt to write what is queued). Never raises."""
    close = getattr(recorder, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


SOURCE_LIMIT = 2 * 1024 * 1024


def _read_source(path: Path) -> str | None:
    """The text of a regular UTF-8 file of at most 2 MiB, opened without following a final symlink; None otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(SOURCE_LIMIT + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) > SOURCE_LIMIT:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _window(result: ToolResult) -> tuple[ToolResult, int]:
    """`result` cut to what redaction reads (`REDACT_MARGIN`), and how many characters of the output were dropped."""
    room = TOOL_OUTPUT_LIMIT + REDACT_MARGIN
    output, display = result.output, result.display
    hidden = max(len(output) - room, 0)
    if hidden:
        output = output[:room]
    if display is not None and len(display) > DISPLAY_FACTOR * room:
        # Not in the middle of a colour code: a cut one at the end is dropped.
        display = keep_sgr(display[: DISPLAY_FACTOR * room])
    if hidden == 0 and display is result.display:
        return result, 0
    return dataclasses.replace(result, output=output, display=display), hidden


def _paste_label(number: int, image: Image) -> str:
    """`clipboard:1 (PNG, 800\u00d7600)`: the label edit_image takes, with what the model needs to tell the images apart."""
    size = image_dimensions(image.format, image.data)
    shown = f"{image.format.upper()}, {size[0]}\u00d7{size[1]}" if size is not None else image.format.upper()
    return f"clipboard:{number} ({shown})"


def _truncate(output: str, hidden: int) -> str:
    """`truncate`, for a redacted output that `_window` cut (`hidden` characters were dropped before redaction).

    The last word goes first: a token that straddles the window edge is left as a prefix no pattern recognises, and
    when redaction has shrunk the text below the limit that prefix would be the end of what the model gets. The note
    counts every character that is not in the output: the window's, that word and what the limit cuts."""
    if hidden == 0:
        return truncate(output)
    output, dropped = drop_partial_tail(output)
    cut = max(len(output) - TOOL_OUTPUT_LIMIT, 0)
    return f"{output[:TOOL_OUTPUT_LIMIT]}\n[{hidden + dropped + cut} characters truncated]"


def _relative_parts(path: Path, folder: Path) -> tuple[str, ...] | None:
    """`path`'s parts below `folder`, case-folded, comparing case-insensitively (APFS); None when it is not inside.
    A leading macOS `/System/Volumes/Data` is dropped from both first: it is another spelling of the same place."""
    parts = [p.casefold() for p in PurePath(strip_firmlink(str(path))).parts]
    base = [p.casefold() for p in PurePath(strip_firmlink(str(folder))).parts]
    return tuple(parts[len(base) :]) if parts[: len(base)] == base else None


class Agent:
    def __init__(
        self,
        provider: Provider,
        tools: Sequence[Tool],
        approver: Approver,
        ctx: ToolContext,
        max_steps: int,
        today: Callable[[], date] = date.today,
        recorder: Recorder | None = None,
        compact_at: float = 0.8,
        compact_keep_turns: int = 2,
        instructions_share: float = 0.25,
        undo_file_limit: int = SNAPSHOT_FILE_LIMIT,
        undo_total_limit: int = SNAPSHOT_TOTAL_LIMIT,
        extra_tools: Callable[[], Sequence[Tool]] | None = None,
        task_tools: Sequence[Tool] | None = None,
        task_max_steps: int = 15,
        role: Literal["main", "task"] = "main",
        rules: Rules | None = None,
        hooks: Hooks | None = None,
        checkpoints: CheckpointStore | None = None,
        agents: Agents | None = None,
        subagent: str = "",
        role_prompt: str | None = None,
        undo: UndoLog | None = None,
        mode_source: Callable[[], Mode] | None = None,
        vision: VisionCheck | None = None,
        preapproved: Callable[[ToolCall], bool] | None = None,
        watch: Watch | None = None,
        lsp: LspHub | None = None,
        format: FormatHub | None = None,
        ledger: TurnLedger | None = None,
        owner: str = "",
    ) -> None:
        self.provider = provider
        # Sessions of one daemon run turns at once: the ledger tells `/undo` and `/diff` which paths another session's
        # overlapping turn changed. Plain `agent`, `-p` and every child pass none.
        self.ledger = ledger if role == "main" and not subagent and undo is None else None
        self.owner = owner
        self._ledger_entry: LedgerEntry | None = None
        self.lsp = lsp if role == "main" and not subagent else None  # diagnostics after write/edit: the main agent only
        self.format = format if role == "main" and not subagent else None  # formatters after write/edit: the main agent only
        # The config watch: checked around every tool call and at the end of a turn; a child shares its parent's.
        self.watch: Watch = watch if watch is not None else NullWatch()
        # A child that runs the calls of an approved workflow plan: the approval dialog is skipped for a call this
        # says yes to (never for a protected-path ask or a guarded settings write).
        self.preapproved = preapproved
        self.vision = vision  # whether the model has vision; None: never (the callers ask it before they send images)
        self.subagent = subagent  # the custom agent this child runs, "" otherwise
        self.role_prompt = role_prompt  # a custom child's own system prompt, in place of the research role
        self._checkpoints = checkpoints  # the `task` child never gets one
        self._checkpoint_warning: str | None = None  # taken from the store, not yet shown
        self._large_before: dict[str, tuple[int, int] | None] = {}  # files over the size cap when the turn began
        self._exclude_before: tuple[int, int, int] | None = None  # `.git/info/exclude` when the turn began
        self._checkpoint_read: asyncio.Task[Any] | None = None  # the `/undo` ledger look or diff + blob read, while it runs
        self.hooks = hooks if hooks is not None else Hooks.empty()  # the `task` child never gets any
        self.rules = rules if rules is not None else Rules.empty()  # not part of `Grants`: `reset()` leaves it alone
        self.role = role
        self.tools = list(tools)
        if task_tools is not None and role == "main":
            self.tools.append(TaskTool(self, list(task_tools), task_max_steps, agents))
        self._extra_tools = extra_tools  # asked at every use: tools that come and go while the agent runs (MCP)
        self._granted_extra: dict[str, Tool] = {}  # "Always" given to an extra tool, to revoke it when that tool is gone
        self.approver = approver
        self._shared_undo = undo is not None  # a custom child writes into its parent's turn: it opens none
        self._undo = undo if undo is not None else UndoLog(ctx.workspace, undo_file_limit, undo_total_limit)
        self.ctx = dataclasses.replace(ctx, journal=self._undo)
        self.max_steps = max_steps
        self._step = 0  # the model request number of the running turn (shown in `Phase` events)
        if not 0 < compact_at <= 1:
            raise ValueError(f"compact_at must be above 0 and at most 1, got {compact_at}")
        if compact_keep_turns < 1:
            raise ValueError(f"compact_keep_turns must be at least 1, got {compact_keep_turns}")
        if not 0 < instructions_share <= 0.5:
            raise ValueError(f"instructions_share must be above 0 and at most 0.5, got {instructions_share}")
        self.instructions_share = instructions_share
        self._announced: tuple[tuple[str, str], ...] = ()  # the files and states of the last Instructions notice
        self._loaded: Instructions | None = None  # what the last request (or notice) read, for `estimate`
        self.compact_at = compact_at
        self.compact_keep_turns = compact_keep_turns
        self.today = today
        self._mode_source = mode_source  # a custom child reads its parent's mode, at every step
        self._mode = Mode.BUILD
        self.history: list[Message] = []  # no system message; the prompt is rebuilt per request
        self.usage = UsageTotals()  # this session's recorded model calls (never lowered by `/undo`)
        self.usage_unrecorded = False  # the session has messages from before its first usage record
        self.todo_base: tuple[TodoItem, ...] = ()  # the list before the history's first todo call (kept by compaction)
        self.grants = Grants()
        self.tool_meta: dict[str, ToolMeta] = {}  # by tool_call_id: how each tool message is shown and pruned
        self._recorder: Recorder = recorder if recorder is not None else NullRecorder()
        self._todos_cache: tuple[tuple[int, Message | None, tuple[TodoItem, ...]], tuple[TodoItem, ...]] | None = None
        self._reported: Reported | None = None  # the model's own count after its last reply
        self._save_error: str | None = None  # a write failure not yet shown to the UI
        self._save_failing = False  # inside a failure streak: no new notice until a write succeeds
        self._busy = False
        self._inflight: str | None = None  # id of the tool call being run, for the interrupt repair
        self._summary_failed = False  # a summary failed in this turn: automatic compaction does not retry it
        self._warned_full = False  # "still full" was shown in this turn
        self._hook_notes: list[str] = []  # redacted context from hooks, for the next request only
        self._pasted: tuple[Image, ...] = ()  # the images the turn's message carried: edit_image's `clipboard:N`

    @property
    def mode(self) -> Mode:
        return self._mode if self._mode_source is None else self._mode_source()

    @mode.setter
    def mode(self, mode: Mode) -> None:
        self._mode = mode

    def set_mode(self, mode: Mode) -> None:
        """Takes effect from the next model request; a step already running keeps its snapshot."""
        self.mode = mode

    @property
    def recorder(self) -> Recorder:
        """The recorder of the live session (it changes with `reset`)."""
        return self._recorder

    @property
    def busy(self) -> bool:
        return self._busy

    def reset(
        self,
        *,
        recorder: Recorder,
        history: Sequence[Message] = (),
        tool_meta: Mapping[str, ToolMeta] | None = None,
        todos: Sequence[TodoItem] = (),
        usage: UsageTotals | None = None,
        unrecorded: bool = False,
    ) -> Recorder | None:
        """Start over with `history` (a new session, or a resumed one): BUILD mode, no grants.

        Returns the old recorder, not closed (closing can write to disk: the caller does it off its
        UI thread with `close_recorder`), or None when it is the one passed in. `recorder` is
        required: pass `NullRecorder()` to save nothing.
        """
        if self._busy:
            raise AgentBusy("cannot reset while a turn is running")
        old = self._recorder if recorder is not self._recorder else None
        self._recorder = recorder
        self.history = list(history)
        self.todo_base = tuple(todos)
        self.usage = usage if usage is not None else UsageTotals()
        self.usage_unrecorded = unrecorded
        self.tool_meta = dict(tool_meta or {})
        self.mode = Mode.BUILD
        self.grants = Grants()
        self._granted_extra.clear()
        self._reported = None
        self._save_error = None
        self._save_failing = False
        self._summary_failed = self._warned_full = False
        self._hook_notes.clear()  # context for the old session: the new one's SessionStart hook adds its own
        self._undo.clear()
        self._forget_ledger()
        return old

    @property
    def undo_log(self) -> UndoLog:
        """The log this agent's file tools write into; a custom subagent is given its parent's."""
        return self._undo

    @property
    def todos(self) -> tuple[TodoItem, ...]:
        """The todo list as the history leaves it (see `current_todos`)."""
        history = self.history
        last = history[-1] if history else None
        key = (len(history), last, self.todo_base)
        cached = self._todos_cache
        if cached is not None and cached[0][0] == key[0] and cached[0][1] is last and cached[0][2] == key[2]:
            return cached[1]
        found = current_todos(history, self.todo_base)
        self._todos_cache = (key, found)
        return found

    def _todo_note(self) -> Message | None:
        """The reminder of the list for a request: the main agent only, and only while something is left to do."""
        todos = self.todos
        if self.role != "main" or not active(todos):
            return None
        return Message("system", render_note(todos))

    def _hook_note(self) -> Message | None:
        """The context hooks added for the next request (already redacted), as one system message."""
        if not self._hook_notes:
            return None
        text = "\n\n".join(self._hook_notes)
        if len(text) > HOOK_CONTEXT_LIMIT:
            text = text[:HOOK_CONTEXT_LIMIT] + HOOK_CONTEXT_CUT
        return Message("system", "[hook context]\n" + text)

    def close(self) -> None:
        """Release the recorder (one last attempt to write what is queued). Never raises."""
        close_recorder(self._recorder)

    def estimate(self) -> int:
        """Tokens the next request would use: the model's last count plus a guess for what came after."""
        specs = [t.spec() for t in self.tools_for(self.mode)]
        loaded = self._loaded if self._loaded is not None else self.instructions()
        total = estimate_tokens(self._system(self.mode, loaded), self.history, self._reported, specs)
        hook = self._hook_note()  # not sent yet, so never in the model's count
        if hook is not None:
            total += message_tokens(hook)
        if self._reported is not None:
            return total  # the model's count already includes the note it was sent
        note = self._todo_note()
        return total + message_tokens(note) if note is not None else total

    def instructions(self) -> Instructions:
        """The AGENTS.md files as they are on disk now, fitted to this model's window; remembered for `estimate`."""
        budget = int(self.instructions_share * self.provider.context_window)
        self._loaded = loaded = load_instructions(self.ctx.workspace, budget, guard_of(self.ctx).paths)
        return loaded

    def forget_instructions_notice(self) -> None:
        """The last notice was not delivered (or the transcript was cleared): show the next one even if unchanged."""
        self._announced = ()

    def instructions_notice(self) -> Notice | None:
        """The notice for the instructions when they differ from the last one announced; remembers what it saw."""
        loaded = self.instructions()
        key, self._announced = self._announced, loaded.key
        shown = loaded.notice()
        if shown is None or loaded.key == key:
            return None
        return Notice(*shown)

    def _system(self, mode: Mode, instructions: Instructions | None = None) -> Message:
        """The system message; reads the files fresh (R4) unless the caller passes what it has."""
        loaded = instructions if instructions is not None else self.instructions()
        skill_tool = next((t for t in self.tools if isinstance(t, SkillTool)), None)
        skills = skill_tool.skills.section() if skill_tool is not None else None
        if self.role == "task":
            prompt = build_task_prompt(
                self.ctx.workspace,
                self.today(),
                platform.system(),
                loaded,
                skills=skills,
                paths=guard_of(self.ctx).paths,
                role=self.role_prompt,
            )
            return Message("system", prompt)
        todo = any(t.name == "todo" for t in self.tools)
        task_tool = next((t for t in self.tools if isinstance(t, TaskTool)), None)
        prompt = build_system_prompt(
            self.ctx.workspace,
            mode,
            self.today(),
            platform.system(),
            instructions=loaded,
            todo=todo,
            skills=skills,
            agents=task_tool.agents.section() if task_tool is not None else None,
            paths=guard_of(self.ctx).paths,
        )
        return Message("system", prompt)

    def _record(self, message: Message, meta: ToolMeta | None = None) -> None:
        """Add to the history and the session record. Synchronous (a cancel can't split the two); never raises."""
        self.history.append(message)
        if meta is not None and message.tool_call_id:
            self.tool_meta[message.tool_call_id] = meta
        self._save(lambda: self._recorder.message(message, meta))

    def _record_compaction(self, pruned: int, summarized: int, before: int, after: int) -> None:
        """Save the history as compaction left it (the file keeps every earlier record). Never raises."""
        self._save(lambda: self._recorder.compaction(self.history, self.tool_meta, pruned, summarized, before, after))

    def model_switched(self, name: str, window: int, emit: EventSink | None = None) -> None:
        """Record that the provider now uses `name` with `window` (the provider itself was switched by the caller).
        The history is untouched. Never raises; a failed save is shown through `emit` when there is one."""
        self._save(lambda: self._recorder.model(name, window))
        if emit is not None:
            self._flush_save_error(emit)

    def _save(self, write: Callable[[], None]) -> None:
        try:
            write()
        except Exception as exc:  # SessionWriteError, or a recorder bug: either way the turn goes on
            if not self._save_failing:  # one notice per streak, not one per message
                self._save_failing = True
                self._save_error = f"Could not save the session: {exc}"
        else:
            self._save_failing = False

    def _record_usage(
        self, prompt_tokens: int, completion_tokens: int, started: float, *, compaction: bool, workflow: bool = False
    ) -> None:
        """One finished model call of the main agent: add it to the totals, then save it (a failed save is a notice)."""
        if self.role != "main":
            return
        record = UsageRecord(
            model=getattr(self.provider, "model", "") or "",
            input_tokens=max(prompt_tokens, 0),
            output_tokens=max(completion_tokens, 0),
            duration_ms=max(int((time.monotonic() - started) * 1000), 0),
            compaction=compaction,
            at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            workflow=workflow,
        )
        self.usage = self.usage.add(record)
        self._save(lambda: self._recorder.usage(record))

    def _flush_save_error(self, emit: EventSink) -> None:
        if self._save_error is not None:
            emit(Notice("error", self._save_error))
            self._save_error = None  # only once shown: a sink that raised gets it again at the next flush point

    def _clear_undo(self) -> None:
        """Compaction drops the undo history of this agent; a shared log belongs to the parent's turns."""
        if not self._shared_undo:
            self._undo.clear()
            self._forget_ledger()

    def _forget_ledger(self) -> None:
        """This agent's undo history is gone: its turns can no longer be undone, but what they changed is still in the
        workspace, so the ledger keeps them (retired) for the other sessions' undo. The turn running now keeps its entry
        here: `finish` still has to set its end."""
        if self.ledger is not None:
            self.ledger.retire(self.owner)

    def _begin_undo(self, prompt: str) -> None:
        self._undo.begin(len(self.history), prompt)
        turn = self._undo.current
        if self.ledger is not None and turn is not None:
            self._ledger_entry = self.ledger.begin(self.owner, turn)

    def _finish_undo(self, *, keep: bool = True) -> None:
        entry, self._ledger_entry = self._ledger_entry, None
        if keep:
            self._undo.finish()
            if entry is not None and self.ledger is not None:
                self.ledger.finish(entry)
        else:
            self._undo.discard()
            if entry is not None and self.ledger is not None:
                self.ledger.drop(entry)

    async def compact(self, emit: EventSink) -> None:
        """Compact now, whatever the estimate (R16): prune, then summarize what is older than the kept turns.

        Esc cancels it like a turn: the history stays valid, and the cancel goes on after a notice.
        """
        if self._busy:
            raise AgentBusy("a turn is already running")
        if not self.history:
            self._clear_undo()
            emit(Notice("info", NOTHING_TO_COMPACT))
            return
        self._busy = True
        self._summary_failed = self._warned_full = False
        try:
            await self._compact(emit, force=True)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):  # the sink must not replace the cancel
                self._flush_save_error(emit)
                emit(Notice("warn", "Compaction interrupted"))
            raise
        finally:
            self._busy = False

    async def _maybe_compact(self, emit: EventSink) -> None:
        """Before each model request: compact when the estimate has reached `compact_at` of the window."""
        await self._compact(emit, force=False)

    async def _compact(self, emit: EventSink, force: bool) -> None:
        """Prune old tool results, then summarize what is older than the kept turns if it is still too big.

        Only called where no tool call is pending (before a request, or between turns), so the history
        can be swapped. Whatever changed is saved as one compaction record first, then shown as notices:
        a sink that raises cannot keep the record from being written. A summary that fails keeps the
        pruned history and is not tried again in the same turn; a cancel during it saves the pruning,
        shows its notice, then goes on.
        """
        window = self.provider.context_window
        limit = self.compact_at * window
        before = self.estimate()
        if not force and before < limit:
            return
        pruned_history, pruned = prune(self.history, self.tool_meta, self.compact_keep_turns)
        if pruned:
            self.history[:] = pruned_history  # new Message objects: the recorder's earlier records stay as they were
            self._reported = None  # the model's count was for the unpruned history
        mid = self.estimate() if pruned else before
        cut = cut_index(self.history, self.compact_keep_turns)
        older = cut - (1 if is_summary(self.history, 0) else 0)  # the real messages before the cut
        summarized = 0
        notices: list[AgentEvent] = []
        if older > 0 and (force or self._worth_summarizing(mid, limit, cut, window)):
            with contextlib.suppress(Exception):  # a status signal: a sink that raises must not stop the summary
                emit(Compacting(True))
            try:
                text = await self._summarize(self.history[:cut], window)
            except asyncio.CancelledError:
                if pruned:  # the pruning already happened: save it, so the file matches the history
                    self._record_compaction(pruned, 0, before, mid)
                    with contextlib.suppress(Exception):  # the sink must not replace the cancel
                        emit(Notice("info", compaction_notice(pruned, 0, before, mid)))
                        emit(UsageUpdate(mid, window, estimated=True))
                raise
            finally:
                with contextlib.suppress(Exception):  # nor replace a cancel
                    emit(Compacting(False))
            if text:
                self.todo_base = self.todos  # the summarized calls leave the history: keep their list (R12)
                self.history[:] = [Message("user", SUMMARY_PREFIX + text), *self.history[cut:]]
                self._reported = None  # the summary request's own Done is never kept as the count
                summarized = older
            else:
                self._summary_failed = True
                notices.append(Notice("warn", NOT_SUMMARIZED))
        after = self.estimate()
        if force or pruned or summarized:
            self._clear_undo()  # the turns before it are folded or pruned: undoing them would not match the history
        if pruned or summarized:
            self._record_compaction(pruned, summarized, before, after)
            self._flush_save_error(emit)
        if pruned or summarized or force:
            notices.append(Notice("info", compaction_notice(pruned, summarized, before, after)))
            notices.append(UsageUpdate(after, window, estimated=True))
        if after >= window and not self._warned_full:
            self._warned_full = True
            notices.append(Notice("warn", STILL_FULL))
        for notice in notices:
            emit(notice)

    def _worth_summarizing(self, estimate: int, limit: float, cut: int, window: int) -> bool:
        """For an automatic compaction: still over the limit, no failed summary yet this turn, and enough to gain."""
        if estimate < limit or self._summary_failed:
            return False
        return sum(message_tokens(m) for m in self.history[:cut]) >= MIN_SUMMARY_GAIN * window

    async def _summarize(self, older: Sequence[Message], window: int) -> str | None:
        """One model request, no tools, that summarizes `older`. None when it fails or says nothing.

        It is not a step of the turn; its Done is only recorded as usage (main agent).
        """
        request, _ = summary_input(older, int(window * SUMMARY_BUDGET))
        parts: list[str] = []
        started = time.monotonic()
        try:
            async with contextlib.aclosing(self.provider.stream(request, [])) as events:
                async for event in events:
                    match event:
                        case TextDelta(text):
                            parts.append(text)
                        case Done(prompt_tokens, completion_tokens):
                            self._record_usage(prompt_tokens, completion_tokens, started, compaction=True)
                        case ProviderError():
                            return None
        except Exception:  # a provider breaking its contract: the pruned history is still fine to go on with
            return None
        text = "".join(parts).strip()
        return cap_summary(text, window) if text else None

    def all_tools(self) -> list[Tool]:
        """The fixed tools plus whatever the extra source offers right now."""
        extra = list(self._extra_tools()) if self._extra_tools is not None else []
        self._revoke_gone(extra)
        return [*self.tools, *extra]

    def _revoke_gone(self, extra: Sequence[Tool]) -> None:
        """Drop the "Always" of an extra tool that is no longer offered, even if another tool now has its name
        (two MCP servers whose names clean to the same text): the user approved that tool, not the name."""
        if not self._granted_extra:
            return
        current = {id(t) for t in extra}
        for name, tool in list(self._granted_extra.items()):
            if id(tool) not in current:
                del self._granted_extra[name]
                self.grants.revoke(name)

    def tools_for(self, mode: Mode) -> list[Tool]:
        return [t for t in self.all_tools() if not (mode is Mode.PLAN and self._plan_blocked(t))]

    def _plan_blocked(self, tool: Tool) -> bool:
        """Hidden and refused in plan mode: mutating tools, and `bash` for a custom subagent (R13: it is offered no
        write tools there; the main agent's own `bash` stays, asking every time)."""
        return tool.mutating or (bool(self.subagent) and tool.name == "bash")

    async def run_turn(
        self,
        text: str,
        emit: EventSink,
        *,
        draft: str | None = None,
        context: Sequence[str] = (),
        images: Sequence[Image] = (),
        pasted: Sequence[Image] = (),
    ) -> None:
        """Run one user turn. `context` is what the `UserPromptSubmit` hooks added: a system note for the first request.
        `images` go with the user message as they are: the caller checked the model's vision and the limits.
        `pasted` are the images the user pasted into this message, whether or not the model can see them: when an
        `edit_image` tool is offered, the model is told their labels and the tool can use them for this turn."""
        if self._busy:
            raise AgentBusy("a turn is already running")
        if not text.strip():
            return
        self._busy = True
        self._summary_failed = self._warned_full = False  # per turn: one retry of a summary, one warning
        if not self._shared_undo:
            self._begin_undo(draft if draft is not None else text)
        kept = len(self._hook_notes)  # what stands before this turn's own context (SessionStart's)
        try:
            self._pasted = tuple(pasted)
            content = text
            if self._pasted and any(t.name == "edit_image" for t in self.tools):
                labels = ", ".join(_paste_label(n, image) for n, image in enumerate(self._pasted, 1))
                content = f"{text}\n\n{PASTED_NOTE.format(labels=labels)}"
            self._record(Message("user", content, images=tuple(images)))
            try:
                await self._checkpoint(emit, end=False)
                for notice in self._add_hook_context("UserPromptSubmit", context):
                    emit(notice)
                self._flush_save_error(emit)
                if self.role == "main" and (notice := self.instructions_notice()) is not None:
                    try:
                        emit(notice)
                    except BaseException:
                        self.forget_instructions_notice()  # not delivered: the next turn shows it
                        raise
                await self._run_steps(emit)
            except asyncio.CancelledError:
                del self._hook_notes[kept:]  # the cancelled prompt's context must not go with the next prompt
                # Esc cancels the worker running this coroutine. Leave a history the model can be
                # asked again with: every tool call answered, the UI told, then let the cancel go on.
                self._answer_unanswered_calls(INTERRUPTED, SKIPPED, "interrupted", "skipped")
                with contextlib.suppress(Exception):
                    self._flush_save_error(emit)
                emit(TurnEnded("interrupted"))
                raise
            except Exception as exc:  # a provider breaking its contract, a bug, a failing sink
                self._answer_unanswered_calls(NOT_COMPLETED, NOT_COMPLETED, "not completed", "not completed")
                with contextlib.suppress(Exception):  # the sink may be what broke
                    emit(Notice("error", f"Internal error: {exc}"))
                with contextlib.suppress(Exception):
                    self._flush_save_error(emit)
                with contextlib.suppress(Exception):
                    emit(TurnEnded("error"))
        finally:
            if self.role == "main":
                self.watch.check(Cause.background())  # a process a command left behind, also after Esc or an error
            try:
                await self._checkpoint(emit, end=True)
            finally:
                if not self._shared_undo:
                    self._finish_undo()
                self._busy = False
                self._inflight = None
                self._pasted = ()

    async def run_call(self, call: ToolCall, emit: EventSink) -> ToolResult:
        """Run one tool call through every check of a model's call (deny rules, mode, arguments, hooks, protected-path
        and approval dialogs, redaction, limits), in the current mode, and return its result. The call and the result
        are recorded in this agent's own history; a cancel is let through after they were."""
        finished: list[ToolResult] = []

        def tap(event: AgentEvent) -> None:
            if isinstance(event, ToolFinished) and event.call_id == call.id:
                finished.append(event.result)
            emit(event)

        await self._run_tool_call(call, self.mode, tap)
        return finished[-1] if finished else ToolResult(f"{call.name} gave no result", True, "failed")

    @contextlib.asynccontextmanager
    async def operation(self, prompt: str, emit: EventSink) -> AsyncGenerator[Operation]:
        """A stretch of work that is one undo unit with its checkpoints but no model turn: `/undo` gives back `prompt`.
        Raises `AgentBusy` while a turn or another operation runs; the agent is busy inside. The body sets
        `keep = False` when it changed nothing, so no empty unit is left for `/undo`."""
        if self._busy:
            raise AgentBusy("a turn is already running")
        self._busy = True
        op = Operation()
        try:
            self._begin_undo(prompt)
            await self._checkpoint(emit, end=False)
            yield op
        finally:
            self.watch.check(Cause.background())
            try:
                await self._checkpoint(emit, end=True)
            finally:
                self._finish_undo(keep=op.keep)
                self._busy = False
                self._inflight = None

    async def ask_json(self, messages: list[Message]) -> str | ProviderError:
        """One model request outside the history, no tools: the reply text, or the provider's error. Not a step of a
        turn: its `Done` is only recorded as usage (a workflow call). A provider that breaks its contract is an error."""
        parts: list[str] = []
        started = time.monotonic()
        try:
            async with contextlib.aclosing(self.provider.stream(list(messages), [])) as events:
                async for event in events:
                    match event:
                        case TextDelta(text):
                            parts.append(text)
                        case Done(prompt_tokens, completion_tokens):
                            self._record_usage(prompt_tokens, completion_tokens, started, compaction=False, workflow=True)
                        case ProviderError():
                            return event
        except Exception as exc:
            return ProviderError(f"{type(exc).__name__}: {exc}")
        return "".join(parts)

    async def _checkpoint(self, emit: EventSink, *, end: bool) -> None:
        """Snapshot the workspace into the turn's undo record (`end`: when the turn is over, never raising: a cancel or
        an error there must not mask the turn's outcome, and the file-level undo is the fallback)."""
        store = self._checkpoints
        if store is None or (end and self._undo.current_before is None):
            return
        try:
            snap = await store.snapshot()
            if snap is not None:
                if end:
                    self._undo.mark(
                        after=snap.commit,
                        skipped_large=self._large_made(snap),
                        exclude_changed=exclude_signature(self.ctx.workspace) != self._exclude_before,
                    )
                else:
                    workspace = self.ctx.workspace
                    self._large_before = {name: file_signature(workspace, name) for name in snap.skipped_large}
                    self._exclude_before = exclude_signature(workspace)
                    # the time is taken after the snapshot, and the file clock is let pass it: what the turn
                    # creates is born later (Linux stamps files from a coarse clock, up to a tick behind)
                    before_time = time.time()
                    await asyncio.to_thread(settle, before_time)  # it sleeps: not on the event loop
                    self._undo.mark(before=snap.commit, before_time=before_time, large_before=snap.skipped_large)
            if self._checkpoint_warning is None:
                self._checkpoint_warning = store.take_warning()
            if self._checkpoint_warning is not None:
                emit(Notice("warn", self._checkpoint_warning))
                self._checkpoint_warning = None  # only once it was shown: a failing `emit` keeps it for the next call
        except asyncio.CancelledError:
            if not end:
                raise
        except Exception:
            pass  # a bug in the snapshot must not break the turn; the file-level undo still works

    def _large_made(self, snap: Snapshot) -> tuple[str, ...]:
        """Files over the size cap that the turn made, or that were over it before and it changed or deleted: the
        checkpoints hold none of them, so undo cannot put them back."""
        workspace = self.ctx.workspace
        made = [name for name in snap.skipped_large if name not in self._large_before]
        changed = [name for name, seen in self._large_before.items() if file_signature(workspace, name) != seen]
        return tuple(dict.fromkeys([*made, *changed]))

    async def last_turn_diff(self) -> TurnDiff | None:
        """The files the newest undoable turn changed (the turn `/undo` would undo next), None when there is no turn.
        Raises AgentBusy while a turn or an undo runs, and GitError when git fails and the file log has nothing."""
        if self._busy:
            raise AgentBusy("a turn is already running")
        turn = self._undo.peek()
        if turn is None:
            return None
        entry = self.ledger.find(turn) if self.ledger is not None else None
        foreign: Collection[str] = ()
        if entry is not None and self.ledger is not None:
            foreign = await self.ledger.foreign_paths(entry, self._checkpoints, self.ctx.workspace)
        return await build_diff(turn, self._checkpoints, self.ctx.workspace, foreign)

    def is_current_diff(self, diff: TurnDiff) -> bool:
        """Is the turn `diff` was built from still the newest one? False after a newer turn, an undo or a reset."""
        return self._undo.peek() is diff.turn

    async def load_diff_file(self, diff: TurnDiff, item: DiffItem) -> FileDiff:
        """The diff of one file of `diff`, read now."""
        return await load_file(diff, item)

    async def undo(self, emit: EventSink) -> str | None:
        """Undo the last turn: restore its files, drop it from the history and the session. Returns its prompt.

        None when there is nothing to undo.
        """
        if self._busy:
            raise AgentBusy("a turn is already running")
        turn = self._undo.pop()
        if turn is None:
            emit(Notice("info", NOTHING_TO_UNDO))
            return None
        entry = self.ledger.find(turn) if self.ledger is not None else None
        self._busy = True
        try:
            # The restore runs in a thread a cancel cannot stop: let it finish and cut the history
            # to match what it did, then hand the cancel on.
            cut_ids = {m.tool_call_id for m in self.history[turn.start :] if m.tool_call_id}
            task = asyncio.ensure_future(self._restore(turn, emit, entry))
            cancel: asyncio.CancelledError | None = None
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as exc:
                    cancel = cancel or exc
                    if self._checkpoint_read is not None:
                        self._checkpoint_read.cancel()  # kills the git process group; the file log restores instead
                except Exception:
                    break
            error = task.exception()
            outcome = None if error is not None else task.result()
            if error is None and outcome is None:  # Esc before any file was touched: still undoable, nothing changed
                self._undo.restack(turn)
                entry = None  # it still guards the others
                raise cancel or asyncio.CancelledError()
            result, from_checkpoints = outcome if outcome is not None else (None, False)
            if result is not None:
                self._accept_undone(turn, result)
            del self.history[turn.start :]
            kept = {m.tool_call_id for m in self.history if m.tool_call_id}
            for call_id in cut_ids - kept:
                self.tool_meta.pop(call_id, None)
            if self._reported is not None and self._reported.history_len > len(self.history):
                self._reported = None
            self._save(lambda: self._recorder.undo(turn.start))
            self._flush_save_error(emit)
            with contextlib.suppress(Exception):  # a status signal: a sink that raises must not break the undo
                emit(TodosChanged(self.todos))
            if cancel is not None:
                raise cancel
            if result is None:
                emit(Notice("error", f"Undo could not restore the files: {error}"))
                return turn.prompt
            notices = (
                checkpoint_notices(result, turn.skipped_large)
                if from_checkpoints
                else undo_notices(result, turn.bash)
            )
            for notice in notices:
                emit(notice)
        finally:
            if entry is not None and self.ledger is not None:
                self.ledger.undone(entry)  # what the undo skipped is still there: it guards the overlapping ones
            self._busy = False
        return turn.prompt

    def _accept_undone(self, turn: UndoTurn, result: UndoResult) -> None:
        """The restore is the agent's own write: the watched files it put back or deleted (only those of the undone
        turn's file log or checkpoint, which is all a restore touches) and the folders that went with them become the
        watch's baseline. Any other change to a watched file is still reported."""
        workspace = self.ctx.workspace.resolve()
        paths: list[Path] = []
        for kind, relative, _ in result.entries:
            if kind not in ("restored", "deleted"):
                continue
            path = workspace / relative
            paths.append(path)
            if kind == "deleted":  # folders the restore removed above it (made by the turn)
                paths.extend(p for p in path.parents if p.is_relative_to(workspace) and p != workspace and not os.path.lexists(p))
        for change in turn.changes.values():
            paths.extend(p for p in change.created_dirs if not os.path.lexists(p))
        if watched := [p for p in paths if self.watch.watches(p)]:
            with contextlib.suppress(Exception):
                self.watch.accept(watched)

    async def _restore(
        self, turn: UndoTurn, emit: EventSink, entry: LedgerEntry | None = None
    ) -> tuple[UndoResult, bool] | None:
        """Restore the turn from its two checkpoints: `(result, True)`. Without a complete pair, when git fails before
        anything was written, or when Esc stops git, from the file-level log: `(result, False)` (which is quick).
        Paths that another session's overlapping turn changed are left alone (`entry`: this turn in the ledger); when
        what they changed cannot be read, no file of this turn is touched. None: Esc came while that was being
        looked up, before any file was touched."""
        store = self._checkpoints
        foreign = Foreign()
        if entry is not None and self.ledger is not None:
            look = asyncio.create_task(self.ledger.foreign_paths(entry, store, self.ctx.workspace))
            self._checkpoint_read = look  # Esc cancels it (it may wait for the snapshot lock, then for a snapshot)
            try:
                foreign = await look
            except asyncio.CancelledError:
                if not look.cancelled():
                    raise
                with contextlib.suppress(Exception):
                    emit(Notice("warn", "Undo interrupted; nothing was changed"))
                return None
            except Exception as exc:
                foreign.failures.append(f"{type(exc).__name__}: {exc}")
            finally:
                self._checkpoint_read = None
            if foreign.failures:
                why = notice_text("; ".join(foreign.failures))
                with contextlib.suppress(Exception):  # a sink that raises must not stop the undo
                    emit(Notice("warn", f"Could not tell what other sessions changed ({why}); no file of this turn is touched"))
        skip: dict[str, str] = dict(foreign)
        if turn.before and turn.after and store is not None:
            read = asyncio.create_task(self._read_checkpoint(store, turn.before, turn.after))
            self._checkpoint_read = read
            loaded: tuple[list[DiffEntry], dict[str, bytes]] | None = None
            try:
                loaded = await read
            except asyncio.CancelledError:
                if not read.cancelled():
                    raise
            except Exception as exc:  # git failed or the folder is gone: nothing was touched yet
                why = notice_text(f"{type(exc).__name__}: {exc}")
                with contextlib.suppress(Exception):  # a sink that raises must not stop the undo
                    emit(Notice("warn", f"Checkpoint restore failed ({why}); using the file-level undo"))
            finally:
                self._checkpoint_read = None
            if loaded is not None:
                entries, blobs = loaded
                if foreign.failures:
                    skip = {**self._blanket(turn, entries), **skip}
                return await asyncio.to_thread(self._restore_checkpoint, turn, entries, blobs, skip), True
        if foreign.failures:
            skip = {**self._blanket(turn, ()), **skip}
        return await asyncio.to_thread(restore_turn, turn, self.ctx.workspace, skip), False

    def _blanket(self, turn: UndoTurn, entries: Collection[DiffEntry]) -> dict[str, str]:
        """Every path this turn changed, for a restore that must leave all of them alone."""
        names = {e.path for e in entries}
        root = self.ctx.workspace.resolve()
        for path in turn.changes:
            with contextlib.suppress(ValueError):
                names.add(path.relative_to(root).as_posix())
        return dict.fromkeys(names, UNSURE_OTHER)

    @staticmethod
    async def _read_checkpoint(
        store: CheckpointStore, before: str, after: str
    ) -> tuple[list[DiffEntry], dict[str, bytes]]:
        entries = await store.diff(before, after)
        return entries, await store.read_blobs([e.old_sha for e in entries if e.old_sha])

    def _restore_checkpoint(
        self, turn: UndoTurn, entries: list[DiffEntry], blobs: dict[str, bytes], skip: Mapping[str, str] | None = None
    ) -> UndoResult:
        """The restore from the snapshots, then the write/edit changes to files the snapshots leave out (ignored files,
        `.agent/` files) from the file log. Blocking."""
        workspace = self.ctx.workspace
        result = restore_checkpoint(
            entries,
            blobs,
            workspace,
            before_time=turn.before_time,
            large_before=frozenset(turn.large_before),
            exclude_changed=turn.exclude_changed,
            skip=skip,
        )
        paths = guard_of(self.ctx).paths

        def protected(path: Path) -> bool:
            try:
                return paths.is_protected(path)
            except Exception:
                return True  # when unsure, leave it alone

        extra = restore_unsnapshotted(turn, entries, workspace, protected, skip)
        return UndoResult([*result.entries, *extra.entries])

    def _answer_unanswered_calls(
        self, in_flight_text: str, skipped_text: str, in_flight_summary: str, skipped_summary: str
    ) -> None:
        """Give each tool call of the last assistant message that has no result one."""
        last = next((i for i in range(len(self.history) - 1, -1, -1) if self.history[i].role == "assistant"), None)
        if last is None:
            return
        answered = {m.tool_call_id for m in self.history[last + 1 :] if m.role == "tool"}
        for call in self.history[last].tool_calls:
            if call.id not in answered:
                in_flight = call.id == self._inflight
                self._record(
                    Message("tool", in_flight_text if in_flight else skipped_text, tool_call_id=call.id, name=call.name),
                    ToolMeta(self._label_for(call), in_flight_summary if in_flight else skipped_summary),
                )

    def _label_for(self, call: ToolCall) -> str:
        """The label the tool gives the call, or its name when the tool is unknown or cannot label it."""
        tool = next((t for t in self.all_tools() if t.name == call.name), None)
        if tool is None:
            return call.name
        try:
            return tool.label(call.arguments)
        except Exception:
            return call.name

    async def _run_steps(self, emit: EventSink) -> None:
        continuations = 0  # times a `Stop` hook sent this user turn on
        for step in range(self.max_steps):
            self._step = step + 1
            await self._maybe_compact(emit)
            mode = self.mode  # snapshot: a Tab press mid-step applies from the next request
            self._phase(emit, "waiting")
            requested = await self._request(mode, emit)
            if requested is None:  # provider error, already reported
                return
            assistant, total = requested
            self._record(assistant)
            self._anchor(total)
            self._flush_save_error(emit)
            if not assistant.tool_calls:
                # On the last step there is none left to continue with: the turn simply ends.
                if step + 1 < self.max_steps:
                    reason = await self._stop_hooks(continuations, emit)
                else:
                    reason = None
                    if not self.subagent and self.hooks.has("Stop"):
                        emit(Notice("info", "Stop hooks skipped: no step left"))
                if reason is None:
                    emit(TurnEnded("done"))
                    return
                continuations += 1
                self._record(Message("user", f"{STOP_NOTE_PREFIX}{reason}"))  # counts as a step: the loop goes on
                continue
            for tool_call in assistant.tool_calls:
                await self._run_tool_call(tool_call, mode, emit)
        emit(Notice("warn", f"Step limit ({self.max_steps}) reached"))
        emit(TurnEnded("step_limit"))

    def _phase(self, emit: EventSink, kind: PhaseKind, detail: str = "") -> None:
        """Tell the UI what the main agent is doing now. Display only: a failing sink never costs the turn."""
        if self.role != "main":
            return
        with contextlib.suppress(Exception):
            emit(Phase(kind, self._step, self.max_steps, detail))

    def _anchor(self, total: int | None) -> None:
        """Right after a reply was recorded: the model's count `total` covers the whole history up to it."""
        if total is not None:
            self._reported = Reported(total, len(self.history))

    async def _request(self, mode: Mode, emit: EventSink) -> tuple[Message, int | None] | None:
        """One model call. Returns the assistant message and the model's token count for the exchange
        (None if it sent none), or None after a ProviderError.

        The count is only kept (`_reported`) once its reply is in the history: a reply that is
        dropped must not make the next message look already counted.
        """
        system = self._system(mode)
        specs = [t.spec() for t in self.tools_for(mode)]
        messages = [system, *self.history]
        notes = [n.content for n in (self._hook_note(), self._todo_note()) if n is not None]
        if notes:
            # Just before the latest user message: the model reads it next to what it is asked now.
            # Trade-off: the note changes the request prefix after it, so a provider's prompt cache
            # is reused only up to here; placing it last would keep the cache but sit apart from the question.
            last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=len(messages))
            messages.insert(last_user, Message("system", "\n\n".join(notes)))
        parts: list[str] = []
        calls: list[ToolCall] = []
        # aclosing so that leaving the loop early (cancel, error) closes the provider stream now
        failure: str | None = None
        total: int | None = None
        started = time.monotonic()
        writing = False
        try:
            async with contextlib.aclosing(self.provider.stream(messages, specs)) as events:
                async for event in events:
                    if not isinstance(event, ProviderError):
                        self._hook_notes.clear()  # the model is answering: this request has it (one request only)
                    match event:
                        case TextDelta(text):
                            parts.append(text)
                            if not writing:
                                writing = True
                                self._phase(emit, "writing")
                            emit(AssistantText(text))
                        case ToolCallEvent(call):
                            calls.append(call)
                        case Done(prompt_tokens, completion_tokens):
                            total = prompt_tokens + completion_tokens
                            self._record_usage(prompt_tokens, completion_tokens, started, compaction=False)
                            emit(UsageUpdate(total, self.provider.context_window, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))
                        case ProviderError(message):
                            failure = message
                            break  # report after the stream is closed: a cancel in aclose() must not repeat it
        except asyncio.CancelledError:
            # Cancelled mid-stream (the stream is closed by now). Keep the text the user already
            # saw, marked as cut short, as the assistant message. Tool calls collected so far are
            # dropped: none has run, and keeping them would need a fake result for each, for a
            # request the user abandoned. A cancel before any text leaves only the user message.
            if parts:
                self._record(Message("assistant", "".join(parts) + PARTIAL_MARK))
                self._anchor(total)
            raise
        if failure is not None:
            if parts:
                self._record(Message("assistant", "".join(parts)))
                self._anchor(total)
            self._flush_save_error(emit)
            emit(Notice("error", failure))
            emit(TurnEnded("error"))
            return None
        return Message("assistant", "".join(parts), calls), total

    def _vision_asker(self, model: str, emit: EventSink) -> Callable[[], Awaitable[bool]]:
        async def ask() -> bool:
            if self.vision is None:
                return False
            return await self.vision.has(model, lambda text: emit(Notice("warn", text)))

        return ask

    async def _run_tool_call(self, call: ToolCall, mode: Mode, emit: EventSink) -> None:
        tool = next((t for t in self.all_tools() if t.name == call.name), None)
        self._inflight = call.id
        label = tool.label(call.arguments) if tool else call.name
        emit(ToolStarted(call.id, label, task=tool is not None and tool.name == "task", name=call.name, args=dict(call.arguments)))
        preview: Preview | None = None
        target: Path | None = None  # write/edit: the file `prepare` resolved
        applied = False
        ran = False  # did the tool itself do its work? (not refused, denied or blocked first)
        cancel: asyncio.CancelledError | None = None
        try:
            if tool is None:
                result = ToolResult(f"Unknown tool: {call.name}", True, "unknown tool")
            elif (verdict := self.rules.deny(call.name, call.arguments, tool.grant_arg)) is not None:
                # Before everything else, in every mode, for every tool: a deny rule only ever refuses.
                text = f"denied by rule {escape(verdict.rule.text)} ({verdict.source})"  # the text came from a file
                result = ToolResult(text, True, "denied by rule")
            elif mode is Mode.PLAN and self._plan_blocked(tool):
                result = ToolResult(f"refused: {call.name} is unavailable in plan mode", True, "refused: plan mode")
            elif (error := check_args(tool.parameters, call.arguments)) is not None:
                result = ToolResult(f"{call.name}: {error}", True, "invalid arguments")
            elif (refused := await self._pre_hooks(tool, call, mode, emit)) is not None:
                result = refused
            elif isinstance(gate := await self._confirm_protected(tool, call, emit), ToolResult):
                result = gate
            else:
                # `gate` is True when the user allowed this call to read a protected file: the tool's own check of
                # the file it opens (a symlink may have changed since the path check) lets that one through.
                call_ctx = dataclasses.replace(
                    self.ctx, progress=lambda event: emit(TaskProgress(call.id, event)), protected_ok=gate
                )
                if call.name == "read":  # asked only when `read` has found an image, about the model that gets it
                    model = self.provider.model
                    call_ctx = dataclasses.replace(call_ctx, has_vision=self._vision_asker(model, emit), model=model)
                if call.name == "edit_image":
                    call_ctx = dataclasses.replace(call_ctx, pasted=self._pasted)
                prepared = await tool.prepare(call.arguments, call_ctx)
                if isinstance(prepared, ToolResult):
                    result, ran = prepared, True  # the tool did its work (and failed, or finished) while preparing
                else:
                    target = prepared.target
                    preview = guard_of(self.ctx).redact_preview(prepared.preview)
                    result, applied, ran = await self._approve_and_run(tool, call, prepared, mode, emit)
        except _Landed as landed:
            result, applied, cancel = landed.result, landed.applied, landed.cancel  # not `ran`: a cancel is on its way
        except Exception as exc:  # NFR4: a failing tool is an error result, never a crashed turn
            result = ToolResult(f"{call.name} failed: {exc}", True, "failed")
            applied = False
        edited = ran and target is not None and not result.is_error and call.name in ("write", "edit")
        if self.format is not None and edited and target is not None:
            try:
                result = await self._format(target, result)
            except asyncio.CancelledError as exc:  # Esc while a formatter runs: it was killed, the file is as written
                cancel = cancel or exc
            except Exception as exc:  # a broken formatter step never costs the result the tool produced
                emit(Notice("warn", f"format failed: {escape(str(exc))[:200]}"))
        if self.lsp is not None and edited and target is not None and cancel is None:
            try:
                result = await self._diagnostics(target, result, emit)
            except asyncio.CancelledError as exc:  # Esc while a server answers: the file was written, keep the result
                cancel = cancel or exc
            except Exception as exc:  # a broken server step never costs the result the tool produced
                emit(Notice("warn", f"lsp failed: {escape(str(exc))[:200]}"))
        # Redaction first, the limit after it: a secret that straddles the cut must not leave its first characters.
        guard = guard_of(self.ctx)
        result, hidden = _window(result)
        result = guard.redact_result(result)
        if result.output == REDACTION_FAILED:  # the message replaced the output: nothing of it was cut
            hidden = 0
        result = dataclasses.replace(result, output=_truncate(result.output, hidden))
        if result.display is not None:
            result = dataclasses.replace(result, display=truncate_ansi(result.display))
        if ran and self.hooks.has("PostToolUse", call.name):
            try:
                result = await self._post_hooks(call, result, emit)
            except asyncio.CancelledError as exc:  # Esc during a hook: the call did finish, so its result is kept
                cancel = cancel or exc
            except Exception as exc:  # a broken hook step never costs the result the tool produced
                emit(hook_failed("PostToolUse", exc))
        self._record(
            Message("tool", result.output, tool_call_id=call.id, name=call.name, images=result.images),
            ToolMeta(label, result.summary)
        )
        self._inflight = None
        self._flush_save_error(emit)
        emit(ToolFinished(call.id, result, preview, applied))
        if call.name == "todo" and not result.is_error:
            emit(TodosChanged(self.todos))
        if cancel is not None:
            raise cancel

    async def _format(self, target: Path, result: ToolResult) -> ToolResult:
        """Run the matching formatters on the file a write/edit just changed. The undo log rehashes the file when one
        changed it, so undo, the diff and the next call's checks see the formatted content. One note line per formatter
        that changed the file or failed, appended to the result."""
        assert self.format is not None
        if self.watch.watches(target):  # the agent's own config: a formatter must not rewrite what the watch guards
            return result
        try:
            written = await asyncio.to_thread(read_nofollow, target, SOURCE_LIMIT)
        except OSError:
            return result
        outcomes = await self.format.format(target, written)
        if not outcomes:
            return result
        if any(o.changed_lines for o in outcomes):
            self._undo.after(target)
        notes = [
            f"[format {o.name} failed: {o.error}]" if o.error else f"[formatted by {o.name}: {o.changed_lines} lines changed]"
            for o in outcomes
            if o.error or o.changed_lines
        ]
        if not notes:
            return result
        return dataclasses.replace(result, output="\n".join([result.output, *notes]), display=None)

    async def _diagnostics(self, target: Path, result: ToolResult, emit: EventSink) -> ToolResult:
        """The language servers' errors for the file a write/edit just changed, appended to the result (which is then
        redacted and cut like any output; the tool's own text is never cut for it)."""
        assert self.lsp is not None
        text = await asyncio.to_thread(_read_source, target)
        if text is None:
            return result
        budget = TOOL_OUTPUT_LIMIT - len(result.output) - 2
        block = await self.lsp.block(target, text, emit, budget)
        if not block:
            return result
        return dataclasses.replace(result, output=f"{result.output}\n\n{block}", display=None)

    def _hook_payload(self, **fields: object) -> dict[str, object]:
        """What every hook gets besides the event's own fields: the session and the workspace."""
        session_id = getattr(self._recorder, "session_id", "")
        payload = {**fields, "session_id": session_id if isinstance(session_id, str) else "", "cwd": str(self.ctx.workspace)}
        return {**payload, "agent": self.subagent} if self.subagent else payload  # a custom child's tool events

    def _redact_hook_text(self, text: str) -> str:
        """Secrets out of text a hook produced, before it is shown. Fails closed: nothing of it when redaction fails.
        Not counted: the same text is counted when it reaches the model."""
        try:
            return guard_of(self.ctx).redactor.redact(text)[0]
        except Exception:
            return ""

    def _add_hook_context(self, event: str, texts: Sequence[str]) -> list[Notice]:
        """Keep a hook's context for the next request, redacted (fail closed: nothing of it when redaction fails).
        Returns the notices to show: a warning when the text was withheld."""
        if not texts:
            return []
        guard = guard_of(self.ctx)
        try:
            text, count = guard.redactor.redact("\n\n".join(texts))
        except Exception:
            return [Notice("warn", f"hook {event}: its text was withheld (redaction failed)")]
        guard.redactions += count
        if text:
            self._hook_notes.append(text)
        return []

    async def prompt_hooks(self, text: str, emit: EventSink) -> HookDecision:
        """Run the `UserPromptSubmit` hooks for a prompt before it is recorded; the app sends `decision.context` to
        `run_turn` and, on a block, drops the prompt. Never raises (but for a cancel); the notices go to `emit`."""
        if self.subagent or not self.hooks.has("UserPromptSubmit"):
            return HookDecision(False, "", [], [])
        decision = HookDecision(False, "", [], [])
        try:
            decision = await self.hooks.run("UserPromptSubmit", self._hook_payload(prompt=text))
            for notice in self.hooks.notices(decision, self._redact_hook_text):
                emit(notice)
        except Exception as exc:  # a broken hook step never costs the user's prompt
            with contextlib.suppress(Exception):
                emit(hook_failed("UserPromptSubmit", exc))
        return decision

    async def session_started(self, source: Literal["startup", "resume", "new", "switch"], emit: EventSink) -> None:
        """Run the `SessionStart` hooks once the session is ready; their context goes to the next request. Never raises
        (but for a cancel). `reset` drops context kept for the session that ended, so call this after it."""
        if self.subagent or not self.hooks.has("SessionStart"):
            return
        try:
            decision = await self.hooks.run("SessionStart", self._hook_payload(source=source))
            notices = [*self.hooks.notices(decision, self._redact_hook_text), *self._add_hook_context("SessionStart", decision.context)]
            for notice in notices:
                emit(notice)
        except Exception as exc:
            with contextlib.suppress(Exception):
                emit(hook_failed("SessionStart", exc))

    async def _stop_hooks(self, continuations: int, emit: EventSink) -> str | None:
        """Run the `Stop` hooks when a turn would end: the reason (redacted) to continue with, or None to end it.
        At most `STOP_CONTINUATIONS` per user turn, then a warning and no more hook runs."""
        if self.subagent or not self.hooks.has("Stop"):
            return None
        if continuations >= STOP_CONTINUATIONS:
            emit(Notice("warn", "Stop hook limit reached"))
            return None
        try:
            self._phase(emit, "hook", "Stop")
            decision = await self.hooks.run("Stop", self._hook_payload(continuations=continuations))
            for notice in self.hooks.notices(decision, self._redact_hook_text):
                emit(notice)
        except Exception as exc:  # a broken hook step never costs the finished turn
            with contextlib.suppress(Exception):
                emit(hook_failed("Stop", exc))
            return None
        if not decision.blocked:
            return None
        return self._redact_hook_text(decision.reason) or DEFAULT_REASON

    async def _pre_hooks(self, tool: Tool, call: ToolCall, mode: Mode, emit: EventSink) -> ToolResult | None:
        """Run the `PreToolUse` hooks for a call; the refusal when one blocks, else None. Before every dialog.

        The call has passed the deny-rule, plan-mode and argument checks by now, so a call that goes on is reported as
        `running` here: a refused call never shows it."""
        if not self.hooks.has("PreToolUse", call.name):
            self._phase(emit, "running", call.name)
            return None
        payload = self._hook_payload(tool=call.name, args=call.arguments, mode=mode.value)
        self._phase(emit, "hook", "PreToolUse")
        decision = await self.hooks.run("PreToolUse", payload, tool=call.name)
        for notice in self.hooks.notices(decision, self._redact_hook_text):
            emit(notice)
        if not decision.blocked:
            self._phase(emit, "running", call.name)
            return None
        return ToolResult(f"blocked by hook: {decision.reason}", True, "blocked by hook")

    async def _post_hooks(self, call: ToolCall, result: ToolResult, emit: EventSink) -> ToolResult:
        """Run the `PostToolUse` hooks on a finished call (`result` is redacted and cut already) and append what they
        add, redacted, to the output the model sees: the tool's own output is kept, the hook's text is cut to fit."""
        payload = self._hook_payload(tool=call.name, args=call.arguments, result=result.output, is_error=result.is_error)
        self._phase(emit, "hook", "PostToolUse")
        decision = await self.hooks.run("PostToolUse", payload, tool=call.name)
        for notice in self.hooks.notices(decision, self._redact_hook_text):
            emit(notice)
        if not decision.context:
            return result
        guard = guard_of(self.ctx)
        try:
            text, count = guard.redactor.redact("\n\n".join(decision.context))
        except Exception:  # fail closed, like a tool's output: nothing of the hook's text
            emit(Notice("warn", "hook PostToolUse: its text was withheld (redaction failed)"))
            return result
        guard.redactions += count
        prefix = "\n[hook PostToolUse] "
        room = TOOL_OUTPUT_LIMIT - len(result.output) - len(prefix)
        if room <= 0 or not text:
            return result
        # The hook's text changes the output: the display copy (colours) would no longer match it.
        return dataclasses.replace(result, output=result.output + prefix + text[:room], display=None)

    def _rule_to_save(self, tool: Tool, call: ToolCall) -> str:
        """The exact rule for this call (what "Always here" saves). Raises `RuleError` when there is none."""
        key = resolve_key(call.name, tool.grant_arg)
        value = call.arguments.get(key) if key is not None else None
        if key is not None and not isinstance(value, str):
            raise RuleError("the call has no text value to build a rule from")
        return exact_rule_text(call.name, value)

    def _offered_rule(self, tool: Tool, call: ToolCall) -> str | None:
        """The rule the dialog offers to save, or None when there is none to offer."""
        if not self.rules.can_save:
            return None
        try:
            return self._rule_to_save(tool, call)
        except RuleError:
            return None

    async def _save_rule(self, tool: Tool, call: ToolCall, emit: EventSink) -> None:
        """"Always here": save the exact rule for this call; a failure is a warning, the call is allowed for this run."""
        try:
            rule = self._rule_to_save(tool, call)
            await asyncio.to_thread(self.rules.save_project, rule)  # file and trust-store I/O: not on the event loop
        except RuleSavedNotTrusted as exc:  # it is saved and active; only the trust could not be written
            emit(Notice("warn", escape(str(exc))))
        except (OSError, ValueError) as exc:  # RuleError is a ValueError
            emit(Notice("warn", escape(f"could not save rule to .agent/settings.json: {exc}; allowed for this run only")))

    def _edits_agent_config(self, tool: Tool, call: ToolCall, target: Path | None = None) -> bool:
        """Does a write or edit target `.agent/settings.json`, `.agent/mcp.json` or a file under `.agent/agents` or
        `.agent/workflows` of the workspace? Those files grant the agent its own permissions, processes and tools: no
        rule, grant or "Always" may approve a change to them.
        `target` is the path the tool resolved and checked; without one the argument is resolved here. Names are
        compared case-insensitively (APFS: `.AGENT/MCP.json` is the same file), and by identity when both exist."""
        if tool.name not in ("write", "edit"):
            return False
        if target is None:
            raw = call.arguments.get("path")
            if not isinstance(raw, str):
                return False
            try:
                target = resolve_path(self.ctx, raw)
            except InvalidPath:
                return False
        relative = _relative_parts(target, self.ctx.workspace.resolve())
        if relative in AGENT_CONFIG_PARTS or (relative is not None and relative[:2] in AGENT_GUARDED_FOLDERS):
            return True
        if self._links_to_agent_file(target):
            return True
        for protected in agent_config_files(self.ctx.workspace):
            if target == protected or (target.exists() and protected.exists() and target.samefile(protected)):
                return True
        return False

    def _links_to_agent_file(self, target: Path) -> bool:
        """Is `target` an existing file with other hard links that is the same file as one directly under
        `.agent/agents` or `.agent/workflows`? Then a write to it is a write to that file. Bounded: those folders' own
        entries only, links not followed."""
        try:
            if target.stat().st_nlink <= 1:
                return False
        except FileNotFoundError:
            return False
        except OSError:
            return True  # cannot tell: ask
        for parts in sorted(AGENT_GUARDED_FOLDERS):
            try:
                folder = self.ctx.workspace.resolve().joinpath(*parts)
                if folder.is_symlink():
                    continue
                with os.scandir(folder) as entries:
                    for count, entry in enumerate(entries):
                        if count >= AGENT_FILE_SCAN_LIMIT:
                            return True  # too many to check: ask
                        if entry.is_file(follow_symlinks=False) and target.samefile(entry.path):
                            return True
            except FileNotFoundError:
                continue
            except OSError:
                return True  # cannot tell: ask
        return False

    def _accepts_edit(self, tool: Tool, prepared: Prepared, mode: Mode) -> bool:
        """Accept-edits mode approves a write or edit without the dialog when the path the tool resolved and checked
        (`Prepared.target`; none: ask) is in the workspace and is not under a `.git` or `.venv` segment (a language
        server may run a program from `.venv/bin`), the agent config files,
        `.agent/commands`, `.agent/skills`, `.agent/agents`, `.agent/workflows` or the agent's own config, data and
        sessions folders (a workspace that holds the home folder). Names are compared case-insensitively (APFS) and
        without a macOS `/System/Volumes/Data` prefix; an existing file with more than one hard link (or one that cannot be checked) always asks. Never asked
        for a guarded call."""
        target = prepared.target
        if mode is not Mode.EDITS or tool.name not in ACCEPTED_EDIT_TOOLS or target is None:
            return False
        workspace = self.ctx.workspace.resolve()
        relative = _relative_parts(target, workspace)
        if relative is None or ".git" in relative or ".venv" in relative or relative[:2] in AGENT_ASKING_PARTS:
            return False
        folders = (
            workspace / ".agent" / "agents",
            workspace / ".agent" / "commands",
            workspace / ".agent" / "skills",
            workspace / ".agent" / "workflows",
            *guard_of(self.ctx).own_folders,
        )
        for folder in folders:
            try:
                resolved = folder.resolve()  # a symlinked `.agent` or data folder: where it really is
            except (OSError, RuntimeError):
                return False
            if _relative_parts(target, folder) is not None or _relative_parts(target, resolved) is not None:
                return False
        try:
            links = target.stat().st_nlink  # the target is resolved: this is the file the tool will change
        except FileNotFoundError:
            return True  # a new file
        except (OSError, ValueError):
            return False
        return links <= 1  # a hard-linked file may also be named inside one of the folders above: always ask

    def _protected_target(self, tool: Tool, call: ToolCall) -> Path | None:
        """The protected file a read, a glob with a wildcard-free pattern or a grep of one file names, as the file
        system sees it (symlinks, `~` and `..` resolved); None when the call names none, or it does not exist."""
        args = call.arguments
        raw: object
        if tool.name in ("read", "grep"):
            raw = args.get("path")
        elif tool.name == "glob":
            pattern, base = args.get("pattern"), args.get("path") or "."
            if not isinstance(pattern, str) or not isinstance(base, str) or any(c in pattern for c in "*?["):
                return None
            raw = str(PurePath(base) / pattern)
        else:
            return None
        if not isinstance(raw, str) or not raw:
            return None
        try:
            target = resolve_path(self.ctx, raw)
        except InvalidPath:
            return None
        return target if target.is_file() and guard_of(self.ctx).paths.is_protected(target) else None

    async def _confirm_protected(self, tool: Tool, call: ToolCall, emit: EventSink) -> ToolResult | bool:
        """Ask before a call reads a protected path, whatever the rules, grants and mode say (deny rules ran before):
        only Allow once or Deny. Returns the refusal; else whether the user allowed a protected read (False: the call
        names no protected file)."""
        target = self._protected_target(tool, call)  # a few stat calls: cheap, like `_edits_agent_config`
        if target is None:
            return False
        request = ApprovalRequest(
            call.name,
            tool.label(call.arguments),
            None,
            False,
            None,
            protected=True,
            protected_path=str(target),
            subagent=self.subagent,
        )
        self._phase(emit, "approval", call.name)
        decision = await self.approver.ask(request)
        if decision.choice in (Choice.ONCE, Choice.ALWAYS, Choice.PROJECT):  # an Always is not offered: it counts as once
            self._phase(emit, "running", call.name)
            return True
        note = f" Note: {decision.note}" if decision.note else ""
        return ToolResult(f"User denied this call.{note}", True, "denied")

    async def _approve_and_run(
        self, tool: Tool, call: ToolCall, prepared: Prepared, mode: Mode, emit: EventSink
    ) -> tuple[ToolResult, bool, bool]:
        """The result, whether it changed something, and whether the tool ran (False: the user said no)."""
        # Order: deny rules (in `_run_tool_call`), allow rules, this launch's "Always", accept-edits mode for
        # `write`/`edit`, then the dialog. Neither rules, grants nor accept-edits mode approve anything in plan
        # mode (the last never applies there), nor a change to the agent's own settings files.
        guarded = tool.needs_approval and self._edits_agent_config(tool, call, prepared.target)
        asked = False  # the user approved this very call in the dialog (not a rule, a grant or accept-edits)
        if tool.needs_approval and (
            guarded
            or not (
                (mode is not Mode.PLAN and self.rules.allow(call.name, call.arguments, tool.grant_arg))
                or self.grants.covers(call.name, call.arguments, mode, tool.grant_arg)
                or self._accepts_edit(tool, prepared, mode)
                or (mode is not Mode.PLAN and self.preapproved is not None and self.preapproved(call))
            )
        ):
            allow_always = mode is not Mode.PLAN and not guarded
            save_rule = self._offered_rule(tool, call) if allow_always else None
            request = ApprovalRequest(
                call.name,
                tool.label(call.arguments),
                guard_of(self.ctx).redact_preview(prepared.preview),
                allow_always,
                save_rule,
                subagent=self.subagent,
            )
            self._phase(emit, "approval", call.name)
            decision = await self.approver.ask(request)
            if decision.choice not in (Choice.ONCE, Choice.ALWAYS, Choice.PROJECT):  # anything but a clear yes is a no
                note = f" Note: {decision.note}" if decision.note else ""
                return ToolResult(f"User denied this call.{note}", True, "denied"), False, False
            asked = True
            self._phase(emit, "running", call.name)
            if decision.choice in (Choice.ALWAYS, Choice.PROJECT) and allow_always:
                self.grants.add(call.name, call.arguments, tool.grant_arg)  # for this run, even if saving fails
                if not any(tool is t for t in self.tools):
                    self._granted_extra[call.name] = tool
                if decision.choice is Choice.PROJECT:
                    await self._save_rule(tool, call, emit)
        if tool.name in UNDO_UNTRACKED:
            self._undo.note_bash()
        self.watch.check(Cause.background())  # whatever changed while the dialog was open is not this call's
        approved = prepared.target if asked else None  # the file a dialog-approved write/edit may change
        if not tool.runs_to_completion():
            result: ToolResult | None = None
            try:
                result = await prepared.execute()  # cancellable at once; bash kills its process group
            finally:
                self._watch_after(tool, call, result, approved)  # also after Esc: it may have written before it died
            return result, not result.is_error, True
        # A write or edit runs in a thread that a cancel cannot stop. Let it finish so history
        # and the UI report what happened to the file, then hand the cancel on.
        task = asyncio.ensure_future(prepared.execute())
        cancel: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancel = cancel or exc  # a repeated cancel must not abandon the write
            except Exception:
                break  # the task holds it; read below
        try:
            result = task.result()
        except Exception as exc:  # NFR4
            result = ToolResult(f"{call.name} failed: {exc}", True, "failed")
        applied = not result.is_error
        self._watch_after(tool, call, result, approved)
        if cancel is not None:
            raise _Landed(result, applied, cancel)
        return result, applied, True

    def _watch_after(self, tool: Tool, call: ToolCall, result: ToolResult | None, approved: Path | None) -> None:
        """The check after a call. A `write` or `edit` that worked, on a watched file the user approved in the dialog
        for this call (`approved`, the target the tool resolved), is the agent's own write: only that file's new
        state is accepted, and any other change is still reported (as background). Anything else (another tool, or a
        write approved by a rule, a grant or accept-edits) is checked, and a change is blamed on the call."""
        if (
            call.name in ACCEPTED_EDIT_TOOLS
            and approved is not None
            and result is not None
            and not result.is_error
            and self.watch.watches(approved)
        ):
            self.watch.accept([approved])
            return
        key = resolve_key(call.name, tool.grant_arg)
        value = call.arguments.get(key) if key else None
        self.watch.check(Cause.tool(call.name, value if isinstance(value, str) else None))
