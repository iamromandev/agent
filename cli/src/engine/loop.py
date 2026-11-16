"""The agent loop: model request -> tool calls -> model request, until the reply has no tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import platform
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path, PurePath
from typing import Literal

from src.engine.approval import ApprovalRequest, Approver, Choice, Grants
from src.engine.compaction import (
    MIN_SUMMARY_GAIN,
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
from src.engine.events import (
    AgentEvent,
    AssistantText,
    Compacting,
    EventSink,
    Notice,
    TaskProgress,
    TodosChanged,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)
from src.engine.guardrails.redact import drop_partial_tail
from src.engine.guardrails.setup import REDACTION_FAILED, guard_of
from src.engine.hooks.setup import Hooks, notice_text
from src.engine.instructions import Instructions, load_instructions
from src.engine.mode import Mode
from src.engine.permissions.config import agent_config_files
from src.engine.permissions.rules import RuleError, exact_rule_text, resolve_key
from src.engine.permissions.setup import Rules, RuleSavedNotTrusted
from src.engine.prompt import build_system_prompt, build_task_prompt
from src.engine.provider.base import (
    Done,
    Message,
    Provider,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from src.engine.session import NullRecorder, Recorder, ToolMeta
from src.engine.session.records import compaction_notice
from src.engine.task import TaskTool
from src.engine.text import escape
from src.engine.todo import INTERRUPTED, NOT_COMPLETED, SKIPPED, TodoItem, active, current_todos, render_note
from src.engine.tools.base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from src.engine.tools.paths import InvalidPath, resolve_path
from src.engine.tools.skill import SkillTool
from src.engine.tools.text import TOOL_OUTPUT_LIMIT, keep_sgr, truncate, truncate_ansi
from src.engine.undo import (
    NOTHING_TO_UNDO,
    SNAPSHOT_FILE_LIMIT,
    SNAPSHOT_TOTAL_LIMIT,
    UNDO_UNTRACKED,
    UndoLog,
    restore_turn,
    undo_notices,
)

# Marks a partial assistant message that a cancel cut short, so the model does not take it for a
# complete answer on the next turn.
# What redaction reads of a result: the output limit plus this much, so a secret that starts before the cut and ends a
# little after it is still seen whole; one that has no end in sight is replaced to the end of what is read. The rest of
# the output is dropped before redaction, because redaction must not cost more than the model could ever see.
REDACT_MARGIN = 4_000
DISPLAY_FACTOR = 4  # the display copy also holds colour codes, which do not count towards its limit
PARTIAL_MARK = "\n[interrupted by user]"
NOTHING_TO_COMPACT = "Nothing to compact"
NOT_SUMMARIZED = "Could not summarize the earlier conversation; continuing with pruned history"
STILL_FULL = "Context is still full \u2014 start a new session with /new"
SUMMARY_BUDGET = 0.6  # of the window: how much older conversation one summary request may carry


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
    ) -> None:
        self.provider = provider
        self.hooks = hooks if hooks is not None else Hooks.empty()  # the `task` child never gets any
        self.rules = rules if rules is not None else Rules.empty()  # not part of `Grants`: `reset()` leaves it alone
        self.role = role
        self.tools = list(tools)
        if task_tools is not None and role == "main":
            self.tools.append(TaskTool(self, list(task_tools), task_max_steps))
        self._extra_tools = extra_tools  # asked at every use: tools that come and go while the agent runs (MCP)
        self._granted_extra: dict[str, Tool] = {}  # "Always" given to an extra tool, to revoke it when that tool is gone
        self.approver = approver
        self._undo = UndoLog(ctx.workspace, undo_file_limit, undo_total_limit)
        self.ctx = dataclasses.replace(ctx, journal=self._undo)
        self.max_steps = max_steps
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
        self.mode = Mode.BUILD
        self.history: list[Message] = []  # no system message; the prompt is rebuilt per request
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

    def set_mode(self, mode: Mode) -> None:
        """Takes effect from the next model request; a step already running keeps its snapshot."""
        self.mode = mode

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
        self.tool_meta = dict(tool_meta or {})
        self.mode = Mode.BUILD
        self.grants = Grants()
        self._granted_extra.clear()
        self._reported = None
        self._save_error = None
        self._save_failing = False
        self._summary_failed = self._warned_full = False
        self._undo.clear()
        return old

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

    def close(self) -> None:
        """Release the recorder (one last attempt to write what is queued). Never raises."""
        close_recorder(self._recorder)

    def estimate(self) -> int:
        """Tokens the next request would use: the model's last count plus a guess for what came after."""
        specs = [t.spec() for t in self.tools_for(self.mode)]
        loaded = self._loaded if self._loaded is not None else self.instructions()
        total = estimate_tokens(self._system(self.mode, loaded), self.history, self._reported, specs)
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
                self.ctx.workspace, self.today(), platform.system(), loaded, skills=skills, paths=guard_of(self.ctx).paths
            )
            return Message("system", prompt)
        todo = any(t.name == "todo" for t in self.tools)
        prompt = build_system_prompt(
            self.ctx.workspace,
            mode,
            self.today(),
            platform.system(),
            instructions=loaded,
            todo=todo,
            skills=skills,
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

    def _save(self, write: Callable[[], None]) -> None:
        try:
            write()
        except Exception as exc:  # SessionWriteError, or a recorder bug: either way the turn goes on
            if not self._save_failing:  # one notice per streak, not one per message
                self._save_failing = True
                self._save_error = f"Could not save the session: {exc}"
        else:
            self._save_failing = False

    def _flush_save_error(self, emit: EventSink) -> None:
        if self._save_error is not None:
            emit(Notice("error", self._save_error))
            self._save_error = None  # only once shown: a sink that raised gets it again at the next flush point

    async def compact(self, emit: EventSink) -> None:
        """Compact now, whatever the estimate (R16): prune, then summarize what is older than the kept turns.

        Esc cancels it like a turn: the history stays valid, and the cancel goes on after a notice.
        """
        if self._busy:
            raise AgentBusy("a turn is already running")
        if not self.history:
            self._undo.clear()
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
            self._undo.clear()  # the turns before it are folded or pruned: undoing them would not match the history
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

        It is not a step of the turn, and its Done is ignored.
        """
        request, _ = summary_input(older, int(window * SUMMARY_BUDGET))
        parts: list[str] = []
        try:
            async with contextlib.aclosing(self.provider.stream(request, [])) as events:
                async for event in events:
                    match event:
                        case TextDelta(text):
                            parts.append(text)
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
        return [t for t in self.all_tools() if not (mode is Mode.PLAN and t.mutating)]

    async def run_turn(self, text: str, emit: EventSink, *, draft: str | None = None) -> None:
        if self._busy:
            raise AgentBusy("a turn is already running")
        if not text.strip():
            return
        self._busy = True
        self._summary_failed = self._warned_full = False  # per turn: one retry of a summary, one warning
        self._undo.begin(len(self.history), draft if draft is not None else text)
        try:
            self._record(Message("user", text))
            try:
                self._flush_save_error(emit)
                if self.role == "main" and (notice := self.instructions_notice()) is not None:
                    try:
                        emit(notice)
                    except BaseException:
                        self.forget_instructions_notice()  # not delivered: the next turn shows it
                        raise
                await self._run_steps(emit)
            except asyncio.CancelledError:
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
            self._undo.finish()
            self._busy = False
            self._inflight = None

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
        self._busy = True
        try:
            # The restore runs in a thread a cancel cannot stop: let it finish and cut the history
            # to match what it did, then hand the cancel on.
            cut_ids = {m.tool_call_id for m in self.history[turn.start :] if m.tool_call_id}
            task = asyncio.ensure_future(asyncio.to_thread(restore_turn, turn, self.ctx.workspace))
            cancel: asyncio.CancelledError | None = None
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as exc:
                    cancel = cancel or exc
                except Exception:
                    break
            error = task.exception()
            result = None if error is not None else task.result()
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
            for notice in undo_notices(result, turn.bash):
                emit(notice)
        finally:
            self._busy = False
        return turn.prompt

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
        for _ in range(self.max_steps):
            await self._maybe_compact(emit)
            mode = self.mode  # snapshot: a Tab press mid-step applies from the next request
            requested = await self._request(mode, emit)
            if requested is None:  # provider error, already reported
                return
            assistant, total = requested
            self._record(assistant)
            self._anchor(total)
            self._flush_save_error(emit)
            if not assistant.tool_calls:
                emit(TurnEnded("done"))
                return
            for tool_call in assistant.tool_calls:
                await self._run_tool_call(tool_call, mode, emit)
        emit(Notice("warn", f"Step limit ({self.max_steps}) reached"))
        emit(TurnEnded("step_limit"))

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
        if (note := self._todo_note()) is not None:
            # Just before the latest user message: the model reads it next to what it is asked now.
            # Trade-off: the note changes the request prefix after it, so a provider's prompt cache
            # is reused only up to here; placing it last would keep the cache but sit apart from the question.
            last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=len(messages))
            messages.insert(last_user, note)
        parts: list[str] = []
        calls: list[ToolCall] = []
        # aclosing so that leaving the loop early (cancel, error) closes the provider stream now
        failure: str | None = None
        total: int | None = None
        try:
            async with contextlib.aclosing(self.provider.stream(messages, specs)) as events:
                async for event in events:
                    match event:
                        case TextDelta(text):
                            parts.append(text)
                            emit(AssistantText(text))
                        case ToolCallEvent(call):
                            calls.append(call)
                        case Done(prompt_tokens, completion_tokens):
                            total = prompt_tokens + completion_tokens
                            emit(UsageUpdate(total, self.provider.context_window))
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

    async def _run_tool_call(self, call: ToolCall, mode: Mode, emit: EventSink) -> None:
        tool = next((t for t in self.all_tools() if t.name == call.name), None)
        self._inflight = call.id
        label = tool.label(call.arguments) if tool else call.name
        emit(ToolStarted(call.id, label, task=tool is not None and tool.name == "task"))
        preview: Preview | None = None
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
            elif mode is Mode.PLAN and tool.mutating:
                result = ToolResult(f"refused: {call.name} is unavailable in plan mode", True, "refused: plan mode")
            elif (error := check_args(tool.parameters, call.arguments)) is not None:
                result = ToolResult(f"{call.name}: {error}", True, "invalid arguments")
            elif (refused := await self._pre_hooks(tool, call, mode, emit)) is not None:
                result = refused
            elif isinstance(gate := await self._confirm_protected(tool, call), ToolResult):
                result = gate
            else:
                # `gate` is True when the user allowed this call to read a protected file: the tool's own check of
                # the file it opens (a symlink may have changed since the path check) lets that one through.
                call_ctx = dataclasses.replace(
                    self.ctx, progress=lambda event: emit(TaskProgress(call.id, event)), protected_ok=gate
                )
                prepared = await tool.prepare(call.arguments, call_ctx)
                if isinstance(prepared, ToolResult):
                    result, ran = prepared, True  # the tool did its work (and failed, or finished) while preparing
                else:
                    preview = guard_of(self.ctx).redact_preview(prepared.preview)
                    result, applied, ran = await self._approve_and_run(tool, call, prepared, mode, emit)
        except _Landed as landed:
            result, applied, cancel = landed.result, landed.applied, landed.cancel  # not `ran`: a cancel is on its way
        except Exception as exc:  # NFR4: a failing tool is an error result, never a crashed turn
            result = ToolResult(f"{call.name} failed: {exc}", True, "failed")
            applied = False
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
                emit(Notice("warn", f"hook PostToolUse failed: {notice_text(str(exc))}"))
        self._record(
            Message("tool", result.output, tool_call_id=call.id, name=call.name), ToolMeta(label, result.summary)
        )
        self._inflight = None
        self._flush_save_error(emit)
        emit(ToolFinished(call.id, result, preview, applied))
        if call.name == "todo" and not result.is_error:
            emit(TodosChanged(self.todos))
        if cancel is not None:
            raise cancel

    def _hook_payload(self, **fields: object) -> dict[str, object]:
        """What every hook gets besides the event's own fields: the session and the workspace."""
        session_id = getattr(self._recorder, "session_id", "")
        return {**fields, "session_id": session_id if isinstance(session_id, str) else "", "cwd": str(self.ctx.workspace)}

    def _redact_hook_text(self, text: str) -> str:
        """Secrets out of text a hook produced, before it is shown. Fails closed: nothing of it when redaction fails.
        Not counted: the same text is counted when it reaches the model."""
        try:
            return guard_of(self.ctx).redactor.redact(text)[0]
        except Exception:
            return ""

    async def _pre_hooks(self, tool: Tool, call: ToolCall, mode: Mode, emit: EventSink) -> ToolResult | None:
        """Run the `PreToolUse` hooks for a call; the refusal when one blocks, else None. Before every dialog."""
        if not self.hooks.has("PreToolUse", call.name):
            return None
        payload = self._hook_payload(tool=call.name, args=call.arguments, mode=mode.value)
        decision = await self.hooks.run("PreToolUse", payload, tool=call.name)
        for notice in self.hooks.notices(decision, self._redact_hook_text):
            emit(notice)
        if not decision.blocked:
            return None
        return ToolResult(f"blocked by hook: {decision.reason}", True, "blocked by hook")

    async def _post_hooks(self, call: ToolCall, result: ToolResult, emit: EventSink) -> ToolResult:
        """Run the `PostToolUse` hooks on a finished call (`result` is redacted and cut already) and append what they
        add, redacted, to the output the model sees: the tool's own output is kept, the hook's text is cut to fit."""
        payload = self._hook_payload(tool=call.name, args=call.arguments, result=result.output, is_error=result.is_error)
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

    def _edits_agent_config(self, tool: Tool, call: ToolCall) -> bool:
        """Does a write or edit target `.agent/settings.json` or `.agent/mcp.json` of the workspace? Those files
        grant the agent its own permissions and processes: no rule, grant or "Always" may approve a change to them."""
        raw = call.arguments.get("path")
        if tool.name not in ("write", "edit") or not isinstance(raw, str):
            return False
        try:
            target = resolve_path(self.ctx, raw)
        except InvalidPath:
            return False
        for protected in agent_config_files(self.ctx.workspace):
            if target == protected or (target.exists() and protected.exists() and target.samefile(protected)):
                return True
        return False

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

    async def _confirm_protected(self, tool: Tool, call: ToolCall) -> ToolResult | bool:
        """Ask before a call reads a protected path, whatever the rules, grants and mode say (deny rules ran before):
        only Allow once or Deny. Returns the refusal; else whether the user allowed a protected read (False: the call
        names no protected file)."""
        target = self._protected_target(tool, call)  # a few stat calls: cheap, like `_edits_agent_config`
        if target is None:
            return False
        request = ApprovalRequest(
            call.name, tool.label(call.arguments), None, False, None, protected=True, protected_path=str(target)
        )
        decision = await self.approver.ask(request)
        if decision.choice in (Choice.ONCE, Choice.ALWAYS, Choice.PROJECT):  # an Always is not offered: it counts as once
            return True
        note = f" Note: {decision.note}" if decision.note else ""
        return ToolResult(f"User denied this call.{note}", True, "denied")

    async def _approve_and_run(
        self, tool: Tool, call: ToolCall, prepared: Prepared, mode: Mode, emit: EventSink
    ) -> tuple[ToolResult, bool, bool]:
        """The result, whether it changed something, and whether the tool ran (False: the user said no)."""
        # Order: deny rules (in `_run_tool_call`), allow rules, this launch's "Always", then the dialog. Neither
        # rules nor grants approve anything in plan mode, nor a change to the agent's own settings files.
        guarded = tool.needs_approval and self._edits_agent_config(tool, call)
        if tool.needs_approval and (
            guarded
            or not (
                (mode is not Mode.PLAN and self.rules.allow(call.name, call.arguments, tool.grant_arg))
                or self.grants.covers(call.name, call.arguments, mode, tool.grant_arg)
            )
        ):
            allow_always = mode is not Mode.PLAN and not guarded
            save_rule = self._offered_rule(tool, call) if allow_always else None
            request = ApprovalRequest(
                call.name, tool.label(call.arguments), guard_of(self.ctx).redact_preview(prepared.preview), allow_always, save_rule
            )
            decision = await self.approver.ask(request)
            if decision.choice not in (Choice.ONCE, Choice.ALWAYS, Choice.PROJECT):  # anything but a clear yes is a no
                note = f" Note: {decision.note}" if decision.note else ""
                return ToolResult(f"User denied this call.{note}", True, "denied"), False, False
            if decision.choice in (Choice.ALWAYS, Choice.PROJECT) and allow_always:
                self.grants.add(call.name, call.arguments, tool.grant_arg)  # for this run, even if saving fails
                if not any(tool is t for t in self.tools):
                    self._granted_extra[call.name] = tool
                if decision.choice is Choice.PROJECT:
                    await self._save_rule(tool, call, emit)
        if tool.name in UNDO_UNTRACKED:
            self._undo.note_bash()
        if not tool.runs_to_completion():
            result = await prepared.execute()  # cancellable at once; bash kills its process group
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
        if cancel is not None:
            raise _Landed(result, applied, cancel)
        return result, applied, True
