"""The agent loop: model request -> tool calls -> model request, until the reply has no tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import platform
from collections.abc import Callable, Mapping, Sequence
from datetime import date

from src.agent.approval import ApprovalRequest, Approver, Choice, Grants
from src.agent.compaction import (
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
from src.agent.events import (
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
from src.agent.instructions import Instructions, load_instructions
from src.agent.mode import Mode
from src.agent.prompt import build_system_prompt
from src.agent.provider.base import (
    Done,
    Message,
    Provider,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from src.agent.session import NullRecorder, Recorder, ToolMeta
from src.agent.session.records import compaction_notice
from src.agent.tools.base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from src.agent.tools.text import truncate, truncate_ansi
from src.agent.undo import (
    NOTHING_TO_UNDO,
    SNAPSHOT_FILE_LIMIT,
    SNAPSHOT_TOTAL_LIMIT,
    UndoLog,
    restore_turn,
    undo_notices,
)

INTERRUPTED = "Interrupted by user"
SKIPPED = "Skipped: the turn was interrupted before this call ran"
NOT_COMPLETED = "Not completed: the turn ended with an internal error"
# Marks a partial assistant message that a cancel cut short, so the model does not take it for a
# complete answer on the next turn.
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
    ) -> None:
        self.provider = provider
        self.tools = list(tools)
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
        self.grants = Grants()
        self.tool_meta: dict[str, ToolMeta] = {}  # by tool_call_id: how each tool message is shown and pruned
        self._recorder: Recorder = recorder if recorder is not None else NullRecorder()
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
        self.tool_meta = dict(tool_meta or {})
        self.mode = Mode.BUILD
        self.grants = Grants()
        self._reported = None
        self._save_error = None
        self._save_failing = False
        self._summary_failed = self._warned_full = False
        self._undo.clear()
        return old

    def close(self) -> None:
        """Release the recorder (one last attempt to write what is queued). Never raises."""
        close_recorder(self._recorder)

    def estimate(self) -> int:
        """Tokens the next request would use: the model's last count plus a guess for what came after."""
        specs = [t.spec() for t in self.tools_for(self.mode)]
        loaded = self._loaded if self._loaded is not None else self.instructions()
        return estimate_tokens(self._system(self.mode, loaded), self.history, self._reported, specs)

    def instructions(self) -> Instructions:
        """The AGENTS.md files as they are on disk now, fitted to this model's window; remembered for `estimate`."""
        budget = int(self.instructions_share * self.provider.context_window)
        self._loaded = loaded = load_instructions(self.ctx.workspace, budget)
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
        prompt = build_system_prompt(
            self.ctx.workspace,
            mode,
            self.today(),
            platform.system(),
            instructions=instructions if instructions is not None else self.instructions(),
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

    def tools_for(self, mode: Mode) -> list[Tool]:
        return [t for t in self.tools if not (mode is Mode.PLAN and t.mutating)]

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
                if (notice := self.instructions_notice()) is not None:
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
        tool = next((t for t in self.tools if t.name == call.name), None)
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
        parts: list[str] = []
        calls: list[ToolCall] = []
        # aclosing so that leaving the loop early (cancel, error) closes the provider stream now
        failure: str | None = None
        total: int | None = None
        try:
            async with contextlib.aclosing(self.provider.stream([system, *self.history], specs)) as events:
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
        tool = next((t for t in self.tools if t.name == call.name), None)
        self._inflight = call.id
        label = tool.label(call.arguments) if tool else call.name
        emit(ToolStarted(call.id, label))
        preview: Preview | None = None
        applied = False
        cancel: asyncio.CancelledError | None = None
        try:
            if tool is None:
                result = ToolResult(f"Unknown tool: {call.name}", True, "unknown tool")
            elif mode is Mode.PLAN and tool.mutating:
                result = ToolResult("refused: write/edit are unavailable in plan mode", True, "refused: plan mode")
            elif (error := check_args(tool.parameters, call.arguments)) is not None:
                result = ToolResult(f"{call.name}: {error}", True, "invalid arguments")
            else:
                prepared = await tool.prepare(call.arguments, self.ctx)
                if isinstance(prepared, ToolResult):
                    result = prepared
                else:
                    preview = prepared.preview
                    result, applied = await self._approve_and_run(tool, call, prepared, mode)
        except _Landed as landed:
            result, applied, cancel = landed.result, landed.applied, landed.cancel
        except Exception as exc:  # NFR4: a failing tool is an error result, never a crashed turn
            result = ToolResult(f"{call.name} failed: {exc}", True, "failed")
            applied = False
        result = dataclasses.replace(result, output=truncate(result.output))
        if result.display is not None:
            result = dataclasses.replace(result, display=truncate_ansi(result.display))
        self._record(
            Message("tool", result.output, tool_call_id=call.id, name=call.name), ToolMeta(label, result.summary)
        )
        self._inflight = None
        self._flush_save_error(emit)
        emit(ToolFinished(call.id, result, preview, applied))
        if cancel is not None:
            raise cancel

    async def _approve_and_run(
        self, tool: Tool, call: ToolCall, prepared: Prepared, mode: Mode
    ) -> tuple[ToolResult, bool]:
        if tool.needs_approval and not self.grants.covers(call.name, call.arguments, mode):
            allow_always = mode is not Mode.PLAN
            request = ApprovalRequest(call.name, tool.label(call.arguments), prepared.preview, allow_always)
            decision = await self.approver.ask(request)
            if decision.choice not in (Choice.ONCE, Choice.ALWAYS):  # anything but a clear yes is a no
                note = f" Note: {decision.note}" if decision.note else ""
                return ToolResult(f"User denied this call.{note}", True, "denied"), False
            if decision.choice is Choice.ALWAYS and allow_always:
                self.grants.add(call.name, call.arguments)
        if tool.name == "bash":
            self._undo.note_bash()
        if not tool.mutating:
            result = await prepared.execute()  # cancellable at once; bash kills its process group
            return result, not result.is_error
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
        return result, applied
