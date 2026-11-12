"""The agent loop: model request -> tool calls -> model request, until the reply has no tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import platform
from collections.abc import Callable, Sequence
from datetime import date

from src.agent.approval import ApprovalRequest, Approver, Choice, Grants
from src.agent.events import (
    AssistantText,
    EventSink,
    Notice,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)
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
from src.agent.tools.base import Prepared, Preview, Tool, ToolContext, ToolResult, check_args
from src.agent.tools.text import truncate

INTERRUPTED = "Interrupted by user"
SKIPPED = "Skipped: the turn was interrupted before this call ran"
NOT_COMPLETED = "Not completed: the turn ended with an internal error"
# Marks a partial assistant message that a cancel cut short, so the model does not take it for a
# complete answer on the next turn.
PARTIAL_MARK = "\n[interrupted by user]"


class AgentBusy(RuntimeError):
    """A turn is running; wait for it (or cancel it) before starting another or clearing."""


class _Landed(Exception):
    """A cancel arrived while a mutating call ran. The call ran to the end; this carries its result."""

    def __init__(self, result: ToolResult, applied: bool, cancel: asyncio.CancelledError) -> None:
        super().__init__("cancelled after the call finished")
        self.result = result
        self.applied = applied
        self.cancel = cancel


class Agent:
    def __init__(
        self,
        provider: Provider,
        tools: Sequence[Tool],
        approver: Approver,
        ctx: ToolContext,
        max_steps: int,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.provider = provider
        self.tools = list(tools)
        self.approver = approver
        self.ctx = ctx
        self.max_steps = max_steps
        self.today = today
        self.mode = Mode.BUILD
        self.history: list[Message] = []  # no system message; the prompt is rebuilt per request
        self.grants = Grants()
        self._busy = False
        self._inflight: str | None = None  # id of the tool call being run, for the interrupt repair

    def set_mode(self, mode: Mode) -> None:
        """Takes effect from the next model request; a step already running keeps its snapshot."""
        self.mode = mode

    @property
    def busy(self) -> bool:
        return self._busy

    def clear(self) -> None:
        if self._busy:
            raise AgentBusy("cannot clear while a turn is running")
        self.history = []

    def tools_for(self, mode: Mode) -> list[Tool]:
        return [t for t in self.tools if not (mode is Mode.PLAN and t.mutating)]

    async def run_turn(self, text: str, emit: EventSink) -> None:
        if self._busy:
            raise AgentBusy("a turn is already running")
        if not text.strip():
            return
        self._busy = True
        try:
            self.history.append(Message("user", text))
            try:
                await self._run_steps(emit)
            except asyncio.CancelledError:
                # Esc cancels the worker running this coroutine. Leave a history the model can be
                # asked again with: every tool call answered, the UI told, then let the cancel go on.
                self._answer_unanswered_calls(INTERRUPTED, SKIPPED)
                emit(TurnEnded("interrupted"))
                raise
            except Exception as exc:  # a provider breaking its contract, a bug, a failing sink
                self._answer_unanswered_calls(NOT_COMPLETED, NOT_COMPLETED)
                with contextlib.suppress(Exception):  # the sink may be what broke
                    emit(Notice("error", f"Internal error: {exc}"))
                with contextlib.suppress(Exception):
                    emit(TurnEnded("error"))
        finally:
            self._busy = False
            self._inflight = None

    def _answer_unanswered_calls(self, in_flight_text: str, skipped_text: str) -> None:
        """Give each tool call of the last assistant message that has no result one."""
        last = next((i for i in range(len(self.history) - 1, -1, -1) if self.history[i].role == "assistant"), None)
        if last is None:
            return
        answered = {m.tool_call_id for m in self.history[last + 1 :] if m.role == "tool"}
        for call in self.history[last].tool_calls:
            if call.id not in answered:
                text = in_flight_text if call.id == self._inflight else skipped_text
                self.history.append(Message("tool", text, tool_call_id=call.id, name=call.name))

    async def _run_steps(self, emit: EventSink) -> None:
        for _ in range(self.max_steps):
            mode = self.mode  # snapshot: a Tab press mid-step applies from the next request
            assistant = await self._request(mode, emit)
            if assistant is None:  # provider error, already reported
                return
            self.history.append(assistant)
            if not assistant.tool_calls:
                emit(TurnEnded("done"))
                return
            for tool_call in assistant.tool_calls:
                await self._run_tool_call(tool_call, mode, emit)
        emit(Notice("warn", f"Step limit ({self.max_steps}) reached"))
        emit(TurnEnded("step_limit"))

    async def _request(self, mode: Mode, emit: EventSink) -> Message | None:
        """One model call. Returns the assistant message, or None after a ProviderError."""
        system = Message("system", build_system_prompt(self.ctx.workspace, mode, self.today(), platform.system()))
        specs = [t.spec() for t in self.tools_for(mode)]
        parts: list[str] = []
        calls: list[ToolCall] = []
        # aclosing so that leaving the loop early (cancel, error) closes the provider stream now
        failure: str | None = None
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
                            emit(UsageUpdate(prompt_tokens + completion_tokens, self.provider.context_window))
                        case ProviderError(message):
                            failure = message
                            break  # report after the stream is closed: a cancel in aclose() must not repeat it
        except asyncio.CancelledError:
            # Cancelled mid-stream (the stream is closed by now). Keep the text the user already
            # saw, marked as cut short, as the assistant message. Tool calls collected so far are
            # dropped: none has run, and keeping them would need a fake result for each, for a
            # request the user abandoned. A cancel before any text leaves only the user message.
            if parts:
                self.history.append(Message("assistant", "".join(parts) + PARTIAL_MARK))
            raise
        if failure is not None:
            if parts:
                self.history.append(Message("assistant", "".join(parts)))
            emit(Notice("error", failure))
            emit(TurnEnded("error"))
            return None
        return Message("assistant", "".join(parts), calls)

    async def _run_tool_call(self, call: ToolCall, mode: Mode, emit: EventSink) -> None:
        tool = next((t for t in self.tools if t.name == call.name), None)
        self._inflight = call.id
        emit(ToolStarted(call.id, tool.label(call.arguments) if tool else call.name))
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
        self.history.append(Message("tool", result.output, tool_call_id=call.id, name=call.name))
        self._inflight = None
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
