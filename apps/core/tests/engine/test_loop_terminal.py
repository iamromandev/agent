"""The terminal tools inside the loop: approval, plan mode, /undo counts and what the session file keeps."""

from __future__ import annotations

from datetime import date
from typing import Any

from core.engine.approval import Choice, Decision
from core.engine.events import AgentEvent, Notice
from core.engine.loop import Agent
from core.engine.provider.base import Done, Message, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.terminal.manager import TerminalManager
from core.engine.tools import default_tools
from core.engine.tools.base import ToolContext
from core.testing.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider

ONCE = Decision(Choice.ONCE)
DENY = Decision(Choice.DENY)


class Events:
    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    def notices(self) -> list[tuple[str, str]]:
        return [(e.level, e.text) for e in self.items if isinstance(e, Notice)]


def call(tool: str, call_id: str = "c1", /, **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(call_id, tool, arguments))


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


def make_agent(
    ctx: ToolContext,
    manager: TerminalManager,
    script: list[list[ProviderEvent | Gate]],
    decisions: list[Decision] | None = None,
    recorder: Any = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script, context_window=1000)
    approver = RecordingApprover(decisions)
    agent = Agent(
        provider,
        default_tools(manager),
        approver,
        ctx,
        25,
        today=lambda: date(2026, 11, 12),
        recorder=recorder or MemoryRecorder(),
    )
    return agent, provider, approver


def tool_messages(agent: Agent) -> list[Message]:
    return [m for m in agent.history if m.role == "tool"]


async def test_undo_counts_term_stop_too(ctx: ToolContext, manager: TerminalManager) -> None:
    manager.start("api", "sleep 30")
    script = [reply(call("term_stop", name="api")), reply(TextDelta("done"))]
    agent, _, _ = make_agent(ctx, manager, script, [ONCE])
    await agent.run_turn("stop", Events())

    events = Events()
    await agent.undo(events)

    assert events.notices()[-1] == ("warn", "That turn ran 1 bash/terminal command; its effects were not undone.")
