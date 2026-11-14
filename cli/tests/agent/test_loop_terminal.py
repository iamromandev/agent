"""The terminal tools inside the loop: approval, plan mode, /undo counts and what the session file keeps."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.agent.approval import Choice, Decision
from src.agent.events import AgentEvent, Notice
from src.agent.loop import Agent
from src.agent.mode import Mode
from src.agent.provider.base import Done, Message, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.agent.session import FileRecorder
from src.agent.session.paths import workspace_dir
from src.agent.terminal.manager import TerminalManager
from src.agent.tools import default_tools
from src.agent.tools.base import ToolContext

from tests.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider
from tests.helpers import wait_until

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


async def test_an_approved_term_start_starts_the_terminal(
    ctx: ToolContext, manager: TerminalManager
) -> None:
    script = [reply(call("term_start", name="api", command="sleep 30")), reply(TextDelta("started"))]
    agent, _, approver = make_agent(ctx, manager, script, [ONCE])

    await agent.run_turn("serve", Events())

    assert manager.names() == ["api"]
    assert approver.requests[0].label == "term_start api"
    assert approver.requests[0].preview is not None and approver.requests[0].preview.body == "sleep 30"
    assert tool_messages(agent)[0].content.startswith("started api (pid ")


async def test_a_denied_term_start_starts_nothing(ctx: ToolContext, manager: TerminalManager) -> None:
    script = [reply(call("term_start", name="api", command="sleep 30")), reply(TextDelta("ok"))]
    agent, _, _ = make_agent(ctx, manager, script, [Decision(Choice.DENY, note="not now")])

    await agent.run_turn("serve", Events())

    assert manager.names() == []
    assert tool_messages(agent)[0].content == "User denied this call. Note: not now"


async def test_a_denied_term_stop_leaves_the_terminal_running(ctx: ToolContext, manager: TerminalManager) -> None:
    terminal = manager.start("api", "sleep 30")
    script = [reply(call("term_stop", name="api")), reply(TextDelta("ok"))]
    agent, _, _ = make_agent(ctx, manager, script, [DENY])

    await agent.run_turn("stop it", Events())

    assert terminal.status == ("running", None)


async def test_term_stop_through_the_loop_stops_the_group(ctx: ToolContext, manager: TerminalManager) -> None:
    terminal = manager.start("api", "sleep 30")
    script = [reply(call("term_stop", name="api")), reply(TextDelta("ok"))]
    agent, _, _ = make_agent(ctx, manager, script, [ONCE])

    await agent.run_turn("stop it", Events())

    assert terminal.status[0] == "exited"
    assert tool_messages(agent)[0].content == "stopped api (exited: SIGTERM)"


async def test_term_read_needs_no_approval(ctx: ToolContext, manager: TerminalManager) -> None:
    terminal = manager.start("api", "echo hello; sleep 30")
    await wait_until(lambda: terminal.buffer.tail(1) == ["hello"])
    script = [reply(call("term_read", name="api")), reply(TextDelta("saw it"))]
    agent, _, approver = make_agent(ctx, manager, script, [])

    await agent.run_turn("look", Events())

    assert approver.requests == []
    assert tool_messages(agent)[0].content == "hello\n\nrunning"


async def test_plan_mode_hides_start_and_stop_but_not_read(ctx: ToolContext, manager: TerminalManager) -> None:
    agent, provider, _ = make_agent(ctx, manager, [reply(TextDelta("planning"))])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("plan", Events())

    names = provider.requests[0][1]
    assert "term_read" in names
    assert "term_start" not in names
    assert "term_stop" not in names


async def test_plan_mode_refuses_a_term_start_the_model_calls_anyway(
    ctx: ToolContext, manager: TerminalManager
) -> None:
    script = [reply(call("term_start", name="api", command="sleep 30")), reply(TextDelta("ok"))]
    agent, _, approver = make_agent(ctx, manager, script, [])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("serve", Events())

    assert manager.names() == []
    assert approver.requests == []
    assert tool_messages(agent)[0].content == "refused: term_start is unavailable in plan mode"


async def test_plan_mode_refuses_a_term_stop_the_model_calls_anyway(
    ctx: ToolContext, manager: TerminalManager
) -> None:
    terminal = manager.start("api", "sleep 30")
    script = [reply(call("term_stop", name="api")), reply(TextDelta("ok"))]
    agent, _, approver = make_agent(ctx, manager, script, [])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("stop", Events())

    assert terminal.status == ("running", None)
    assert approver.requests == []
    assert tool_messages(agent)[0].content == "refused: term_stop is unavailable in plan mode"


async def test_a_second_term_start_with_a_running_name_is_refused_without_asking(
    ctx: ToolContext, manager: TerminalManager
) -> None:
    manager.start("api", "sleep 30")
    script = [reply(call("term_start", name="api", command="sleep 31")), reply(TextDelta("ok"))]
    agent, _, approver = make_agent(ctx, manager, script, [])

    await agent.run_turn("again", Events())

    assert approver.requests == []
    assert tool_messages(agent)[0].content == "terminal api is already running"


async def test_undo_counts_term_start_with_bash(ctx: ToolContext, manager: TerminalManager) -> None:
    script = [
        reply(call("term_start", "c1", name="api", command="sleep 30"), call("bash", "c2", command="true")),
        reply(TextDelta("done")),
    ]
    agent, _, _ = make_agent(ctx, manager, script, [ONCE, ONCE])
    await agent.run_turn("go", Events())

    events = Events()
    await agent.undo(events)

    assert events.notices() == [
        ("info", "Undid the last turn: no files changed"),
        ("warn", "That turn ran 2 bash/terminal commands; their effects were not undone."),
    ]


async def test_undo_counts_term_stop_too(ctx: ToolContext, manager: TerminalManager) -> None:
    manager.start("api", "sleep 30")
    script = [reply(call("term_stop", name="api")), reply(TextDelta("done"))]
    agent, _, _ = make_agent(ctx, manager, script, [ONCE])
    await agent.run_turn("stop", Events())

    events = Events()
    await agent.undo(events)

    assert events.notices()[-1] == ("warn", "That turn ran 1 bash/terminal command; its effects were not undone.")


async def test_undo_does_not_count_term_read_or_a_denied_call(ctx: ToolContext, manager: TerminalManager) -> None:
    manager.start("api", "sleep 30")
    script = [
        reply(call("term_read", "c1", name="api"), call("term_stop", "c2", name="api")),
        reply(TextDelta("done")),
    ]
    agent, _, _ = make_agent(ctx, manager, script, [DENY])
    await agent.run_turn("look", Events())

    events = Events()
    await agent.undo(events)

    assert events.notices() == [("info", "Undid the last turn: no files changed")]


async def test_the_session_file_holds_only_the_tool_messages(
    ctx: ToolContext, manager: TerminalManager, tmp_path: Path, workspace: Path
) -> None:
    terminal = manager.start("api", "printf '\\033[31mred\\033[0m\\nplain\\n'; sleep 30")
    await wait_until(lambda: terminal.buffer.tail(2) == ["red", "plain"])
    recorder = FileRecorder(workspace_dir(tmp_path / "sessions", workspace), workspace)
    script = [reply(call("term_read", name="api")), reply(TextDelta("it printed red"))]
    agent, _, _ = make_agent(ctx, manager, script, [], recorder=recorder)

    await agent.run_turn("what did it print", Events())

    recorder.close()
    assert recorder.path is not None
    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [r["type"] for r in records] == ["session", "title", "message", "message", "message", "message"]
    messages = [r for r in records if r["type"] == "message"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    tool = messages[2]
    assert tool["name"] == "term_read"
    assert tool["content"] == "red\nplain\n\nrunning"
    assert "\\u001b" not in recorder.path.read_text()  # no escape code reached the file


@pytest.mark.parametrize("name", ["term_start", "term_read", "term_stop"])
async def test_the_terminal_tools_have_the_labels_the_transcript_shows(
    ctx: ToolContext, manager: TerminalManager, name: str
) -> None:
    tool = next(t for t in default_tools(manager) if t.name == name)

    assert tool.label({"name": "api", "command": "x"}) == f"{name} api"
