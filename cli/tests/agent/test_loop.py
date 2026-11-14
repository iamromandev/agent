from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import src.agent
from src.agent.approval import Choice, Decision
from src.agent.events import (
    AgentEvent,
    AssistantText,
    Notice,
    ToolFinished,
    ToolStarted,
    TurnEnded,
    UsageUpdate,
)
from src.agent.loop import Agent, AgentBusy
from src.agent.mode import Mode
from src.agent.provider.base import Done, Message, ProviderError, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.agent.session import NullRecorder
from src.agent.tools import default_tools
from src.agent.tools.base import Prepared, Tool, ToolContext, ToolResult
from src.agent.tools.text import TOOL_OUTPUT_LIMIT

from tests.fakes import BlockingApprover, Gate, RecordingApprover, ScriptedProvider

TODAY = date(2026, 11, 12)


class Events:
    """An EventSink that keeps what it was given."""

    def __init__(self) -> None:
        self.items: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.items.append(event)

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.items if isinstance(e, kind)]

    @property
    def last(self) -> AgentEvent:
        return self.items[-1]


class BigTool(Tool):
    name: ClassVar[str] = "big"
    description: ClassVar[str] = "Returns 100,000 characters."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        return ToolResult("x" * 100_000, summary="100000 chars")


class BoomTool(Tool):
    name: ClassVar[str] = "boom"
    description: ClassVar[str] = "Always raises."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        raise RuntimeError("disk on fire")


class BoomOnExecuteTool(Tool):
    name: ClassVar[str] = "boom_exec"
    description: ClassVar[str] = "Raises while executing."
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}, "required": []}
    needs_approval: ClassVar[bool] = True

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        async def execute() -> ToolResult:
            raise RuntimeError("late failure")

        return Prepared(None, execute)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("print('hi')\n")
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0)


def make_agent(
    ctx: ToolContext,
    script: list[list[ProviderEvent | Gate]],
    approver: RecordingApprover | None = None,
    max_steps: int = 25,
    extra_tools: list[Tool] | None = None,
) -> tuple[Agent, ScriptedProvider, RecordingApprover]:
    provider = ScriptedProvider(script)
    approver = approver or RecordingApprover()
    agent = Agent(provider, [*default_tools(), *(extra_tools or [])], approver, ctx, max_steps, today=lambda: TODAY)
    return agent, provider, approver


def call(name: str, call_id: str = "c1", **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(call_id, name, arguments))


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


def tool_messages(agent: Agent) -> list[Message]:
    return [m for m in agent.history if m.role == "tool"]


# --- AC6: the basic loop ---------------------------------------------------------------------


async def test_a_tool_call_runs_and_its_result_reaches_the_second_request(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(
        ctx,
        [reply(TextDelta("looking"), call("read", path="a.py")), reply(TextDelta("done"))],
    )
    events = Events()

    await agent.run_turn("what is in a.py?", events)

    assert events.of(ToolStarted) == [ToolStarted("c1", "read a.py")]
    assert [e.delta for e in events.of(AssistantText)] == ["looking", "done"]
    assert events.last == TurnEnded("done")
    second_request, _ = provider.requests[1]
    assert second_request[-1].role == "tool"
    assert "print('hi')" in second_request[-1].content
    assert second_request[-1].tool_call_id == "c1"
    assert second_request[-1].name == "read"


async def test_history_holds_user_assistant_tool_assistant_and_no_system_message(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("looking"), call("read", path="a.py")), reply(TextDelta("done"))])

    await agent.run_turn("q", Events())

    assert [m.role for m in agent.history] == ["user", "assistant", "tool", "assistant"]
    assert agent.history[0].content == "q"
    assert agent.history[1].content == "looking"
    assert agent.history[1].tool_calls == [ToolCall("c1", "read", {"path": "a.py"})]
    assert agent.history[3].content == "done"


async def test_each_request_starts_with_a_fresh_system_prompt(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("ok"))])

    await agent.run_turn("q", Events())

    messages, _ = provider.requests[0]
    assert messages[0].role == "system"
    assert f"Workspace: {ctx.workspace}" in messages[0].content
    assert "Mode: build" in messages[0].content
    assert "Date: 2026-11-12" in messages[0].content
    assert messages[1:] == [Message("user", "q")]


async def test_the_events_of_a_tool_turn_come_in_order(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(call("read", path="a.py")), reply(TextDelta("done"))])
    events = Events()

    await agent.run_turn("q", events)

    kinds = [type(e).__name__ for e in events.items]
    assert kinds == ["UsageUpdate", "ToolStarted", "ToolFinished", "AssistantText", "UsageUpdate", "TurnEnded"]


async def test_several_tool_calls_in_one_reply_run_in_order(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(
        ctx,
        [reply(call("read", "c1", path="a.py"), call("glob", "c2", pattern="*.py")), reply(TextDelta("ok"))],
    )

    await agent.run_turn("q", Events())

    assert [(m.tool_call_id, m.name) for m in tool_messages(agent)] == [("c1", "read"), ("c2", "glob")]


async def test_blank_text_makes_no_request(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("x"))])
    events = Events()

    await agent.run_turn("  \n\t", events)

    assert provider.requests == []
    assert agent.history == []
    assert events.items == []


async def test_a_done_event_updates_usage(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [[TextDelta("hi"), Done(700, 123)]])
    events = Events()

    await agent.run_turn("q", events)

    assert events.of(UsageUpdate) == [UsageUpdate(used=823, window=1000)]


async def test_usage_updates_are_not_marked_estimated_by_default() -> None:
    assert UsageUpdate(used=1, window=2).estimated is False
    assert UsageUpdate(used=1, window=2, estimated=True).estimated is True


# --- AC7: step limit -------------------------------------------------------------------------


async def test_the_turn_stops_after_max_steps_model_calls(ctx: ToolContext) -> None:
    script = [reply(call("read", f"c{i}", path="a.py")) for i in range(10)]
    agent, provider, _ = make_agent(ctx, script, max_steps=3)
    events = Events()

    await agent.run_turn("loop forever", events)

    assert len(provider.requests) == 3
    assert events.of(Notice) == [Notice("warn", "Step limit (3) reached")]
    assert events.last == TurnEnded("step_limit")
    assert len(tool_messages(agent)) == 3  # the last step's calls were still answered


# --- AC10: bad calls -------------------------------------------------------------------------


async def test_an_unknown_tool_is_reported_and_the_model_is_asked_again(ctx: ToolContext) -> None:
    agent, provider, approver = make_agent(ctx, [reply(call("delete_all")), reply(TextDelta("sorry"))])
    events = Events()

    await agent.run_turn("q", events)

    assert tool_messages(agent)[0].content == "Unknown tool: delete_all"
    assert provider.requests[1][0][-1].content == "Unknown tool: delete_all"
    assert events.of(ToolFinished)[0].result.is_error
    assert events.last == TurnEnded("done")
    assert approver.requests == []


async def test_missing_arguments_give_an_argument_error_without_a_dialog(ctx: ToolContext) -> None:
    agent, _, approver = make_agent(
        ctx, [reply(call("edit", path="a.py", new_string="x")), reply(TextDelta("ok"))]
    )

    await agent.run_turn("q", Events())

    assert tool_messages(agent)[0].content == 'edit: missing "old_string"'
    assert approver.requests == []


async def test_arguments_the_provider_could_not_parse_give_an_argument_error(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(
        ctx, [reply(ToolCallEvent(ToolCall("c1", "read", {"__invalid__": "{oops"}))), reply(TextDelta("ok"))]
    )

    await agent.run_turn("q", Events())

    assert tool_messages(agent)[0].content.startswith("read: ")
    assert "valid JSON" in tool_messages(agent)[0].content


# --- NFR4: a failing tool never escapes ------------------------------------------------------


async def test_a_tool_that_raises_in_prepare_becomes_an_error_result(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(call("boom")), reply(TextDelta("ok"))], extra_tools=[BoomTool()])
    events = Events()

    await agent.run_turn("q", events)

    assert tool_messages(agent)[0].content == "boom failed: disk on fire"
    assert events.of(ToolFinished)[0].result.is_error
    assert events.last == TurnEnded("done")


async def test_a_tool_that_raises_in_execute_becomes_an_error_result(ctx: ToolContext) -> None:
    approver = RecordingApprover([Decision(Choice.ONCE)])
    agent, _, _ = make_agent(
        ctx, [reply(call("boom_exec")), reply(TextDelta("ok"))], approver, extra_tools=[BoomOnExecuteTool()]
    )
    events = Events()

    await agent.run_turn("q", events)

    assert tool_messages(agent)[0].content == "boom_exec failed: late failure"
    assert events.of(ToolFinished)[0].applied is False


async def test_an_approver_that_raises_becomes_an_error_result(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(call("bash", command="ls")), reply(TextDelta("ok"))])  # no decisions

    await agent.run_turn("q", Events())

    assert tool_messages(agent)[0].content.startswith("bash failed: ")


# --- AC15: output limit ----------------------------------------------------------------------


async def test_tool_output_is_truncated_before_history_and_the_event(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(call("big")), reply(TextDelta("ok"))], extra_tools=[BigTool()])
    events = Events()

    await agent.run_turn("q", events)

    content = tool_messages(agent)[0].content
    assert len(content) == TOOL_OUTPUT_LIMIT + len("\n[70000 characters truncated]")
    assert content.endswith("[70000 characters truncated]")
    assert provider.requests[1][0][-1].content == content
    assert events.of(ToolFinished)[0].result.output == content


# --- AC17: approvals -------------------------------------------------------------------------


def _edit(new: str = "print('bye')", call_id: str = "c1") -> ToolCallEvent:
    return call("edit", call_id, path="a.py", old_string="print('hi')", new_string=new)


async def test_the_first_edit_asks_and_once_does_not_grant(ctx: ToolContext, workspace: Path) -> None:
    approver = RecordingApprover([Decision(Choice.ONCE), Decision(Choice.ONCE)])
    agent, _, _ = make_agent(
        ctx,
        [reply(_edit("print('one')")), reply(call("edit", "c2", path="a.py", old_string="one", new_string="two")), reply(TextDelta("ok"))],
        approver,
    )
    events = Events()

    await agent.run_turn("q", events)

    assert len(approver.requests) == 2
    request = approver.requests[0]
    assert (request.tool, request.label, request.allow_always) == ("edit", "edit a.py", True)
    assert request.preview is not None
    assert request.preview.kind == "diff"
    assert "+print('one')" in request.preview.body
    assert (workspace / "a.py").read_text() == "print('two')\n"
    assert [e.applied for e in events.of(ToolFinished)] == [True, True]
    assert events.of(ToolFinished)[0].preview == request.preview


async def test_after_always_a_second_edit_does_not_ask(ctx: ToolContext, workspace: Path) -> None:
    approver = RecordingApprover([Decision(Choice.ALWAYS)])
    agent, _, _ = make_agent(
        ctx,
        [reply(_edit("print('one')")), reply(call("edit", "c2", path="a.py", old_string="one", new_string="two")), reply(TextDelta("ok"))],
        approver,
    )

    await agent.run_turn("q", Events())

    assert len(approver.requests) == 1
    assert (workspace / "a.py").read_text() == "print('two')\n"


async def test_always_on_bash_covers_only_the_exact_command(ctx: ToolContext) -> None:
    approver = RecordingApprover([Decision(Choice.ALWAYS), Decision(Choice.ONCE)])
    agent, _, _ = make_agent(
        ctx,
        [
            reply(call("bash", "c1", command="ls")),
            reply(call("bash", "c2", command="ls")),
            reply(call("bash", "c3", command="ls -la")),
            reply(TextDelta("ok")),
        ],
        approver,
    )

    await agent.run_turn("q", Events())

    assert [r.label for r in approver.requests] == ["bash ls", "bash ls -la"]
    assert all("exit 0" in m.content for m in tool_messages(agent))


async def test_deny_gives_the_model_a_result_with_the_note_and_changes_nothing(
    ctx: ToolContext, workspace: Path
) -> None:
    approver = RecordingApprover([Decision(Choice.DENY, "use pathlib")])
    agent, provider, _ = make_agent(ctx, [reply(_edit()), reply(TextDelta("ok"))], approver)
    events = Events()

    await agent.run_turn("q", events)

    content = tool_messages(agent)[0].content
    assert content == "User denied this call. Note: use pathlib"
    assert provider.requests[1][0][-1].content == content
    assert (workspace / "a.py").read_text() == "print('hi')\n"
    finished = events.of(ToolFinished)[0]
    assert finished.applied is False
    assert finished.result.is_error
    assert finished.preview is not None


async def test_deny_without_a_note_has_no_note_text(ctx: ToolContext) -> None:
    approver = RecordingApprover([Decision(Choice.DENY)])
    agent, _, _ = make_agent(ctx, [reply(_edit()), reply(TextDelta("ok"))], approver)

    await agent.run_turn("q", Events())

    assert tool_messages(agent)[0].content == "User denied this call."


async def test_a_fresh_agent_asks_again_after_a_relaunch(ctx: ToolContext, workspace: Path) -> None:
    first_approver = RecordingApprover([Decision(Choice.ALWAYS)])
    first, _, _ = make_agent(ctx, [reply(_edit("print('one')")), reply(TextDelta("ok"))], first_approver)
    await first.run_turn("q", Events())

    second_approver = RecordingApprover([Decision(Choice.ONCE)])
    second, _, _ = make_agent(
        ctx,
        [reply(call("edit", path="a.py", old_string="one", new_string="two")), reply(TextDelta("ok"))],
        second_approver,
    )
    await second.run_turn("q", Events())

    assert len(first_approver.requests) == 1
    assert len(second_approver.requests) == 1


async def test_read_glob_and_grep_never_ask(ctx: ToolContext) -> None:
    agent, _, approver = make_agent(
        ctx,
        [
            reply(call("read", "c1", path="a.py"), call("glob", "c2", pattern="*.py"), call("grep", "c3", pattern="print")),
            reply(TextDelta("ok")),
        ],
    )

    await agent.run_turn("q", Events())

    assert approver.requests == []
    assert [m.name for m in tool_messages(agent)] == ["read", "glob", "grep"]
    assert not any("denied" in m.content for m in tool_messages(agent))


async def test_a_tool_error_without_a_dialog_is_not_asked(ctx: ToolContext) -> None:
    agent, _, approver = make_agent(
        ctx, [reply(call("edit", path="a.py", old_string="nope", new_string="x")), reply(TextDelta("ok"))]
    )

    await agent.run_turn("q", Events())

    assert approver.requests == []
    assert "not found" in tool_messages(agent)[0].content


# --- AC18: modes -----------------------------------------------------------------------------


def test_tools_for_plan_drops_mutating_tools(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [])

    assert [t.name for t in agent.tools_for(Mode.BUILD)] == ["read", "glob", "grep", "write", "edit", "bash"]
    assert [t.name for t in agent.tools_for(Mode.PLAN)] == ["read", "glob", "grep", "bash"]


async def test_plan_mode_offers_no_write_or_edit_and_says_so_in_the_prompt(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("plan"))])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("q", Events())

    messages, tool_names = provider.requests[0]
    assert tool_names == ["read", "glob", "grep", "bash"]
    assert "Mode: plan" in messages[0].content
    assert "PLAN mode" in messages[0].content


async def test_a_forced_edit_in_plan_mode_is_refused_without_the_approver(
    ctx: ToolContext, workspace: Path
) -> None:
    agent, _, approver = make_agent(ctx, [reply(_edit()), reply(TextDelta("ok"))])
    agent.set_mode(Mode.PLAN)
    events = Events()

    await agent.run_turn("q", events)

    assert tool_messages(agent)[0].content == "refused: edit is unavailable in plan mode"
    assert events.of(ToolFinished)[0].result.is_error
    assert approver.requests == []
    assert (workspace / "a.py").read_text() == "print('hi')\n"


async def test_a_forced_write_in_plan_mode_is_refused(ctx: ToolContext, workspace: Path) -> None:
    agent, _, approver = make_agent(ctx, [reply(call("write", path="n.txt", content="x")), reply(TextDelta("ok"))])
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("q", Events())

    assert tool_messages(agent)[0].content == "refused: write is unavailable in plan mode"
    assert approver.requests == []
    assert not (workspace / "n.txt").exists()


async def test_bash_in_plan_mode_asks_without_always(ctx: ToolContext) -> None:
    approver = RecordingApprover([Decision(Choice.ONCE)])
    agent, _, _ = make_agent(ctx, [reply(call("bash", command="ls")), reply(TextDelta("ok"))], approver)
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("q", Events())

    assert [r.allow_always for r in approver.requests] == [False]


async def test_grants_never_cover_bash_in_plan_mode(ctx: ToolContext) -> None:
    approver = RecordingApprover([Decision(Choice.ONCE)])
    agent, _, _ = make_agent(ctx, [reply(call("bash", command="ls")), reply(TextDelta("ok"))], approver)
    agent.grants.add("bash", {"command": "ls"})
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("q", Events())

    assert len(approver.requests) == 1


async def test_an_always_answer_in_plan_mode_grants_nothing(ctx: ToolContext) -> None:
    approver = RecordingApprover([Decision(Choice.ALWAYS)])
    agent, _, _ = make_agent(ctx, [reply(call("bash", command="ls")), reply(TextDelta("ok"))], approver)
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("q", Events())
    agent.set_mode(Mode.BUILD)

    assert not agent.grants.covers("bash", {"command": "ls"}, Mode.BUILD)
    assert "exit 0" in tool_messages(agent)[0].content  # still ran, as a one-off


async def test_a_mode_switch_during_a_turn_applies_from_the_next_request(ctx: ToolContext, workspace: Path) -> None:
    approver = RecordingApprover([Decision(Choice.ONCE)], on_ask=lambda _: agent.set_mode(Mode.PLAN))
    agent, provider, _ = make_agent(ctx, [reply(_edit()), reply(TextDelta("ok"))], approver)

    await agent.run_turn("q", Events())

    first_messages, first_tools = provider.requests[0]
    second_messages, second_tools = provider.requests[1]
    assert "edit" in first_tools
    assert "Mode: build" in first_messages[0].content
    assert second_tools == ["read", "glob", "grep", "bash"]
    assert "Mode: plan" in second_messages[0].content
    assert (workspace / "a.py").read_text() == "print('bye')\n"  # this step still ran as BUILD


async def test_set_mode_changes_the_mode(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [])
    assert agent.mode is Mode.BUILD
    agent.set_mode(Mode.PLAN)
    assert agent.mode is Mode.PLAN


# --- AC5 loop part: provider errors ----------------------------------------------------------


async def test_a_provider_error_gives_a_notice_and_an_error_turn(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [[TextDelta("par"), ProviderError("stream dropped")], reply(TextDelta("fine"))])
    events = Events()

    await agent.run_turn("q", events)

    assert events.of(Notice) == [Notice("error", "stream dropped")]
    assert events.last == TurnEnded("error")
    assert [(m.role, m.content) for m in agent.history] == [("user", "q"), ("assistant", "par")]


async def test_a_provider_error_without_text_adds_no_assistant_message(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [[ProviderError("HTTP 500")]])

    await agent.run_turn("q", Events())

    assert [m.role for m in agent.history] == ["user"]


async def test_the_next_turn_works_after_a_provider_error(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [[ProviderError("HTTP 500")], reply(TextDelta("fine"))])
    await agent.run_turn("q1", Events())
    events = Events()

    await agent.run_turn("q2", events)

    assert events.last == TurnEnded("done")
    assert agent.history[-1].content == "fine"
    assert [m.role for m in provider.requests[1][0][1:]] == ["user", "user"]


async def test_a_provider_error_after_a_tool_step_ends_the_turn(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(call("read", path="a.py")), [ProviderError("boom")]])
    events = Events()

    await agent.run_turn("q", events)

    assert events.last == TurnEnded("error")
    assert [m.role for m in agent.history] == ["user", "assistant", "tool"]


def test_the_package_re_exports_the_agent_mode_and_events() -> None:
    assert src.agent.Agent is Agent
    assert src.agent.Mode is Mode
    assert src.agent.TurnEnded is TurnEnded


# --- N1: an approval answer that is not an allow is a denial ---------------------------------


async def test_an_unexpected_approval_answer_denies_the_call(ctx: ToolContext, workspace: Path) -> None:
    approver = RecordingApprover([Decision(cast(Any, "maybe"))])
    agent, _, _ = make_agent(ctx, [reply(_edit()), reply(TextDelta("ok"))], approver)

    await agent.run_turn("q", Events())

    assert tool_messages(agent)[0].content.startswith("User denied this call.")
    assert (workspace / "a.py").read_text() == "print('hi')\n"


# --- N7: plan mode matches tool names exactly ------------------------------------------------


async def test_plan_mode_does_not_normalise_tool_names(ctx: ToolContext, workspace: Path) -> None:
    edit_args = {"path": "a.py", "old_string": "print('hi')", "new_string": "print('bye')"}
    script = [reply(call("Edit", "c1", **edit_args), call(" edit", "c2", **edit_args)), reply(TextDelta("ok"))]
    agent, _, approver = make_agent(ctx, script)
    agent.set_mode(Mode.PLAN)

    await agent.run_turn("q", Events())

    assert [m.content for m in tool_messages(agent)] == ["Unknown tool: Edit", "Unknown tool:  edit"]
    assert (workspace / "a.py").read_text() == "print('hi')\n"
    assert approver.requests == []


# --- S3: one turn at a time ------------------------------------------------------------------


async def test_run_turn_while_a_turn_runs_raises_busy(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    agent, _, _ = make_agent(ctx, [reply(call("bash", command="ls"))])
    agent.approver = approver
    first = asyncio.create_task(agent.run_turn("one", Events()))
    await approver.asked.wait()

    with pytest.raises(AgentBusy):
        await agent.run_turn("two", Events())

    assert [m.content for m in agent.history if m.role == "user"] == ["one"]
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_reset_while_a_turn_runs_raises_busy(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    agent, _, _ = make_agent(ctx, [reply(call("bash", command="ls"))])
    agent.approver = approver
    first = asyncio.create_task(agent.run_turn("one", Events()))
    await approver.asked.wait()

    with pytest.raises(AgentBusy):
        agent.reset(recorder=NullRecorder())

    assert agent.history  # untouched
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_the_agent_is_not_busy_after_a_cancelled_turn(ctx: ToolContext) -> None:
    approver = BlockingApprover()
    agent, provider, _ = make_agent(ctx, [reply(call("bash", command="ls"))])
    agent.approver = approver
    first = asyncio.create_task(agent.run_turn("one", Events()))
    await approver.asked.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    agent.reset(recorder=NullRecorder())  # no AgentBusy
    provider.script.append(reply(TextDelta("fine")))
    await agent.run_turn("two", Events())

    assert [m.role for m in agent.history] == ["user", "assistant"]


async def test_the_agent_is_not_busy_after_a_finished_turn(ctx: ToolContext) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("a")), reply(TextDelta("b"))])

    await agent.run_turn("one", Events())
    await agent.run_turn("two", Events())

    assert len(provider.requests) == 2
    agent.reset(recorder=NullRecorder())


# --- N4: a surprise never escapes run_turn ---------------------------------------------------


class BreakingProvider:
    """Breaks the provider contract: raises mid-stream instead of yielding ProviderError."""

    model = "broken"
    context_window = 1000

    async def stream(self, messages: Any, tools: Any) -> Any:
        yield TextDelta("half")
        raise RuntimeError("socket exploded")

    async def check(self) -> str | None:
        return None


async def test_a_provider_that_raises_ends_the_turn_with_an_error(ctx: ToolContext) -> None:
    agent = Agent(BreakingProvider(), default_tools(), RecordingApprover(), ctx, 25, today=lambda: TODAY)
    events = Events()

    await agent.run_turn("q", events)

    assert events.of(Notice) == [Notice("error", "Internal error: socket exploded")]
    assert events.last == TurnEnded("error")
    assert len(events.of(TurnEnded)) == 1


async def test_a_turn_works_after_an_internal_error(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    agent, provider, _ = make_agent(ctx, [reply(TextDelta("fine"))])
    real = src.agent.loop.build_system_prompt
    monkeypatch.setattr(src.agent.loop, "build_system_prompt", _raise_once(real))
    events = Events()

    await agent.run_turn("one", events)

    assert events.of(Notice) == [Notice("error", "Internal error: prompt broke")]
    assert events.last == TurnEnded("error")
    again = Events()
    await agent.run_turn("two", again)
    assert again.last == TurnEnded("done")
    assert agent.history[-1] == Message("assistant", "fine")
    assert len(provider.requests) == 1


def _raise_once(real: Any) -> Any:
    state = {"raised": False}

    def build(*args: Any, **kwargs: Any) -> str:
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("prompt broke")
        return real(*args, **kwargs)

    return build


async def test_an_internal_error_between_a_call_and_its_result_leaves_valid_history(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(call("read", "c1", path="a.py"), call("glob", "c2", pattern="*"))])
    seen: list[AgentEvent] = []

    def emit(event: AgentEvent) -> None:
        seen.append(event)
        if isinstance(event, ToolStarted) and event.call_id == "c1":
            raise RuntimeError("ui exploded")

    await agent.run_turn("q", emit)

    assert [(m.role, m.tool_call_id) for m in agent.history] == [
        ("user", None),
        ("assistant", None),
        ("tool", "c1"),
        ("tool", "c2"),
    ]
    assert seen[-1] == TurnEnded("error")


async def test_an_emit_that_always_raises_does_not_escape_run_turn(ctx: ToolContext) -> None:
    agent, _, _ = make_agent(ctx, [reply(TextDelta("hi"))])

    def emit(event: AgentEvent) -> None:
        raise RuntimeError("ui is gone")

    await agent.run_turn("q", emit)  # must not raise
