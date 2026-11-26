"""The seams the workflow runner uses: `preapproved`, `run_call`, `operation` and `ask_json`."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import cast

import pytest
from src.engine.approval import Choice, Decision
from src.engine.checkpoints.store import CheckpointStore, Snapshot
from src.engine.events import AgentEvent
from src.engine.loop import Agent, AgentBusy
from src.engine.provider.base import Done, Message, ProviderError, TextDelta, ToolCall
from src.engine.tools import default_tools
from src.engine.tools.base import ToolContext

from tests.fakes import MemoryRecorder, RecordingApprover, ScriptedProvider

DENY = Decision(Choice.DENY)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


def make(workspace: Path, script: list | None = None, decisions: list[Decision] | None = None, **kw) -> tuple[Agent, RecordingApprover, MemoryRecorder, ScriptedProvider]:  # type: ignore[no-untyped-def,type-arg]
    approver = RecordingApprover(decisions)
    recorder = MemoryRecorder()
    provider = ScriptedProvider(script or [], context_window=10_000)
    agent = Agent(
        provider,
        default_tools(),
        approver,
        ToolContext(workspace=workspace, bash_timeout=5.0),
        25,
        today=lambda: date(2026, 11, 20),
        recorder=recorder,
        **kw,
    )
    return agent, approver, recorder, provider


def write(call_id: str, path: str, content: str = "x") -> ToolCall:
    return ToolCall(call_id, "write", {"path": path, "content": content})


def sink(_: AgentEvent) -> None:
    return None


async def test_preapproved_call_skips_the_dialog(workspace: Path) -> None:
    agent, approver, _, _ = make(workspace, preapproved=lambda c: c.id == "ok")

    result = await agent.run_call(write("ok", "a.txt"), sink)

    assert not result.is_error
    assert (workspace / "a.txt").read_text() == "x"
    assert approver.requests == []


async def test_a_call_that_is_not_preapproved_still_asks(workspace: Path) -> None:
    agent, approver, _, _ = make(workspace, decisions=[DENY], preapproved=lambda c: c.id == "ok")

    result = await agent.run_call(write("other", "a.txt"), sink)

    assert result.is_error
    assert result.summary == "denied"
    assert len(approver.requests) == 1
    assert not (workspace / "a.txt").exists()


async def test_without_preapproved_the_dialog_is_shown(workspace: Path) -> None:
    agent, approver, _, _ = make(workspace, decisions=[DENY])

    await agent.run_call(write("c", "a.txt"), sink)

    assert len(approver.requests) == 1


async def test_protected_path_ask_survives_preapproval(workspace: Path) -> None:
    (workspace / ".env").write_text("SECRET=1\n")
    agent, approver, _, _ = make(workspace, decisions=[DENY], preapproved=lambda c: True)

    result = await agent.run_call(ToolCall("p", "read", {"path": ".env"}), sink)

    assert result.is_error
    assert [r.protected for r in approver.requests] == [True]


async def test_guarded_settings_write_survives_preapproval(workspace: Path) -> None:
    (workspace / ".agent").mkdir()
    agent, approver, _, _ = make(workspace, decisions=[DENY], preapproved=lambda c: True)

    result = await agent.run_call(write("g", ".agent/settings.json", "{}"), sink)

    assert result.is_error
    assert len(approver.requests) == 1
    assert not (workspace / ".agent" / "settings.json").exists()


# --- operation --------------------------------------------------------------------------------


class SnapStore:
    def __init__(self, commits: list[str]) -> None:
        self.commits = list(commits)
        self.taken = 0

    async def snapshot(self) -> Snapshot | None:
        self.taken += 1
        return Snapshot(self.commits.pop(0), ())

    def take_warning(self) -> str | None:
        return None


async def test_operation_is_one_undo_unit_and_undo_removes_the_files(workspace: Path) -> None:
    agent, _, _, _ = make(workspace, preapproved=lambda c: True)

    async with agent.operation("/workflow run reel", sink):
        await agent.run_call(write("a", "one.txt"), sink)
        await agent.run_call(write("b", "two.txt"), sink)

    assert not agent.busy
    unit = agent.undo_log.peek()
    assert unit is not None
    assert unit.prompt == "/workflow run reel"
    assert await agent.undo(sink) == "/workflow run reel"
    assert not (workspace / "one.txt").exists()
    assert not (workspace / "two.txt").exists()
    assert agent.undo_log.peek() is None


async def test_operation_takes_a_checkpoint_before_and_after(workspace: Path) -> None:
    store = SnapStore(["before", "after"])
    agent, _, _, _ = make(workspace, checkpoints=cast(CheckpointStore, store))

    async with agent.operation("p", sink):
        pass

    turn = agent.undo_log.peek()
    assert store.taken == 2
    assert turn is not None
    assert (turn.before, turn.after) == ("before", "after")


async def test_operation_raises_when_busy_and_clears_busy_after(workspace: Path) -> None:
    agent, _, _, _ = make(workspace)

    async with agent.operation("p", sink):
        assert agent.busy
        with pytest.raises(AgentBusy):
            async with agent.operation("q", sink):
                pytest.fail("entered while busy")
        with pytest.raises(AgentBusy):
            await agent.run_turn("hello", sink)

    assert not agent.busy


async def test_operation_clears_busy_and_finishes_the_unit_when_the_body_raises(workspace: Path) -> None:
    agent, _, _, _ = make(workspace, preapproved=lambda c: True)

    with pytest.raises(asyncio.CancelledError):
        async with agent.operation("p", sink):
            await agent.run_call(write("a", "one.txt"), sink)
            raise asyncio.CancelledError

    assert not agent.busy
    unit = agent.undo_log.peek()
    assert unit is not None
    assert unit.prompt == "p"


async def test_operation_adds_nothing_to_the_history(workspace: Path) -> None:
    agent, _, recorder, _ = make(workspace)

    async with agent.operation("p", sink):
        pass

    assert agent.history == []
    assert recorder.messages == []


# --- ask_json ---------------------------------------------------------------------------------


async def test_ask_json_returns_the_text_and_leaves_the_history_alone(workspace: Path) -> None:
    script = [[TextDelta('{"a":'), TextDelta("1}"), Done(30, 7)]]
    agent, _, recorder, provider = make(workspace, script)
    asked = [Message("system", "json only"), Message("user", "go")]

    reply = await agent.ask_json(asked)

    assert reply == '{"a":1}'
    assert agent.history == []
    assert recorder.messages == []
    assert [m.content for m in provider.requests[0][0]] == ["json only", "go"]
    assert provider.requests[0][1] == []  # no tools


async def test_ask_json_records_usage_as_a_workflow_call(workspace: Path) -> None:
    agent, _, recorder, _ = make(workspace, [[TextDelta("{}"), Done(30, 7)]])

    await agent.ask_json([Message("user", "go")])

    assert [(r.input_tokens, r.output_tokens, r.workflow, r.compaction) for r in recorder.usage_records] == [
        (30, 7, True, False)
    ]
    assert agent.usage.total.workflow_calls == 1


async def test_ask_json_returns_a_provider_error(workspace: Path) -> None:
    agent, _, recorder, _ = make(workspace, [[ProviderError("model is down")]])

    reply = await agent.ask_json([Message("user", "go")])

    assert reply == ProviderError("model is down")
    assert recorder.usage_records == []


# --- the boundaries of preapproval --------------------------------------------------------------


async def test_preapproval_does_not_override_plan_mode(workspace: Path) -> None:
    from src.engine.approval import Mode

    agent, approver, _, _ = make(workspace, preapproved=lambda c: True)
    agent.mode = Mode.PLAN

    result = await agent.run_call(write("p", "a.txt"), sink)

    assert result.is_error
    assert not (workspace / "a.txt").exists()
    assert approver.requests == []


async def test_preapproval_does_not_override_a_deny_rule(workspace: Path) -> None:
    from src.engine.permissions.config import RuleFile
    from src.engine.permissions.rules import parse_rule
    from src.engine.permissions.setup import Rules

    rules = Rules(RuleFile((), (parse_rule("write"),), (), None, None), RuleFile((), (), (), None, None), True)
    agent, approver, _, _ = make(workspace, preapproved=lambda c: True, rules=rules)

    result = await agent.run_call(write("d", "a.txt"), sink)

    assert result.is_error
    assert "denied by rule" in result.output
    assert not (workspace / "a.txt").exists()
    assert approver.requests == []


async def test_only_the_planned_call_ids_are_preapproved(workspace: Path) -> None:
    planned = {"wf:r:one"}
    agent, approver, _, _ = make(workspace, decisions=[DENY], preapproved=lambda c: c.id in planned)

    ok = await agent.run_call(write("wf:r:one", "a.txt"), sink)
    other = await agent.run_call(write("wf:r:one2", "b.txt"), sink)

    assert not ok.is_error
    assert other.is_error and other.summary == "denied"
    assert len(approver.requests) == 1
    assert not (workspace / "b.txt").exists()


async def test_operation_clears_busy_when_the_undo_begin_fails(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent, _, _, _ = make(workspace)

    def boom(*_: object) -> None:
        raise RuntimeError("no undo")

    monkeypatch.setattr(agent.undo_log, "begin", boom)

    with pytest.raises(RuntimeError):
        async with agent.operation("p", sink):
            pytest.fail("entered")

    assert not agent.busy


async def test_an_operation_marked_not_kept_leaves_no_undo_unit(workspace: Path) -> None:
    agent, _, _, _ = make(workspace)

    async with agent.operation("p", sink) as op:
        op.keep = False

    assert agent.undo_log.peek() is None
