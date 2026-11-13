"""Undo end to end: a real FileRecorder, scripted turns, then the saved file read back."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from src.agent.approval import Choice, Decision
from src.agent.events import AgentEvent
from src.agent.loop import Agent
from src.agent.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.agent.session import FileRecorder, SessionStore, sessions_root, workspace_dir
from src.agent.tools import default_tools
from src.agent.tools.base import ToolContext

from tests.fakes import Gate, RecordingApprover, ScriptedProvider

ONCE = Decision(Choice.ONCE)


@pytest.fixture(autouse=True)
def _posix_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")


def call(name: str, call_id: str, **arguments: Any) -> ToolCallEvent:
    return ToolCallEvent(ToolCall(call_id, name, arguments))


def reply(*events: ProviderEvent | Gate) -> list[ProviderEvent | Gate]:
    return [*events, Done(10, 5)]


def sink(_: AgentEvent) -> None:
    return None


def build(tmp_path: Path, script: list[list[ProviderEvent | Gate]], approvals: int) -> tuple[Agent, FileRecorder, SessionStore]:
    root = tmp_path / "ws"
    root.mkdir()
    root = root.resolve()
    recorder = FileRecorder(workspace_dir(sessions_root(None), root), root)
    agent = Agent(
        ScriptedProvider(script, context_window=1000),
        default_tools(),
        RecordingApprover([ONCE] * approvals),
        ToolContext(root, 5.0),
        25,
        today=lambda: date(2026, 11, 12),
        recorder=recorder,
        compact_keep_turns=1,
    )
    return agent, recorder, SessionStore(sessions_root(None), root)


def reload(recorder: FileRecorder, store: SessionStore):
    assert recorder.path is not None
    loaded = store.load(recorder.path.stem)
    assert loaded.skipped_lines == 0
    return loaded


def user_items(loaded) -> list[str]:
    return [i.text for i in loaded.items if i.kind == "user"]


async def test_two_undos_leave_a_saved_session_equal_to_the_live_history(tmp_path: Path) -> None:
    script = [
        reply(call("write", "w1", path="a.txt", content="one")),
        reply(TextDelta("done 1")),
        reply(call("write", "w2", path="b.txt", content="two")),
        reply(TextDelta("done 2")),
        reply(call("bash", "b1", command="echo hi")),
        reply(TextDelta("done 3")),
    ]
    agent, recorder, store = build(tmp_path, script, approvals=3)
    await agent.run_turn("first", sink)
    await agent.run_turn("second", sink)
    await agent.run_turn("third", sink)

    assert await agent.undo(sink) == "third"
    assert await agent.undo(sink) == "second"

    loaded = reload(recorder, store)
    assert loaded.history == agent.history
    assert loaded.tool_meta == agent.tool_meta
    assert user_items(loaded) == ["first"]
    assert [m.content for m in agent.history if m.role == "user"] == ["first"]
    assert not (tmp_path / "ws" / "b.txt").exists()
    assert (tmp_path / "ws" / "a.txt").exists()


async def test_undo_after_a_compaction_still_reads_back_equal(tmp_path: Path) -> None:
    script = [
        reply(TextDelta("1")),
        reply(TextDelta("2")),
        reply(TextDelta("3")),
        reply(TextDelta("summary")),
        reply(call("write", "w1", path="c.txt", content="x")),
        reply(TextDelta("done 4")),
    ]
    agent, recorder, store = build(tmp_path, script, approvals=1)
    for text in ("one", "two", "three"):
        await agent.run_turn(text, sink)
    await agent.compact(sink)

    assert await agent.undo(sink) is None  # right after the compaction: nothing to undo
    assert reload(recorder, store).history == agent.history

    await agent.run_turn("four", sink)
    assert await agent.undo(sink) == "four"

    loaded = reload(recorder, store)
    assert loaded.history == agent.history
    assert loaded.tool_meta == agent.tool_meta
    assert "four" not in user_items(loaded)
    assert any(i.kind == "notice" and i.text.startswith("Context compacted") for i in loaded.items)
    assert not (tmp_path / "ws" / "c.txt").exists()


async def test_a_reused_call_id_keeps_its_meta_live_and_after_reload(tmp_path: Path) -> None:
    script = [
        reply(call("write", "same", path="a.txt", content="1")),
        reply(TextDelta("a")),
        reply(call("write", "same", path="b.txt", content="2")),
        reply(TextDelta("b")),
    ]
    agent, recorder, store = build(tmp_path, script, approvals=2)
    await agent.run_turn("one", sink)
    await agent.run_turn("two", sink)
    await agent.undo(sink)

    assert "same" in agent.tool_meta
    assert "same" in reload(recorder, store).tool_meta
