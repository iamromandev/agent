"""Instructions in the loop: the budget, re-reading per request, ctx, compaction, the notice (R4, R8, R10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.agent.compaction import SUMMARY_PREFIX
from src.agent.events import AgentEvent, Notice
from src.agent.loop import Agent
from src.agent.provider.base import Done, Message, TextDelta
from src.agent.tools.base import ToolContext

from tests.fakes import MemoryRecorder, RecordingApprover, ScriptedProvider


class Sink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)

    @property
    def notices(self) -> list[Notice]:
        return [e for e in self.events if isinstance(e, Notice) and e.text.startswith("Instructions:")]


def _agent(
    workspace: Path,
    script: int = 3,
    window: int = 1000,
    instructions_share: float = 0.25,
    compact_keep_turns: int = 2,
) -> tuple[Agent, ScriptedProvider]:
    provider = ScriptedProvider([[TextDelta("ok"), Done(1, 1)] for _ in range(script)], context_window=window)
    agent = Agent(
        provider,
        [],
        RecordingApprover(),
        ToolContext(workspace, 5.0),
        5,
        recorder=MemoryRecorder(),
        compact_keep_turns=compact_keep_turns,
        instructions_share=instructions_share,
    )
    return agent, provider


def _system(provider: ScriptedProvider, index: int = -1) -> str:
    return provider.requests[index][0][0].content


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "cli").mkdir()
    return root / "cli"


async def test_parent_and_workspace_files_reach_the_model(ws: Path) -> None:
    (ws.parent / "AGENTS.md").write_text("root rules")
    (ws / "AGENTS.md").write_text("cli rules")
    agent, provider = _agent(ws)
    await agent.run_turn("hi", Sink())
    assert "## AGENTS.md\nroot rules" in _system(provider)
    assert "## cli/AGENTS.md\ncli rules" in _system(provider)


async def test_an_edit_of_a_parent_file_applies_to_the_next_request(ws: Path) -> None:
    root_file = ws.parent / "AGENTS.md"
    root_file.write_text("old rule")
    agent, provider = _agent(ws)
    await agent.run_turn("one", Sink())
    root_file.write_text("new rule")
    await agent.run_turn("two", Sink())
    assert "old rule" in _system(provider, 0)
    assert "new rule" in _system(provider, 1) and "old rule" not in _system(provider, 1)


async def test_a_deleted_parent_file_leaves_no_section_and_the_turn_completes(ws: Path) -> None:
    root_file = ws.parent / "AGENTS.md"
    root_file.write_text("gone soon")
    agent, provider = _agent(ws)
    await agent.run_turn("one", Sink())
    root_file.unlink()
    await agent.run_turn("two", Sink())
    assert "gone soon" not in _system(provider, 1)
    assert agent.history[-1] == Message("assistant", "ok")


async def test_the_budget_is_the_share_of_the_window(ws: Path) -> None:
    (ws / "AGENTS.md").write_text("x" * 5000)
    agent, _ = _agent(ws, window=2048, instructions_share=0.25)
    loaded = agent.instructions()
    assert loaded.budget == 512
    assert len(loaded.text) <= 2048
    assert loaded.files[0].state == "cut"


@pytest.mark.parametrize("share", [0, -0.1, 0.51, 1])
def test_a_share_outside_the_range_is_rejected(ws: Path, share: float) -> None:
    with pytest.raises(ValueError, match="instructions_share"):
        _agent(ws, instructions_share=share)


def test_the_share_of_one_half_is_accepted(ws: Path) -> None:
    _agent(ws, instructions_share=0.5)


def test_ctx_counts_the_instructions(ws: Path) -> None:
    (ws / "AGENTS.md").write_text("w" * 3000)
    agent, _ = _agent(ws, window=8192)
    bare, _ = _agent(ws.parent)  # same workspace tree, no AGENTS.md at this level
    assert agent.estimate() >= 750
    assert agent.estimate() > bare.estimate()


async def test_compaction_summarizes_the_history_but_never_the_instructions(ws: Path) -> None:
    (ws.parent / "AGENTS.md").write_text("root rule " * 50)
    (ws / "AGENTS.md").write_text("keep me " * 100)
    agent, provider = _agent(ws, script=8, window=8192, compact_keep_turns=1)
    for n in range(3):
        await agent.run_turn(f"turn {n}", Sink())
    before = len(agent.history)
    requests_before = len(provider.requests)
    await agent.compact(Sink())

    assert len(provider.requests) == requests_before + 1  # the summary call
    assert agent.history[0].content.startswith(SUMMARY_PREFIX)
    assert len(agent.history) < before
    summary_request = " ".join(m.content for m in provider.requests[-1][0])
    assert "keep me" not in summary_request and "root rule" not in summary_request

    await agent.run_turn("after", Sink())
    system = _system(provider)
    assert "## AGENTS.md\n" + "root rule " * 50 in system
    assert "## cli/AGENTS.md\n" + "keep me " * 100 in system


async def test_the_notice_comes_at_the_start_of_the_first_turn_and_not_again(ws: Path) -> None:
    (ws / "AGENTS.md").write_text("rules")
    agent, _ = _agent(ws)
    first, second = Sink(), Sink()
    await agent.run_turn("one", first)
    await agent.run_turn("two", second)
    assert [n.level for n in first.notices] == ["info"]
    assert first.notices[0].text.startswith("Instructions: cli/AGENTS.md (~")
    assert second.notices == []


async def test_a_notice_repeats_when_a_file_appears(ws: Path) -> None:
    agent, _ = _agent(ws)
    first, second, third = Sink(), Sink(), Sink()
    await agent.run_turn("one", first)
    (ws / "AGENTS.md").write_text("new")
    await agent.run_turn("two", second)
    await agent.run_turn("three", third)
    assert first.notices == [] and third.notices == []
    assert len(second.notices) == 1


async def test_a_notice_repeats_when_a_file_becomes_cut(ws: Path) -> None:
    (ws / "AGENTS.md").write_text("small")
    agent, _ = _agent(ws, script=2)
    await agent.run_turn("one", Sink())
    (ws / "AGENTS.md").write_text("x" * 5000)
    again = Sink()
    await agent.run_turn("two", again)
    assert [n.level for n in again.notices] == ["warn"]


def test_instructions_notice_stores_the_key_and_is_none_without_files(ws: Path) -> None:
    agent, _ = _agent(ws)
    assert agent.instructions_notice() is None
    (ws / "AGENTS.md").write_text("rules")
    assert agent.instructions_notice() is not None
    assert agent.instructions_notice() is None


async def test_the_request_path_loads_fresh_and_estimate_reuses_the_loaded_files(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.agent.loop as loop

    (ws / "AGENTS.md").write_text("first")
    calls: list[int] = []
    real = loop.load_instructions

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "load_instructions", counting)
    agent, provider = _agent(ws, script=2)
    await agent.run_turn("one", Sink())
    after_turn = len(calls)
    for _ in range(10):
        agent.estimate()
    assert len(calls) == after_turn  # no file read per estimate

    (ws / "AGENTS.md").write_text("second")
    await agent.run_turn("two", Sink())
    assert len(calls) > after_turn  # each request reloads
    assert "second" in _system(provider) and "first" not in _system(provider)


def test_estimate_loads_once_when_nothing_was_loaded(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.loop as loop

    calls: list[int] = []
    real = loop.load_instructions
    monkeypatch.setattr(loop, "load_instructions", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    agent, _ = _agent(ws)
    for _ in range(5):
        agent.estimate()
    assert len(calls) == 1


async def test_a_notice_that_could_not_be_delivered_is_announced_again(ws: Path) -> None:
    (ws / "AGENTS.md").write_text("rules")
    agent, _ = _agent(ws, script=3)

    def failing(event: AgentEvent) -> None:
        if isinstance(event, Notice) and event.text.startswith("Instructions:"):
            raise RuntimeError("sink broke")

    await agent.run_turn("one", failing)
    again = Sink()
    await agent.run_turn("two", again)
    assert len(again.notices) == 1


def test_forgetting_the_notice_makes_the_next_one_show_again(ws: Path) -> None:
    (ws / "AGENTS.md").write_text("rules")
    agent, _ = _agent(ws)
    assert agent.instructions_notice() is not None
    assert agent.instructions_notice() is None
    agent.forget_instructions_notice()
    assert agent.instructions_notice() is not None
