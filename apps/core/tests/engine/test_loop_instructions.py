"""Instructions in the loop: the budget, re-reading per request, ctx, compaction, the notice (R4, R8, R10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.engine.events import AgentEvent, Notice
from core.engine.loop import Agent
from core.engine.provider.base import Done, TextDelta
from core.engine.tools.base import ToolContext
from core.testing.fakes import MemoryRecorder, RecordingApprover, ScriptedProvider


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


@pytest.mark.parametrize("share", [1])
def test_a_share_outside_the_range_is_rejected(ws: Path, share: float) -> None:
    with pytest.raises(ValueError, match="instructions_share"):
        _agent(ws, instructions_share=share)


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
