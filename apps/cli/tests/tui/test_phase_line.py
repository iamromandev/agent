"""The status bar during a real turn: the phase line replaces `● working`, then goes away."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.testing.fakes import Gate
from src.tui.widgets.status import StatusBar
from textual.pilot import Pilot

from tests.tui.conftest import AppFactory, choose, dialog, reached, settled, type_and_send, until


def record_lines(status: StatusBar, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every line the bar drew, in order."""
    drawn: list[str] = []
    original = status.update

    def update(content: str = "", *, layout: bool = True) -> None:
        drawn.append(status.line)
        original(content, layout=layout)

    monkeypatch.setattr(status, "update", update)
    return drawn


def activities(lines: list[str]) -> list[str]:
    """The distinct activity parts (timer cut off), in order of appearance."""
    seen: list[str] = []
    for line in lines:
        part = next((p for p in line.split(" · ") if p.startswith(("●", "◐"))), "")
        step = line.split(" · step ")[-1].split(" · ")[0] if " · step " in line else ""
        text = f"{part} [{step}]" if step else part
        if part == "● working":  # before the first request and between TurnEnded and the end of the worker
            continue
        if part and (not seen or seen[-1] != text):
            seen.append(text)
    return seen


def call(name: str, **args: object) -> list:
    return [ToolCallEvent(ToolCall("c1", name, dict(args))), Done(10, 5)]


async def test_phases_follow_a_tool_call_then_text(make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.txt").write_text("a")
    app = make_app([call("read", path="a.txt"), [TextDelta("ok"), Done(1, 1)]], max_steps=5)
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        drawn = record_lines(status, monkeypatch)
        await type_and_send(pilot, "go")
        await settled(app)
        await until(pilot, lambda: not status.busy)

    steps = [a for a in activities(drawn) if a.startswith("●")]
    assert steps == [
        "● waiting for model [1/5]",
        "● running read [1/5]",
        "● waiting for model [2/5]",
        "● writing reply [2/5]",
    ]


async def test_an_approval_dialog_shows_awaiting_approval_then_running(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app([call("write", path="o.txt", content="x"), [TextDelta("ok"), Done(1, 1)]])
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "go")
        await dialog(pilot)
        assert "● awaiting approval: write · step 1/5" in status.line
        await choose(pilot, "once")
        await until(pilot, lambda: not status.busy)


async def test_the_dot_is_gone_after_the_turn(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("ok"), Done(1, 1)]])
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "go")
        await settled(app)
        await until(pilot, lambda: not status.busy)
        assert "●" not in status.line


async def test_escape_clears_the_phase_line(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[TextDelta("part"), gate]])
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "go")
        await reached(gate)
        await until(pilot, lambda: "● writing reply · step 1/5" in status.line)
        await pilot.press("escape")
        await until(pilot, lambda: not status.busy)
        assert "●" not in status.line


async def test_a_provider_error_clears_the_phase_line(make_app: AppFactory) -> None:
    app = make_app([])  # script exhausted: the provider answers with an error
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "go")
        await settled(app)
        await until(pilot, lambda: not status.busy)
        assert "●" not in status.line


async def test_the_step_limit_clears_the_phase_line(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    app = make_app([call("read", path="a.txt")], max_steps=1)
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "go")
        await settled(app)
        await until(pilot, lambda: not status.busy)
        assert "●" not in status.line


async def test_the_reply_token_count_shows_the_models_own_number(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[TextDelta("hello there"), Done(5, 137), gate]])
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await type_and_send(pilot, "go")
        await reached(gate)
        await until(pilot, lambda: status._tick() is None and "137 tok" in status.line)
        await pilot.press("escape")
        await until(pilot, lambda: not status.busy)


async def test_a_task_child_does_not_change_the_phase_line(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.txt").write_text("hello\n")
    task = [ToolCallEvent(ToolCall("t1", "task", {"description": "d", "prompt": "p"})), Done(10, 5)]
    child_read = [ToolCallEvent(ToolCall("r1", "read", {"path": "a.txt"})), Done(300, 20)]
    app = make_app([task, child_read, [TextDelta("REPORT"), Done(4, 3)], [TextDelta("ok"), Done(1, 1)]], task=True)
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        drawn = record_lines(status, monkeypatch)
        await type_and_send(pilot, "go")
        await settled(app)
        await until(pilot, lambda: not status.busy)

    assert activities(drawn)[:3] == [
        "● waiting for model [1/5]",
        "● running task [1/5]",
        "● waiting for model [2/5]",
    ]


async def test_compacting_wins_then_the_phase_line_returns(make_app: AppFactory) -> None:
    from core.engine.events import Compacting, Phase

    app = make_app([[TextDelta("ok"), Done(1, 1)]])
    async with app.run_test():
        status = app.query_one(StatusBar)
        status.busy = True
        await app._on_agent_event(Phase("waiting", 1, 5))
        await app._on_agent_event(Compacting(True))
        assert "◐ compacting…" in status.line and "●" not in status.line
        await app._on_agent_event(Compacting(False))
        assert "● waiting for model · step 1/5" in status.line
        status.busy = False


async def test_a_bare_compact_keeps_the_compacting_line(make_app: AppFactory) -> None:
    app = make_app([])
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        status.busy = True
        status.compacting = True
        assert "◐ compacting…" in status.line
        assert "step" not in status.line
        await _noop(pilot)
        status.compacting = False
        status.busy = False


async def _noop(pilot: Pilot) -> None:
    await pilot.pause()
