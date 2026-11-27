"""The /init command: a normal turn with a built-in prompt, its guards and its closing notice (R11-R14, AC12-AC15)."""

from __future__ import annotations

from pathlib import Path

from core.engine.mode import Mode
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import sessions_root
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, scripted, settled, type_and_send

DONE = "/init finished without changing AGENTS.md"
PLAN_REFUSAL = "/init needs BUILD mode — press Tab to switch"
WRITE = ToolCallEvent(ToolCall("w1", "write", {"path": "AGENTS.md", "content": "# Project\n\nUse make test.\n"}))
REPLY = [TextDelta("ok"), Done(1, 1)]


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _sessions() -> list[Path]:
    return list(sessions_root(None).rglob("*.jsonl"))


def _last_user(app) -> str:
    return scripted(app).requests[0][0][-1].content


async def test_init_on_an_existing_file_asks_to_improve_it_with_edit(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("old rules")
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)

        assert "improve it with edit" in _last_user(app)


async def test_init_is_refused_while_a_session_loads(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        with app.host.hold_claim_for_test("switch"):
            await type_and_send(pilot, "/init")
            await pilot.pause()

            assert any("busy — a session is loading" in n for n in _notices(app))
            assert scripted(app).requests == [] and _sessions() == []


async def test_init_is_refused_in_plan_mode(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.press("tab", "tab")
        assert app.agent.mode is Mode.PLAN
        await type_and_send(pilot, "/init")
        await pilot.pause()

        assert PLAN_REFUSAL in _notices(app)[-1]
        assert scripted(app).requests == [] and _sessions() == []
        assert not app.host.turn_active


def _git_with_parent(tmp_path: Path, parent_text: str, work_text: str) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text(parent_text)
    work = tmp_path / "sub"
    work.mkdir()
    (work / "AGENTS.md").write_text(work_text)
    return work


async def test_an_error_building_the_prompt_is_a_notice_and_starts_nothing(
    make_app: AppFactory, monkeypatch
) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)

        def boom() -> str:
            raise OSError("disk gone")

        monkeypatch.setattr(app, "_init_prompt", boom)
        await type_and_send(pilot, "/init")
        await settled(app)
        await pilot.pause()

        assert "Could not start /init: disk gone" in _notices(app)[-1]
        assert scripted(app).requests == [] and not app.host.turn_active and not app.host.switching


async def test_esc_before_the_init_worker_runs_gives_back_init(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test():
        await settled(app)
        app._begin_turn("long prompt", draft="/init", init=True)
        app.interrupt_turn()  # no await in between: the worker has not had its first line yet
        await settled(app)  # the outcome is applied: the box has the text back

        assert app.query_one(ChatInput).text == "/init"
        assert scripted(app).requests == [] and not app.host.turn_active
        assert not any(DONE in n for n in _notices(app))
