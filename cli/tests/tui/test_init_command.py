"""The /init command: a normal turn with a built-in prompt, its guards and its closing notice (R11-R14, AC12-AC15)."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from src.agent.mode import Mode
from src.agent.prompt import build_init_prompt
from src.agent.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.agent.session import sessions_root
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, UserEntry

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, choose, reached, scripted, settled, type_and_send, until

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


async def test_init_starts_a_turn_with_the_built_in_prompt(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)
        await pilot.pause()

        expected = build_init_prompt(tmp_path.resolve(), False, [])
        (request,) = scripted(app).requests
        assert request[0][-1].role == "user" and request[0][-1].content == expected
        assert [e.text for e in app.query(UserEntry)] == [expected]  # not "/init"
        (path,) = _sessions()
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["content"] for r in records if r.get("role") == "user"] == [expected]


async def test_init_on_an_existing_file_asks_to_improve_it_with_edit(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("old rules")
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)

        assert "improve it with edit" in _last_user(app)


async def test_init_names_the_parent_files_already_loaded(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("root rules")
    work = tmp_path / "sub"
    work.mkdir()
    app = make_app([REPLY], workspace=work)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)

        assert "The instructions in AGENTS.md are already loaded" in _last_user(app)


async def test_a_denied_write_creates_no_file_and_says_so(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app([[WRITE, Done(1, 1)], REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "deny")
        await pilot.press("enter")  # the note box, empty
        await settled(app)
        await pilot.pause()

        assert not (tmp_path / "AGENTS.md").exists()
        assert DONE in _notices(app)[-1]


async def test_an_allowed_write_creates_the_file_without_the_unchanged_notice(
    make_app: AppFactory, tmp_path: Path
) -> None:
    app = make_app([[WRITE, Done(1, 1)], REPLY, REPLY], context_window=8192)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await settled(app)
        await pilot.pause()

        assert (tmp_path / "AGENTS.md").read_text().startswith("# Project")
        assert not any(DONE in n for n in _notices(app))
        # The set of loaded files changed, so the next turn announces it (R10) and the prompt carries the file.
        await type_and_send(pilot, "next")
        await settled(app)
        await pilot.pause()
        assert "Use make test." in scripted(app).requests[-1][0][0].content
        assert any("Instructions: AGENTS.md" in n for n in _notices(app))


async def test_a_text_only_reply_ends_with_the_unchanged_notice(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)
        await pilot.pause()

        assert sum(DONE in n for n in _notices(app)) == 1


async def test_an_interrupted_init_ends_with_the_unchanged_notice(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, *REPLY]])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await reached(gate)
        await pilot.press("escape")
        await settled(app)
        await pilot.pause()

        assert sum(DONE in n for n in _notices(app)) == 1


async def test_a_normal_turn_has_no_unchanged_notice(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "hello")
        await settled(app)
        await pilot.pause()

        assert not any(DONE in n for n in _notices(app))


async def test_init_is_refused_while_a_turn_runs(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, *REPLY]])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "first")
        await reached(gate)
        await type_and_send(pilot, "/init")
        await pilot.pause()

        assert any("busy — press Esc to interrupt, then /init" in n for n in _notices(app))
        gate.release.set()
        await settled(app)
        assert len(scripted(app).requests) == 1


async def test_init_is_refused_while_a_session_loads(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        app._switching = True
        try:
            await type_and_send(pilot, "/init")
            await pilot.pause()

            assert any("busy — a session is loading" in n for n in _notices(app))
            assert scripted(app).requests == [] and _sessions() == []
        finally:
            app._switching = False


async def test_init_is_refused_in_plan_mode(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.press("tab")
        assert app.agent.mode is Mode.PLAN
        await type_and_send(pilot, "/init")
        await pilot.pause()

        assert PLAN_REFUSAL in _notices(app)[-1]
        assert scripted(app).requests == [] and _sessions() == []
        assert not app._turn_active


async def test_a_model_that_is_not_ready_gives_back_init_not_the_long_prompt(make_app: AppFactory) -> None:
    app = make_app([REPLY], check_result="Model fake is not pulled")
    async with app.run_test() as pilot:
        await settled(app)
        await until(pilot, lambda: bool(app.query_one(StatusBar).error))
        await type_and_send(pilot, "/init")
        await settled(app)
        await pilot.pause()

        assert scripted(app).requests == []
        assert app.query_one(ChatInput).text == "/init"


def _git_with_parent(tmp_path: Path, parent_text: str, work_text: str) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text(parent_text)
    work = tmp_path / "sub"
    work.mkdir()
    (work / "AGENTS.md").write_text(work_text)
    return work


async def test_an_omitted_parent_is_not_called_loaded(make_app: AppFactory, tmp_path: Path) -> None:
    work = _git_with_parent(tmp_path, "root rules", "w" * 5000)  # the workspace file takes the whole budget
    app = make_app([REPLY], workspace=work)
    async with app.run_test() as pilot:
        await settled(app)
        assert app.agent.instructions().files[0].state == "omitted"
        await type_and_send(pilot, "/init")
        await settled(app)

        assert "already loaded" not in _last_user(app)


async def test_a_cut_parent_is_still_called_loaded(make_app: AppFactory, tmp_path: Path) -> None:
    work = _git_with_parent(tmp_path, "r" * 5000, "short")
    app = make_app([REPLY], workspace=work)
    async with app.run_test() as pilot:
        await settled(app)
        assert app.agent.instructions().files[0].state == "cut"
        await type_and_send(pilot, "/init")
        await settled(app)

        assert "The instructions in AGENTS.md are already loaded" in _last_user(app)


async def test_a_parent_outside_a_git_root_is_not_named(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("outer rules")
    work = tmp_path / "sub"
    work.mkdir()
    app = make_app([REPLY], workspace=work)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)

        assert "already loaded" not in _last_user(app)


async def test_tab_while_init_builds_its_prompt_refuses_the_plan_turn(
    make_app: AppFactory, monkeypatch
) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        entered, release = threading.Event(), threading.Event()
        original = app._init_prompt

        def gated() -> str:
            entered.set()
            release.wait(5)
            return original()

        monkeypatch.setattr(app, "_init_prompt", gated)
        # The pilot waits for the handler, which waits for us: send the keys in a task and poll the thread's flag.
        sending = asyncio.create_task(type_and_send(pilot, "/init"))
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
        app.action_toggle_mode()  # what Tab runs; a real key press is queued behind the handler in the test pilot
        assert app.agent.mode is Mode.PLAN
        release.set()
        await sending
        await settled(app)
        await pilot.pause()

        assert PLAN_REFUSAL in _notices(app)[-1]
        assert scripted(app).requests == [] and not app._turn_active and not app._switching


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
        assert scripted(app).requests == [] and not app._turn_active and not app._switching


async def test_esc_before_the_init_worker_runs_gives_back_init(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        app._begin_turn("long prompt", draft="/init", init=True)
        app.interrupt_turn()  # no await in between: the worker has not had its first line yet
        await pilot.pause()

        assert app.query_one(ChatInput).text == "/init"
        assert scripted(app).requests == [] and not app._turn_active
        assert not any(DONE in n for n in _notices(app))


async def test_init_right_after_new_starts_in_the_new_session(make_app: AppFactory) -> None:
    app = make_app([REPLY, REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "hello")
        await settled(app)
        (old,) = _sessions()
        await type_and_send(pilot, "/new")
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)
        await pilot.pause()

        new = [p for p in _sessions() if p != old]
        assert len(new) == 1
        users = [json.loads(line).get("content") for line in new[0].read_text().splitlines()]
        assert any(u and u.startswith("Write the project instructions") for u in users)
        assert "Write the project instructions" not in old.read_text()


async def test_tab_during_an_init_turn_is_pending_and_the_turn_completes(
    make_app: AppFactory, tmp_path: Path
) -> None:
    gate = Gate()
    app = make_app([[gate, *REPLY]])
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await reached(gate)
        await pilot.press("tab")
        assert app.query_one(StatusBar).pending_mode and app.agent.mode is Mode.PLAN
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert not app._turn_active
        assert sum(DONE in n for n in _notices(app)) == 1  # the file is still unchanged
