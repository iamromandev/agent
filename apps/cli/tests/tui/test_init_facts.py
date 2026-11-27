"""/init with facts from the project scan in its message (R8, R10, AC7, AC9, AC14)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import src.tui.app as app_module
from core.engine.mode import Mode
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.testing.facts_fixtures import CI_YML, ROOT_MAKEFILE
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, choose, scripted, settled, type_and_send, until

REPLY = [TextDelta("ok"), Done(1, 1)]
PLAN_REFUSAL = "/init needs BUILD mode — press Tab to switch"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _repo(root: Path) -> Path:
    """This repo's shape, as fixtures: root makefile and workflow, cli/ and api/ with a makefile, ui/ with bun."""
    _write(root / "makefile", ROOT_MAKEFILE)
    _write(root / ".github" / "workflows" / "ci.yml", CI_YML)
    for name in ("cli", "api"):
        _write(root / name / "makefile", "check-ci: # Run lint + type checks without fixing anything. For CI.\ntest: # Run the unit test suite\n")
        _write(root / name / "pyproject.toml", f'[project]\nname = "{name}"\n')
    _write(root / "ui" / "package.json", '{"scripts": {"web:test": "bun test"}}')
    _write(root / "ui" / "bun.lock", "")
    return root


def _user_message(app) -> str:
    return scripted(app).requests[0][0][-1].content


async def test_toggling_the_mode_while_the_scan_is_held_refuses_the_plan_turn(
    make_app: AppFactory, monkeypatch, tmp_path: Path
) -> None:
    ws = _repo(tmp_path / "ws")
    app = make_app([REPLY], workspace=ws, context_window=8192)
    async with app.run_test() as pilot:
        await settled(app)
        entered, release = threading.Event(), threading.Event()
        original = app_module.scan

        def held(workspace: Path):
            entered.set()
            release.wait(5)
            return original(workspace)

        monkeypatch.setattr(app_module, "scan", held)
        sending = asyncio.create_task(type_and_send(pilot, "/init"))
        await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
        app.action_toggle_mode()  # the UI still handles keys while the scan runs in its thread
        app.action_toggle_mode()  # build -> edits -> plan
        assert app.agent.mode is Mode.PLAN
        release.set()
        await sending
        await settled(app)
        await pilot.pause()

        assert PLAN_REFUSAL in [e.text for e in app.query(NoticeEntry)][-1]
        assert scripted(app).requests == [] and not app._turn_active


# The check after the turn (R11-R13, AC10, AC12, AC13, AC15)

CHECK = "/init check"


def _agents_md(content: str) -> ToolCallEvent:
    return ToolCallEvent(ToolCall("w1", "write", {"path": "AGENTS.md", "content": content}))


def _check_notices(app) -> list[str]:
    return [e.text[2:] for e in app.query(NoticeEntry) if e.text[2:].startswith(CHECK)]  # [2:] drops the level glyph


async def _run_init(app, pilot, *, approve: bool = True) -> None:
    await type_and_send(pilot, "/init")
    await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
    await choose(pilot, "once" if approve else "deny")
    if not approve:
        await pilot.press("enter")
    await settled(app)
    await pilot.pause()


async def test_all_commands_found_gives_the_info_notice(make_app: AppFactory, tmp_path: Path) -> None:
    ws = _repo(tmp_path / "ws")
    text = "`make test`, `make -C cli check-ci`\n```\nbun run --cwd ui web:test\n```\n"
    app = make_app([[_agents_md(text), Done(1, 1)], REPLY], workspace=ws, context_window=8192)
    async with app.run_test() as pilot:
        await settled(app)
        await _run_init(app, pilot)

        assert _check_notices(app) == ["/init check: all 3 make and script commands were found by the scan"]


async def test_a_file_with_no_commands_gets_no_check_notice(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app([[_agents_md("# P\n\nPlease make sure it works. `uv run pytest`\n"), Done(1, 1)], REPLY], workspace=_repo(tmp_path / "ws"))
    async with app.run_test() as pilot:
        await settled(app)
        await _run_init(app, pilot)

        assert _check_notices(app) == []
        assert not any("finished without changing" in e.text for e in app.query(NoticeEntry))


async def test_a_failing_check_is_one_error_notice_and_the_app_goes_on(
    make_app: AppFactory, monkeypatch, tmp_path: Path
) -> None:
    def boom(workspace: Path):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(app_module, "init_check", boom)
    app = make_app([[_agents_md("`make x`"), Done(1, 1)], REPLY, REPLY], workspace=_repo(tmp_path / "ws"))
    async with app.run_test() as pilot:
        await settled(app)
        await _run_init(app, pilot)

        errors = [e.text[2:] for e in app.query(NoticeEntry) if "/init check failed" in e.text]
        assert errors == ["/init check failed: disk gone"]
        assert not app._turn_active
        await type_and_send(pilot, "next")
        await settled(app)
        assert scripted(app).requests[-1][0][-1].content == "next"
