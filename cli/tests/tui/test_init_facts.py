"""/init with facts from the project scan in its message (R8, R10, AC7, AC9, AC14)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import src.tui.app as app_module
from src.agent.facts import INTRO
from src.agent.mode import Mode
from src.agent.prompt import build_init_prompt
from src.agent.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.tui.screens.approval import ApprovalScreen
from src.tui.widgets.transcript import NoticeEntry, UserEntry

from tests.agent.facts_fixtures import CI_YML, ROOT_MAKEFILE
from tests.fakes import Gate
from tests.tui.conftest import AppFactory, choose, reached, scripted, settled, type_and_send, until

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


async def test_init_message_carries_the_facts_of_a_repo_shaped_project(make_app: AppFactory, tmp_path: Path) -> None:
    ws = _repo(tmp_path / "ws")
    app = make_app([REPLY], workspace=ws, context_window=8192)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)
        await pilot.pause()

        message = _user_message(app)
        head, _, facts = message.partition("\n\n" + INTRO)
        assert facts, "the facts block follows the instructions"
        assert message.endswith(facts)
        step_1 = next(line for line in head.splitlines() if line.startswith("1. "))
        assert "README" in step_1 and "makefile" not in step_1
        lines = message.splitlines()
        for expected in (
            "- make test — Run the API and CLI test suites",
            "- make -C cli check-ci — Run lint + type checks without fixing anything. For CI.",
            "- bun run --cwd ui web:test",
            "- make check-ci (in api)",
            "- ui/ (package.json)",
            "- api/ (makefile, pyproject.toml)",
        ):
            assert expected in lines, expected
        assert [e.text for e in app.query(UserEntry)] == [message]


async def test_the_facts_block_stays_within_a_fifth_of_the_window(make_app: AppFactory, tmp_path: Path) -> None:
    ws = _repo(tmp_path / "ws")
    app = make_app([REPLY], workspace=ws, context_window=2048)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)

        message = _user_message(app)
        facts = message[message.index(INTRO) :]
        assert len(facts) <= int(0.2 * 2048) * 4


async def test_a_folder_with_no_facts_gets_todays_message(make_app: AppFactory, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)  # a subfolder alone is not a fact
    _write(ws / "README.md", "hi")
    app = make_app([REPLY], workspace=ws)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await settled(app)

        assert _user_message(app) == build_init_prompt(ws.resolve(), False, [])


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


async def test_a_wrong_command_in_the_written_file_gets_one_warning(make_app: AppFactory, tmp_path: Path) -> None:
    ws = _repo(tmp_path / "ws")
    text = "# P\n\n`make test` `make lint` `make -C cli nope` `bun run --cwd ui web:test`\n"
    app = make_app([[_agents_md(text), Done(1, 1)], REPLY], workspace=ws, context_window=8192)
    async with app.run_test() as pilot:
        await settled(app)
        await _run_init(app, pilot)

        assert _check_notices(app) == [
            "/init check: not found by the scan: make lint, make -C cli nope. "
            "They may still be right; check before relying on them."
        ]
        assert not app._turn_active


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


async def test_a_denied_write_gets_the_old_notice_and_no_check(make_app: AppFactory, tmp_path: Path) -> None:
    ws = _repo(tmp_path / "ws")
    (ws / "AGENTS.md").write_text("`make nope`\n")  # an existing wrong command must not be reported when nothing changed
    app = make_app([[_agents_md("`make lint`"), Done(1, 1)], REPLY], workspace=ws)
    async with app.run_test() as pilot:
        await settled(app)
        await _run_init(app, pilot, approve=False)

        assert _check_notices(app) == []
        assert any("finished without changing" in e.text for e in app.query(NoticeEntry))


async def test_a_target_added_in_the_same_turn_is_found(make_app: AppFactory, tmp_path: Path) -> None:
    ws = _repo(tmp_path / "ws")
    edit = ToolCallEvent(ToolCall("w0", "write", {"path": "makefile", "content": ROOT_MAKEFILE + "\nfresh:\n"}))
    app = make_app([[edit, _agents_md("`make fresh`"), Done(1, 1)], REPLY], workspace=ws, context_window=8192)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        for _ in range(2):
            await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
            await choose(pilot, "once")
            await pilot.pause()
        await settled(app)
        await pilot.pause()

        assert _check_notices(app) == ["/init check: the 1 make or script command was found by the scan"]


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


async def test_esc_after_the_write_still_reports_once_and_ends_the_turn(
    make_app: AppFactory, tmp_path: Path
) -> None:
    gate = Gate()
    app = make_app([[_agents_md("`make lint`"), Done(1, 1)], [gate, *REPLY]], workspace=_repo(tmp_path / "ws"))
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/init")
        await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await choose(pilot, "once")
        await reached(gate)
        await pilot.press("escape")
        await settled(app)
        await pilot.pause()

        assert len(_check_notices(app)) == 1 and "make lint" in _check_notices(app)[0]
        assert not app._turn_active
