"""/diff in the app: what it says when there is nothing to show, with arguments, while busy, and when git fails."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest
from core.engine.checkpoints.git import GitError
from core.engine.checkpoints.store import CheckpointStore
from core.engine.guardrails.setup import Guardrails
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import NullRecorder
from core.testing.fakes import Gate
from src.tui.screens.diff import DiffScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, choose, reached, settled, type_and_send, until

REPLY = [TextDelta("ok"), Done(1, 1)]


def _notices(app) -> list[str]:
    """The notice texts without their icon (a word and a space)."""
    return [e.text.split(" ", 1)[1] if " " in e.text else e.text for e in app.query(NoticeEntry)]


def _store(tmp_path: Path) -> CheckpointStore:
    ws = (tmp_path).resolve()
    return CheckpointStore(ws, tmp_path.parent / f"{tmp_path.name}-data", Guardrails.off(ws), dict(os.environ))


async def _bash_turn(pilot, app, text: str = "go") -> None:
    from src.tui.screens.approval import ApprovalScreen

    await type_and_send(pilot, text)
    await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
    await choose(pilot, "once")
    await settled(app)


def _bash(command: str) -> list:
    return [ToolCallEvent(ToolCall("b1", "bash", {"command": command})), Done(1, 1)]


async def test_before_any_turn_there_is_nothing_to_diff(make_app: AppFactory) -> None:
    app = make_app([])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/diff")
        await until(pilot, lambda: "Nothing to diff" in _notices(app))
        assert not isinstance(app.screen, DiffScreen)


async def test_arguments_are_refused_with_a_notice(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hi")
        await settled(app)
        await type_and_send(pilot, "/diff x")
        await until(pilot, lambda: "/diff takes no arguments" in _notices(app))
        assert "The last turn changed no files" not in _notices(app)


async def test_a_text_turn_without_checkpoints_says_no_files_changed_and_shows_the_banner(
    make_app: AppFactory,
) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hi")
        await settled(app)
        await type_and_send(pilot, "/diff")
        await until(pilot, lambda: "The last turn changed no files" in _notices(app))
        notices = _notices(app)
        assert notices[notices.index("The last turn changed no files") + 1] == (
            "checkpoints off: bash and terminal changes are not shown"
        )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_text_turn_with_checkpoints_says_no_files_changed_without_a_banner(
    make_app: AppFactory, tmp_path: Path
) -> None:
    app = make_app([REPLY], checkpoints=_store(tmp_path))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "hi")
        await settled(app)
        await type_and_send(pilot, "/diff")
        await until(pilot, lambda: "The last turn changed no files" in _notices(app))
        assert not any(n.startswith("checkpoints off") for n in _notices(app))


async def test_diff_during_a_turn_shows_the_busy_line(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, *REPLY]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "slow")
        await reached(gate)
        await type_and_send(pilot, "/diff")
        await until(pilot, lambda: any("press Esc to interrupt, then /diff" in n for n in _notices(app)))
        gate.release.set()
        await settled(app)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_failing_git_is_one_error_notice(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    app = make_app([_bash("echo hi > a.txt"), REPLY], checkpoints=store)
    async with app.run_test() as pilot:
        await _bash_turn(pilot, app)

        async def boom(before: str, after: str):
            raise GitError("cat-file \x1b[31m exploded")

        monkeypatch.setattr(store, "diff", boom)
        await type_and_send(pilot, "/diff")
        await until(pilot, lambda: any(n.startswith("Could not build the diff:") for n in _notices(app)))
        text = next(n for n in _notices(app) if n.startswith("Could not build the diff:"))
        assert "\x1b" not in text and "\\x1b[31m exploded" in text


@pytest.fixture
def slow_diff(monkeypatch: pytest.MonkeyPatch):
    """Hold `store.diff` until the test releases it."""

    def hold(store: CheckpointStore) -> asyncio.Event:
        real = store.diff
        release = asyncio.Event()

        async def slow(before: str, after: str):
            await release.wait()
            return await real(before, after)

        monkeypatch.setattr(store, "diff", slow)
        return release

    return hold


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_diff_still_building_when_a_message_is_sent_never_opens_over_the_new_turn(
    make_app: AppFactory, tmp_path: Path, slow_diff
) -> None:
    store = _store(tmp_path)
    app = make_app([_bash("echo hi > a.txt"), REPLY, REPLY], checkpoints=store)
    async with app.run_test() as pilot:
        await _bash_turn(pilot, app)
        release = slow_diff(store)
        await type_and_send(pilot, "/diff")
        await type_and_send(pilot, "next")
        await settled(app)

        release.set()
        await pilot.pause()
        await pilot.pause()

        assert not isinstance(app.screen, DiffScreen)
        assert app.query_one(ChatInput).text == ""


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_diff_of_a_turn_that_is_no_longer_the_newest_is_dropped_even_if_the_worker_was_not_cancelled(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    app = make_app([_bash("echo hi > a.txt"), REPLY], checkpoints=store)
    async with app.run_test() as pilot:
        await _bash_turn(pilot, app)
        real = app.agent.last_turn_diff

        async def then_the_history_is_cleared():
            diff = await real()
            app.agent.reset(recorder=NullRecorder())  # as if /new ran right after the list was built
            return diff

        monkeypatch.setattr(app.agent, "last_turn_diff", then_the_history_is_cleared)
        await type_and_send(pilot, "/diff")
        await settled(app)
        await pilot.pause()

        assert not isinstance(app.screen, DiffScreen)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
async def test_a_diff_built_while_a_turn_started_shows_the_busy_line_instead_of_opening(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    app = make_app([_bash("echo hi > a.txt"), REPLY], checkpoints=store)
    async with app.run_test() as pilot:
        await _bash_turn(pilot, app)
        real = app.agent.last_turn_diff

        async def then_a_turn_is_active():
            diff = await real()
            app._turn_active = True
            return diff

        monkeypatch.setattr(app.agent, "last_turn_diff", then_a_turn_is_active)
        await type_and_send(pilot, "/diff")
        await until(pilot, lambda: any("press Esc to interrupt, then /diff" in n for n in _notices(app)))
        app._turn_active = False

        assert not isinstance(app.screen, DiffScreen)
