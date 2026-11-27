"""The /diff viewer: rows, the lazy diff pane, keys, escaping, and that the input stays usable while it is built."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest
from core.engine.checkpoints.store import CheckpointStore
from core.engine.guardrails.setup import Guardrails
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.engine.turn_diff import DiffItem
from rich.text import Text
from src.tui.screens.approval import ApprovalScreen
from src.tui.screens.diff import DiffScreen, describe
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.transcript import Transcript
from textual.widgets import Label, OptionList, Static

from tests.tui.conftest import AppFactory, choose, settled, type_and_send, until

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _store(tmp_path: Path) -> CheckpointStore:
    ws = tmp_path.resolve()
    return CheckpointStore(ws, tmp_path.parent / f"{tmp_path.name}-data", Guardrails.off(ws), dict(os.environ))


def _bash(command: str) -> list:
    return [ToolCallEvent(ToolCall("b1", "bash", {"command": command})), Done(1, 1)]


REPLY = [TextDelta("ok"), Done(1, 1)]


async def _turn(pilot, app) -> None:
    await type_and_send(pilot, "go")
    await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
    await choose(pilot, "once")
    await settled(app)


def _rows(app) -> list[str]:
    picker = app.screen.query_one(OptionList)
    return [str(picker.get_option_at_index(i).prompt) for i in range(picker.option_count)]


def _pane(app) -> str:
    content = app.screen.query_one("#diff", Static).content
    return content.plain if isinstance(content, Text) else str(content)


async def _open(pilot, app) -> None:
    await type_and_send(pilot, "/diff")
    await until(pilot, lambda: isinstance(app.screen, DiffScreen))


async def test_the_viewer_lists_the_turns_files_and_shows_the_first_diff_then_esc_closes_it(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n")
    (tmp_path / "b.py").write_text("x\ny\n")
    command = "sed -i.bak 's/two/TWO/' a.py && rm a.py.bak && rm b.py && printf 'n1\\nn2\\n' > c.py"
    app = make_app([_bash(command), REPLY], checkpoints=_store(tmp_path))
    async with app.run_test() as pilot:
        await _turn(pilot, app)
        entries_before = len(app.query_one(Transcript).children)

        await _open(pilot, app)

        assert _rows(app) == ["M a.py +1 \u22121", "D b.py \u22122", "A c.py +2"]
        await until(pilot, lambda: "+TWO" in _pane(app))
        assert "-two" in _pane(app)
        assert not app.screen.query("#banner")

        await pilot.press("down")
        await until(pilot, lambda: "-x" in _pane(app))

        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, DiffScreen))
        assert len(app.query_one(Transcript).children) == entries_before
        assert app.focused is app.query_one(ChatInput)


async def test_without_checkpoints_the_banner_is_shown_over_a_write_turn(make_app: AppFactory, tmp_path: Path) -> None:
    write = [ToolCallEvent(ToolCall("w", "write", {"path": "n.txt", "content": "hello\n"})), Done(1, 1)]
    app = make_app([write, REPLY])
    async with app.run_test() as pilot:
        await _turn(pilot, app)

        await _open(pilot, app)

        assert "checkpoints off: bash and terminal changes are not shown" in str(app.screen.query_one("#banner", Label).render())
        await until(pilot, lambda: "+hello" in _pane(app))


async def test_the_input_stays_usable_while_the_diff_is_built(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    app = make_app([_bash("echo hi > a.txt"), REPLY], checkpoints=store)
    async with app.run_test() as pilot:
        await _turn(pilot, app)
        real = store.diff
        release = asyncio.Event()

        async def slow(before: str, after: str):
            await release.wait()
            return await real(before, after)

        monkeypatch.setattr(store, "diff", slow)
        await type_and_send(pilot, "/diff")
        await pilot.press(*"abc")

        assert app.query_one(ChatInput).text == "abc"
        assert not isinstance(app.screen, DiffScreen)
        release.set()
        await until(pilot, lambda: isinstance(app.screen, DiffScreen))


async def test_a_failing_file_load_is_shown_in_the_pane_not_raised(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    app = make_app([_bash("echo hi > a.txt"), REPLY], checkpoints=store)
    async with app.run_test() as pilot:
        await _turn(pilot, app)

        async def boom(shas, **kw):
            raise RuntimeError("blob \x1b[31mgone")

        monkeypatch.setattr(store, "read_blobs", boom)
        await _open(pilot, app)

        await until(pilot, lambda: "Could not load this file" in _pane(app))
        assert "\x1b" not in _pane(app)
        assert isinstance(app.screen, DiffScreen)


async def test_pagedown_scrolls_the_diff_pane(make_app: AppFactory, tmp_path: Path) -> None:
    command = "seq 1 300 > long.txt"
    app = make_app([_bash(command), REPLY], checkpoints=_store(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        await _turn(pilot, app)
        await _open(pilot, app)
        await until(pilot, lambda: "+300" in _pane(app))
        pane = app.screen.query_one("#pane")

        await pilot.press("pagedown")
        await pilot.pause()

        assert pane.scroll_y > 0


async def test_a_click_on_a_row_shows_that_files_diff(make_app: AppFactory, tmp_path: Path) -> None:
    command = "printf 'first\\n' > a.txt && printf 'second\\n' > b.txt"
    app = make_app([_bash(command), REPLY], checkpoints=_store(tmp_path))
    async with app.run_test() as pilot:
        await _turn(pilot, app)
        await _open(pilot, app)
        await until(pilot, lambda: "+first" in _pane(app))
        await pilot.pause()

        await pilot.click("#files", offset=(3, 2))  # the list has a border: the second row is on line 2

        await until(pilot, lambda: "+second" in _pane(app))
        assert app.screen.query_one(OptionList).highlighted == 1


def _item(status="M", added=None, removed=None, note=None) -> DiffItem:
    return DiffItem("p.py", status, added, removed, note)


def test_a_row_with_unknown_counts_shows_a_question_mark_unless_a_note_says_why() -> None:
    assert describe(_item("M")).plain == "M p.py ?"
    assert describe(_item("A", note="binary")).plain == "A p.py (binary)"
    assert describe(_item("M", note="too large to compare")).plain == "M p.py (too large to compare)"
