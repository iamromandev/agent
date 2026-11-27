"""Message queueing in the app: messages sent during a turn wait and run as one turn, or come back to the box."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from core.engine.commands.loader import Command, Commands
from core.engine.hooks.setup import prepare as prepare_hooks
from core.engine.mcp.trust import TrustStore, trust_store_path
from core.engine.permissions.setup import prepare
from core.engine.provider.base import Done, ProviderError, TextDelta, ToolCall, ToolCallEvent
from core.testing.fakes import FakeRunner, Gate, exit_run
from src.tui.app import AgentApp
from src.tui.queue import Queued
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.queue_panel import QueuePanel
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry, UserEntry

from tests.tui.conftest import AppFactory, reached, scripted, settled, type_and_send, until

REPLY = [TextDelta("ok"), Done(1, 1)]


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _box(app) -> str:
    return app.query_one(ChatInput).text


def _panel(app) -> QueuePanel:
    return app.query_one(QueuePanel)


def _users(app) -> list[str]:
    return [m.content for m in scripted(app).requests[-1][0] if m.role == "user"]


def _last_user(app) -> str:
    return _users(app)[-1]


def _commands(*commands: Command) -> Commands:
    return Commands({c.name: c for c in commands}, ())


REVIEW = Command("review", "Review: $ARGUMENTS", None, "global")


async def _hold(pilot, app, gate: Gate, first: str = "first") -> None:
    await type_and_send(pilot, first)
    await reached(gate)


async def test_the_panel_exists_and_is_hidden_at_start(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test():
        assert not _panel(app).display and _panel(app).plain == ""


async def test_ac1_a_message_sent_during_a_turn_is_queued(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate], REPLY])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "hello")
        await pilot.pause()

        assert _box(app) == "" and _panel(app).display and _panel(app).plain == "hello"
        assert not any("busy" in n for n in _notices(app))
        assert app.query_one(StatusBar).busy and len(scripted(app).requests) == 1
        gate.release.set()
        await settled(app)


async def test_ac2_the_queue_runs_as_one_merged_turn_when_the_turn_ends_done(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate], REPLY])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "a")
        await type_and_send(pilot, "b")
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == 2
        assert _last_user(app) == "a\n\nb"
        assert [e.text for e in app.query(UserEntry)] == ["first", "a\n\nb"]
        assert not _panel(app).display and not app.query_one(StatusBar).busy and not app._turn_active


async def test_ac3_a_mention_in_the_queue_is_attached(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("file body\n")
    gate = Gate()
    app = make_app([[gate], REPLY])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "see @x.txt ")
        await pilot.pause()
        gate.release.set()
        await until(pilot, lambda: len(scripted(app).requests) == 2 and not app._turn_active)

        last = _last_user(app)
        assert last.startswith("see @x.txt") and "file body" in last and '<attached path="' in last


async def test_ac4_and_ac11_the_hook_sees_the_merged_prompt_once_and_a_block_restores_the_typed_lines(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.json").write_text('{"hooks": {"UserPromptSubmit": [{"command": "gate"}]}}')
    runner = FakeRunner(lambda _c, p: exit_run(2, "no") if p["prompt"] == "a\n\nb" else exit_run(0))
    setup = prepare(
        tmp_path.resolve(), tmp_path / "config" / "settings.json", TrustStore(trust_store_path(name="permission-trust.json"))
    )
    hooks = prepare_hooks(setup, tmp_path.resolve(), runner)
    gate = Gate()
    app = make_app([[gate], REPLY], permissions=setup.rules, hooks=hooks)
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "a")
        await type_and_send(pilot, "b")
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert [c[1]["prompt"] for c in runner.calls] == ["first", "a\n\nb"]  # once for the merged prompt
        assert len(scripted(app).requests) == 1  # the blocked turn made no request
        assert _box(app) == "a\nb" and not _panel(app).display


async def test_ac5_a_queued_custom_command_runs_expanded_with_its_label(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate], REPLY], commands=_commands(REVIEW))
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "/review x")
        await pilot.pause()
        assert _panel(app).plain == "/review x"
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert _last_user(app) == "Review: x"
        entries = list(app.query(UserEntry))
        assert entries[-1].text == "Review: x" and "/review (global)" in repr(entries[-1].label)


async def test_ac6_a_mix_of_message_and_command_has_no_label(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate], REPLY], commands=_commands(REVIEW))
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "hi")
        await type_and_send(pilot, "/review x")
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert _last_user(app) == "hi\n\nReview: x"
        assert list(app.query(UserEntry))[-1].label in (None, "")


async def test_ac7_builtins_are_still_refused_during_a_turn_and_nothing_is_queued(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate]])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        for command in ("/undo", "/model x", "/new"):
            await type_and_send(pilot, command)
            await pilot.pause()
        assert len([n for n in _notices(app) if "busy" in n]) == 3
        assert not app._queue and not _panel(app).display
        gate.release.set()
        await settled(app)


async def test_ac7_a_message_during_a_session_load_is_refused_and_restored(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        app._switching = True
        try:
            await type_and_send(pilot, "hello")
            await pilot.pause()
            assert any("busy — a session is loading" in n for n in _notices(app))
            assert _box(app) == "hello" and not app._queue
        finally:
            app._switching = False


async def test_ac8_whitespace_is_not_queued(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate]])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "   ")
        await pilot.pause()
        assert not app._queue and not _panel(app).display
        gate.release.set()
        await settled(app)


@pytest.mark.parametrize("key", ["escape", "ctrl+c"])
async def test_ac9_interrupting_puts_the_queue_and_the_typed_text_back(make_app: AppFactory, key: str) -> None:
    gate = Gate()
    app = make_app([[gate], REPLY])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "a")
        await type_and_send(pilot, "b")
        await pilot.press("c")
        await pilot.press(key)
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == 1
        assert not _panel(app).display and _box(app) == "a\nb\nc"
        assert not app.query_one(StatusBar).busy


async def test_ac10_an_error_puts_the_queue_back(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, ProviderError("boom")], REPLY])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "a")
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == 1 and _box(app) == "a" and not _panel(app).display


async def test_ac10_the_step_limit_puts_the_queue_back(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n")
    gate = Gate()
    read = ToolCallEvent(ToolCall("r1", "read", {"path": "a.txt"}))
    app = make_app([[gate, read, Done(1, 1)], REPLY], max_steps=1)
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "a")
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == 1 and _box(app) == "a"


async def test_ac12_undo_gives_back_the_typed_lines(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate], REPLY], commands=_commands(REVIEW))
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "a")
        await type_and_send(pilot, "/review x")
        gate.release.set()
        await settled(app)
        await pilot.pause()
        assert _box(app) == ""

        await type_and_send(pilot, "/undo")
        await pilot.pause()
        assert _box(app) == "a\n/review x"


async def test_ac13_something_queued_during_the_merged_turn_runs_after_it(make_app: AppFactory) -> None:
    first, second = Gate(), Gate()
    app = make_app([[first], [second], REPLY])
    async with app.run_test() as pilot:
        await _hold(pilot, app, first)
        await type_and_send(pilot, "a")
        first.release.set()
        await reached(second)
        await type_and_send(pilot, "b")
        await pilot.pause()
        assert _panel(app).plain == "b"
        second.release.set()
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == 3 and _last_user(app) == "b"


async def test_ac14_a_message_queued_during_compact_runs_when_it_finishes(make_app: AppFactory) -> None:
    app = make_app([REPLY, REPLY, REPLY, REPLY])
    async with app.run_test() as pilot:
        for text in ("one", "two", "three"):
            await type_and_send(pilot, text)
            await settled(app)
        gate = Gate()
        scripted(app).script.insert(0, [gate, TextDelta("gist"), Done(1, 1)])
        before = len(scripted(app).requests)
        await type_and_send(pilot, "/compact")
        await reached(gate)
        await type_and_send(pilot, "a")
        await pilot.pause()
        assert _panel(app).plain == "a"
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == before + 2 and _last_user(app) == "a"


async def test_ac15_the_panel_shows_five_lines_a_count_and_escapes(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate]])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        for n in range(7):
            await type_and_send(pilot, str(n))
        await pilot.pause()
        assert _panel(app).plain.splitlines() == [f"{n}" for n in range(5)] + ["and 2 more"]

        app._queue.take()
        from src.tui.queue import Queued

        app._queue.add(Queued("a\x1b[2Jb‮c\nmore", "x", None))
        app._show_queue()
        plain = _panel(app).plain
        assert "\x1b" not in plain and "‮" not in plain and plain.endswith("…")
        gate.release.set()
        app._queue.take()
        await settled(app)


async def test_ac16_quitting_with_a_queue_exits_cleanly(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate], REPLY])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "a")
        await pilot.pause()
        app.exit()
    assert len(scripted(app).requests) == 1


async def test_ac17_an_empty_custom_command_is_not_queued(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate]], commands=_commands(Command("blank", "  \n", None, "project")))
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        await type_and_send(pilot, "/blank")
        await pilot.pause()
        assert any("/blank is empty" in n for n in _notices(app))
        assert not app._queue
        gate.release.set()
        await settled(app)


async def test_ac18_a_hundred_large_entries_queue_quickly(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate]])
    async with app.run_test() as pilot:
        await _hold(pilot, app, gate)
        start = time.perf_counter()
        for _ in range(100):
            await app.on_chat_input_submitted(ChatInput.Submitted("z" * 10_000))
        assert time.perf_counter() - start < 1.0
        assert len(app._queue.entries) == 100
        app._queue.take()
        gate.release.set()
        await settled(app)


class _SlowRunner(FakeRunner):
    """A hook runner that waits for `gate` before it answers."""

    def __init__(self, gate: asyncio.Event, answer: Callable[[str, dict[str, Any]], Any]) -> None:
        super().__init__(answer)
        self.gate = gate

    async def run(self, command: str, stdin: bytes, timeout: float, env_extra: Mapping[str, str], *, event: str = ""):
        await self.gate.wait()
        return await super().run(command, stdin, timeout, env_extra)


async def test_ac22_a_blocked_turn_restores_its_text_then_the_queue_then_what_was_typed(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.json").write_text('{"hooks": {"UserPromptSubmit": [{"command": "gate"}]}}')
    release = asyncio.Event()
    runner = _SlowRunner(release, lambda _c, _p: exit_run(2, "no"))
    setup = prepare(
        tmp_path.resolve(), tmp_path / "config" / "settings.json", TrustStore(trust_store_path(name="permission-trust.json"))
    )
    hooks = prepare_hooks(setup, tmp_path.resolve(), runner)
    app = make_app([REPLY], permissions=setup.rules, hooks=hooks)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "D")
        await until(pilot, lambda: app._turn_active)
        await type_and_send(pilot, "Q")
        await pilot.pause()
        await pilot.press("B")
        release.set()
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == 0 and _box(app) == "D\nQ\nB"


async def test_ac21_escape_after_done_while_init_checks_the_file_restores_the_queue(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    hold = asyncio.Event()

    async def slow_check(self) -> None:
        await hold.wait()

    monkeypatch.setattr(AgentApp, "_check_init", slow_check)
    prints = iter(range(100))
    monkeypatch.setattr("src.tui.app.fingerprint", lambda _path: next(prints))  # AGENTS.md "changed": the check runs
    app = make_app([REPLY, REPLY])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/init")
        await until(pilot, lambda: app._job_status == "done")
        await type_and_send(pilot, "a")
        await pilot.pause()
        assert _panel(app).plain == "a"
        await pilot.press("escape")
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == 1 and _box(app) == "a" and not _panel(app).display


async def _compact_with_failing_summary(pilot, app) -> Gate:
    for text in ("one", "two", "three"):
        await type_and_send(pilot, text)
        await settled(app)
    gate = Gate()
    scripted(app).script.insert(0, [gate, ToolCallEvent(ToolCall("x", "read", {})), Done(10, 0)])
    await type_and_send(pilot, "/compact")
    await reached(gate)
    return gate


async def test_ac20_a_failed_compact_summary_puts_the_queue_back(make_app: AppFactory) -> None:
    app = make_app([REPLY, REPLY, REPLY, REPLY])
    async with app.run_test() as pilot:
        gate = await _compact_with_failing_summary(pilot, app)
        before = len(scripted(app).requests)
        await type_and_send(pilot, "a")
        await pilot.pause()
        gate.release.set()
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == before and _box(app) == "a" and not _panel(app).display


@pytest.mark.parametrize("key", ["escape", "ctrl+c"])
async def test_interrupting_compact_puts_the_queue_back(make_app: AppFactory, key: str) -> None:
    app = make_app([REPLY, REPLY, REPLY, REPLY])
    async with app.run_test() as pilot:
        await _compact_with_failing_summary(pilot, app)
        before = len(scripted(app).requests)
        await type_and_send(pilot, "a")
        await pilot.pause()
        await pilot.press(key)
        await settled(app)
        await pilot.pause()

        assert len(scripted(app).requests) == before and _box(app) == "a" and not app.query_one(StatusBar).busy


async def test_cancel_before_the_turn_entered_puts_the_draft_and_the_queue_back(make_app: AppFactory) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        app._queue.add(Queued("q", "q", None))
        app._begin_turn("x", "x")
        app.interrupt_turn()
        await pilot.pause()

        assert _box(app) == "x\nq" and not app._turn_active and not _panel(app).display
        assert len(scripted(app).requests) == 0


async def test_a_failed_provider_recheck_puts_the_draft_and_the_queue_back(
    make_app: AppFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app([REPLY])
    async with app.run_test() as pilot:
        await settled(app)
        release = asyncio.Event()

        async def check() -> str:
            await release.wait()
            return "down"

        app.query_one(StatusBar).error = "not ready"
        monkeypatch.setattr(type(app.provider), "check", lambda _self: check())
        await type_and_send(pilot, "x")
        await until(pilot, lambda: app._turn_active)
        await type_and_send(pilot, "q")
        await pilot.pause()
        release.set()
        await settled(app)
        await pilot.pause()

        assert _box(app) == "x\nq" and len(scripted(app).requests) == 0
