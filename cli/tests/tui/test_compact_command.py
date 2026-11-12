"""/compact and automatic compaction in the TUI: the notice, the `~` estimate, the compacting status, Esc."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path

from src.agent.compaction import is_summary
from src.agent.provider.base import Done, Message, ProviderError, TextDelta, ToolCall
from src.agent.session import SessionStore, ToolMeta, sessions_root
from src.tui import StartChoice
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import ICONS, NoticeEntry, UserEntry

from tests.fakes import Gate
from tests.tui.conftest import AppFactory, chat, reached, scripted, settled, type_and_send, until

SaveSession = Callable[..., Path]
INFO = ICONS["info"]
NOTICE = re.compile(rf"{INFO} Context compacted: pruned (\d+) tool outputs?(?:, summarized (\d+) messages?)? \(ctx ~(\d+) → ~(\d+)\)")


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _compaction_notice(app) -> re.Match[str]:
    (found,) = [m for n in _notices(app) if (m := NOTICE.fullmatch(n))]
    return found


def tool_turns(count: int, output_chars: int = 400, reply_chars: int = 600) -> list[tuple[Message, ToolMeta | None]]:
    """`count` saved turns, each: user, an assistant read call, its output, an assistant reply."""
    messages: list[tuple[Message, ToolMeta | None]] = []
    for n in range(1, count + 1):
        call = ToolCall(f"c{n}", "read", {"path": f"src/{n}.py"})
        messages += [
            (Message("user", f"u{n}"), None),
            (Message("assistant", "", [call]), None),
            (Message("tool", f"{n}" * output_chars, tool_call_id=f"c{n}", name="read"), ToolMeta(f"read src/{n}.py", f"{n}0 lines")),
            (Message("assistant", f"a{n}" + "." * reply_chars), None),
        ]
    return messages


# --- /compact -----------------------------------------------------------------------------------


async def test_compact_prunes_summarizes_and_shows_the_notice_and_the_estimate(
    make_app: AppFactory, save_session: SaveSession
) -> None:
    save_session(tool_turns(4))
    app = make_app([[TextDelta("The gist"), Done(5, 5)]], start=StartChoice("continue"), tools=False)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/compact")
        await settled(app)
        await pilot.pause()

        found = _compaction_notice(app)
        pruned, summarized, before, after = (int(g) for g in found.groups())
        assert (pruned, summarized) == (2, 8)  # two old read outputs; two turns of four messages
        assert before > after
        status = app.query_one(StatusBar)
        assert status.estimated and status.used == after
        assert f"ctx ~{after} / 1000" in status.line
        assert is_summary(app.agent.history, 0)
        assert [m.content for m in app.agent.history if m.role == "user"][1:] == ["u3", "u4"]
        (_messages, tools) = scripted(app).requests[0]
        assert tools == []  # the summary request carries no tools
        assert not status.busy and not status.compacting


async def test_compact_below_the_threshold_still_shows_the_notice(make_app: AppFactory, save_session: SaveSession) -> None:
    save_session(chat("q", "a"))
    app = make_app(start=StartChoice("continue"), tools=False)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/compact")
        await settled(app)
        await pilot.pause()

        pruned, summarized, before, after = _compaction_notice(app).groups()
        assert (pruned, summarized) == ("0", None)
        assert before == after
        assert scripted(app).requests == []  # nothing older than the kept turns: no request
        assert not app.query_one(StatusBar).busy


async def test_compact_is_saved_so_a_resume_gets_the_compacted_history(
    make_app: AppFactory, save_session: SaveSession, tmp_path: Path
) -> None:
    path = save_session(tool_turns(4))
    app = make_app([[TextDelta("The gist"), Done(5, 5)]], start=StartChoice("continue"), tools=False)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/compact")
        await settled(app)

        loaded = SessionStore(sessions_root(None), tmp_path.resolve()).load(path.stem)
        assert loaded.history == app.agent.history
        assert any(i.kind == "notice" and "Context compacted" in i.text for i in loaded.items)


async def test_compact_while_a_turn_runs_is_refused(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[TextDelta("a"), gate, TextDelta("b"), Done(1, 1)]])
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await type_and_send(pilot, "/compact")
        await pilot.pause()

        assert any("busy" in n and "Esc" in n for n in _notices(app))
        assert not any("Context compacted" in n for n in _notices(app))
        gate.release.set()
        await settled(app)
        assert len(scripted(app).requests) == 1


async def test_a_message_while_compact_runs_is_refused_and_a_second_compact_too(
    make_app: AppFactory, save_session: SaveSession
) -> None:
    save_session(tool_turns(4))
    gate = Gate()
    app = make_app([[gate, TextDelta("The gist"), Done(5, 5)]], start=StartChoice("continue"), tools=False)
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/compact")
        await reached(gate)
        await type_and_send(pilot, "hello")
        await type_and_send(pilot, "/compact")
        await type_and_send(pilot, "/new")
        await pilot.pause()

        assert [n for n in _notices(app) if "busy" in n].__len__() == 3
        assert [e.text for e in app.query(UserEntry)] == ["u1", "u2", "u3", "u4"]
        gate.release.set()
        await settled(app)
        assert len(scripted(app).requests) == 1


async def test_the_status_shows_compacting_while_the_summary_is_written(
    make_app: AppFactory, save_session: SaveSession
) -> None:
    save_session(tool_turns(4))
    gate = Gate()
    app = make_app([[gate, TextDelta("The gist"), Done(5, 5)]], start=StartChoice("continue"), tools=False)
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await settled(app)
        assert "compacting" not in status.line

        await type_and_send(pilot, "/compact")
        await reached(gate)
        await until(pilot, lambda: status.compacting)
        assert "compacting…" in status.line
        assert "working" not in status.line

        gate.release.set()
        await settled(app)
        await pilot.pause()
        assert not status.compacting
        assert "compacting" not in status.line


async def test_esc_cancels_compact_within_half_a_second_and_keeps_the_pruned_history(
    make_app: AppFactory, save_session: SaveSession
) -> None:
    save_session(tool_turns(4))
    gate = Gate()
    app = make_app([[gate, TextDelta("never"), Done(5, 5)]], start=StartChoice("continue"), tools=False)
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await settled(app)
        await type_and_send(pilot, "/compact")
        await reached(gate)

        started = time.monotonic()
        await pilot.press("escape")
        await until(pilot, lambda: not status.busy, timeout=2.0)

        assert time.monotonic() - started < 0.5
        await settled(app)
        await pilot.pause()
        assert "⚠ Compaction interrupted" in _notices(app)
        assert not status.compacting
        assert not is_summary(app.agent.history, 0)
        assert [m.content for m in app.agent.history if m.role == "tool"][:2] == [
            "[output pruned: read src/1.py · 10 lines]",
            "[output pruned: read src/2.py · 20 lines]",
        ]
        assert not app.agent.busy


async def test_compact_works_again_after_an_interrupted_compact(make_app: AppFactory, save_session: SaveSession) -> None:
    save_session(tool_turns(4))
    gate = Gate()
    app = make_app(
        [[gate, TextDelta("never")], [TextDelta("The gist"), Done(5, 5)]], start=StartChoice("continue"), tools=False
    )
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "/compact")
        await reached(gate)
        await pilot.press("escape")
        await settled(app)
        await type_and_send(pilot, "/compact")
        await settled(app)
        await pilot.pause()

        assert is_summary(app.agent.history, 0)


# --- automatic compaction inside a turn ----------------------------------------------------------


async def test_automatic_compaction_shows_the_notice_and_the_estimate_and_the_turn_goes_on(
    make_app: AppFactory, save_session: SaveSession
) -> None:
    save_session(tool_turns(3, output_chars=100, reply_chars=1200))
    app = make_app(
        [[TextDelta("The gist"), Done(5, 5)], [TextDelta("answer"), Done(100, 5)]],
        start=StartChoice("continue"),
        tools=False,
    )
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "next question")
        await settled(app)
        await pilot.pause()

        _pruned, summarized, before, after = (int(g) for g in _compaction_notice(app).groups(default="0"))
        assert summarized > 0 and before > after
        assert is_summary(scripted(app).requests[1][0][1:], 0)
        status = app.query_one(StatusBar)
        assert not status.busy and not status.compacting
        assert status.used == 105 and not status.estimated  # the model's own count after the reply


async def test_the_status_shows_compacting_during_an_automatic_summary_and_esc_ends_the_turn_quickly(
    make_app: AppFactory, save_session: SaveSession
) -> None:
    save_session(tool_turns(3, output_chars=100, reply_chars=1200))
    gate = Gate()
    app = make_app([[gate, TextDelta("never"), Done(5, 5)]], start=StartChoice("continue"), tools=False)
    async with app.run_test() as pilot:
        status = app.query_one(StatusBar)
        await settled(app)
        await type_and_send(pilot, "next question")
        await reached(gate)
        await until(pilot, lambda: status.compacting)
        assert "compacting…" in status.line

        started = time.monotonic()
        await pilot.press("escape")
        await until(pilot, lambda: not status.busy, timeout=2.0)

        assert time.monotonic() - started < 0.5
        await settled(app)
        await pilot.pause()
        assert any("interrupted" in n for n in _notices(app))
        assert not status.compacting
        assert len(scripted(app).requests) == 1  # no real request followed
        assert not app.agent.busy


async def test_a_failed_summary_warns_and_the_turn_still_answers(make_app: AppFactory, save_session: SaveSession) -> None:
    save_session(tool_turns(3, output_chars=100, reply_chars=1200))
    app = make_app(
        [[ProviderError("down")], [TextDelta("answer"), Done(100, 5)]], start=StartChoice("continue"), tools=False
    )
    async with app.run_test() as pilot:
        await settled(app)
        await type_and_send(pilot, "next question")
        await settled(app)
        await pilot.pause()

        assert any(n.startswith("⚠ Could not summarize") for n in _notices(app))
        assert not app.query_one(StatusBar).compacting
        assert app.query_one(StatusBar).used == 105


async def test_compact_with_no_messages_says_there_is_nothing_to_compact_and_saves_nothing(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/compact")
        await settled(app)
        await pilot.pause()

        assert f"{INFO} Nothing to compact" in _notices(app)
        assert not any("Context compacted" in n for n in _notices(app))
        assert not app.query_one(StatusBar).busy

    root = sessions_root(None)
    assert not root.exists() or list(root.rglob("*.jsonl")) == []


async def test_compact_after_one_turn_still_shows_the_compaction_notice(make_app: AppFactory) -> None:
    app = make_app([[TextDelta("a"), Done(5, 5)]], tools=False)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "q")
        await settled(app)
        await type_and_send(pilot, "/compact")
        await settled(app)
        await pilot.pause()

        _compaction_notice(app)
        assert f"{INFO} Nothing to compact" not in _notices(app)
