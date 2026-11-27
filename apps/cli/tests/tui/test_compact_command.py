"""/compact and automatic compaction in the TUI: the notice, the `~` estimate, the compacting status, Esc."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from core.engine.provider.base import Done, Message, TextDelta, ToolCall
from core.engine.session import ToolMeta
from core.testing.fakes import Gate
from src.tui import StartChoice
from src.tui.styles import ICONS
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, reached, settled, type_and_send, until

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


# --- automatic compaction inside a turn ----------------------------------------------------------
