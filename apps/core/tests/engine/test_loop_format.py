"""Formatters after write/edit inside the loop: the note, undo, ordering before the LSP block, Esc, and who never gets it."""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import date
from pathlib import Path

import pytest
from core.engine.approval import Choice, Decision
from core.engine.events import AgentEvent, EventSink, Notice, TurnEnded
from core.engine.format.config import FormatSetup, FormatterSpec
from core.engine.format.runner import FormatHub
from core.engine.guardrails.setup import prepare as prepare_guard
from core.engine.loop import Agent
from core.engine.lsp import server as server_module
from core.engine.lsp.config import LspSetup, ServerSpec
from core.engine.lsp.hub import LspHub
from core.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from core.engine.tools import default_tools
from core.engine.tools.base import ToolContext
from core.testing.fakes import Gate, MemoryRecorder, RecordingApprover, ScriptedProvider
from core.testing.format_fakes import fake_formatter
from core.testing.helpers import wait_until
from core.testing.lsp_fakes import fake_lsp

ONCE = Decision(Choice.ONCE)
Script = list[list[ProviderEvent | Gate]]


def call(call_id: str, name: str, **args: object) -> list[ProviderEvent | Gate]:
    return [ToolCallEvent(ToolCall(call_id, name, dict(args))), Done(10, 5)]


def done() -> list[ProviderEvent | Gate]:
    return [TextDelta("done"), Done(1, 1)]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(server_module, "QUIET", 0.1)
    monkeypatch.setenv("FAKE_FORMAT_DUMP", str(tmp_path / "fmt.jsonl"))
    monkeypatch.setenv("FAKE_LSP_DUMP", str(tmp_path / "lsp.jsonl"))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("hello\n")
    return root.resolve()


@pytest.fixture
def ctx(workspace: Path, tmp_path: Path) -> ToolContext:
    return ToolContext(workspace, 5.0, guard=prepare_guard(workspace, None, None, {}, home=tmp_path / "home"))


def fmt_hub(tmp_path: Path, ctx: ToolContext, mode: str = "upper", timeout: float = 5) -> FormatHub:
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    spec = FormatterSpec("fmt", (str(fake_formatter(folder, "fmt", mode)), "{file}"), frozenset({".txt"}), {}, "global")
    assert ctx.guard is not None
    return FormatHub(FormatSetup(True, (spec,), (), ()), ctx.workspace, ctx.guard, timeout=timeout)


def make(ctx: ToolContext, script: Script, hub: FormatHub | None, *, writes: int = 1, lsp: LspHub | None = None, subagent: str = "") -> Agent:
    return Agent(
        ScriptedProvider(script, context_window=100_000),
        default_tools(),
        RecordingApprover([ONCE] * (writes + 2)),
        ctx,
        25,
        today=lambda: date(2026, 11, 14),
        recorder=MemoryRecorder(),
        format=hub,
        lsp=lsp,
        subagent=subagent,
    )


async def run(agent: Agent) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    await agent.run_turn("go", events.append)
    return events


def result_of(agent: Agent, call_id: str = "c1") -> str:
    return next(m.content for m in agent.history if m.role == "tool" and m.tool_call_id == call_id)


WRITE = call("c1", "write", path="a.txt", content="one\ntwo\n")


async def test_ac1_without_a_hub_no_process_starts_and_the_file_is_as_written(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent = make(ctx, [WRITE, done()], None)
    await run(agent)
    assert (workspace / "a.txt").read_text() == "one\ntwo\n" and not (tmp_path / "fmt.jsonl").exists()


async def test_ac2_a_change_adds_one_note_line_and_no_change_adds_nothing(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    plain = make(ctx, [WRITE, done()], None)
    await run(plain)
    agent = make(ctx, [WRITE, done()], fmt_hub(tmp_path, ctx))
    await run(agent)
    assert (workspace / "a.txt").read_text() == "ONE\nTWO\n"
    assert result_of(agent) == f"{result_of(plain)}\n[formatted by fmt: 4 lines changed]"
    quiet = make(ctx, [WRITE, done()], fmt_hub(tmp_path, ctx, "same"))
    await run(quiet)
    assert result_of(quiet) == result_of(plain)


async def test_a_failure_adds_the_failure_note_and_keeps_the_written_file(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent = make(ctx, [WRITE, done()], fmt_hub(tmp_path, ctx, "fail"))
    await run(agent)
    assert (workspace / "a.txt").read_text() == "one\ntwo\n"
    assert "[format fmt failed: exit 1: boom: cannot format]" in result_of(agent)


async def test_ac5_undo_restores_the_original_and_the_log_holds_the_formatted_digest(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent = make(ctx, [WRITE, done()], fmt_hub(tmp_path, ctx))
    await run(agent)
    turn = agent.undo_log.peek()
    assert turn is not None
    [change] = turn.changes.values()
    assert change.after == hashlib.sha256(b"ONE\nTWO\n").digest()
    events: list[AgentEvent] = []
    await agent.undo(events.append)
    assert (workspace / "a.txt").read_text() == "hello\n"


async def test_a_new_file_that_was_formatted_is_still_deleted_by_undo(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent = make(ctx, [call("c1", "write", path="new.txt", content="a\n"), done()], fmt_hub(tmp_path, ctx))
    await run(agent)
    assert (workspace / "new.txt").read_text() == "A\n"
    await agent.undo([].append)
    assert not (workspace / "new.txt").exists()


async def test_ac5_a_second_edit_of_the_formatted_file_passes_the_file_log_check(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    script: Script = [WRITE, call("c2", "edit", path="a.txt", old_string="ONE", new_string="uno"), done()]
    agent = make(ctx, script, fmt_hub(tmp_path, ctx), writes=2)
    await run(agent)
    assert (workspace / "a.txt").read_text() == "UNO\nTWO\n"
    await agent.undo([].append)
    assert (workspace / "a.txt").read_text() == "hello\n"


async def test_ac8_the_language_server_gets_the_formatted_text(ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "lspbin"
    folder.mkdir()
    spec = ServerSpec("ty", str(fake_lsp(folder, "ty")), (), frozenset({".txt"}), {}, "global")
    lsp = LspHub(LspSetup(True, False, (spec,), ()), workspace, dict(os.environ), wait=3, start_wait=5, max_lines=20)
    seen: list[str] = []
    original = lsp.block

    async def spy(path: Path, text: str, emit: EventSink, budget: int) -> str:
        seen.append(text)
        return await original(path, text, emit, budget)

    monkeypatch.setattr(lsp, "block", spy)
    agent = make(ctx, [WRITE, done()], fmt_hub(tmp_path, ctx), lsp=lsp)
    await run(agent)
    assert seen == ["ONE\nTWO\n"]
    await lsp.close()


async def test_esc_during_the_formatter_keeps_the_tool_output_and_the_written_file(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    agent = make(ctx, [WRITE, done()], fmt_hub(tmp_path, ctx, "sleep", timeout=30))
    events: list[AgentEvent] = []
    task = asyncio.ensure_future(agent.run_turn("go", events.append))
    await wait_until(lambda: (tmp_path / "fmt.jsonl").exists())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (workspace / "a.txt").read_text() == "one\ntwo\n"
    assert "formatted by" not in result_of(agent) and TurnEnded("interrupted") in events


async def test_a_failed_edit_is_not_formatted(ctx: ToolContext, tmp_path: Path) -> None:
    agent = make(ctx, [call("c1", "edit", path="a.txt", old_string="nope", new_string="y"), done()], fmt_hub(tmp_path, ctx))
    await run(agent)
    assert not (tmp_path / "fmt.jsonl").exists()


async def test_a_child_agent_gets_no_formatting(ctx: ToolContext, tmp_path: Path) -> None:
    assert make(ctx, [done()], fmt_hub(tmp_path, ctx), subagent="w").format is None
    assert make(ctx, [done()], fmt_hub(tmp_path, ctx)).format is not None


async def test_a_broken_formatter_step_is_a_warning_and_keeps_the_result(ctx: ToolContext, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = fmt_hub(tmp_path, ctx)

    async def boom(path: Path, written: bytes) -> list[object]:
        raise RuntimeError("bug")

    monkeypatch.setattr(hub, "format", boom)
    agent = make(ctx, [WRITE, done()], hub)
    events = await run(agent)
    assert any(isinstance(e, Notice) and e.text.startswith("format failed:") for e in events)
    assert (workspace / "a.txt").read_text() == "one\ntwo\n" and "format" not in result_of(agent)


async def test_a_file_the_config_watch_watches_is_never_formatted(ctx: ToolContext, workspace: Path, tmp_path: Path) -> None:
    from core.engine.configwatch import ConfigWatch, WatchedSet

    (workspace / ".agent").mkdir()
    (workspace / ".agent" / "settings.json").write_text("{}")
    seen: list[object] = []
    xdg = {"XDG_CONFIG_HOME": str(tmp_path / "config"), "XDG_DATA_HOME": str(tmp_path / "data")}
    watch = ConfigWatch(WatchedSet.for_run(workspace, xdg, tmp_path / "home"), seen.append)
    folder = tmp_path / "bin"
    folder.mkdir()
    spec = FormatterSpec("fmt", (str(fake_formatter(folder, "fmt", "upper")), "{file}"), frozenset({".json", ".txt"}), {}, "global")
    assert ctx.guard is not None
    hub = FormatHub(FormatSetup(True, (spec,), (), ()), workspace, ctx.guard, timeout=5)
    script: Script = [call("c1", "write", path=".agent/settings.json", content='{"a": 1}\n'), call("c2", "write", path="a.txt", content="x\n"), done()]
    agent = make(ctx, script, hub, writes=2)
    agent.watch = watch
    await run(agent)
    assert (workspace / ".agent" / "settings.json").read_text() == '{"a": 1}\n'
    assert "formatted" not in result_of(agent, "c1") and "formatted by fmt" in result_of(agent, "c2")
    assert seen == []
