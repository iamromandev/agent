"""`agent -p --output json|stream-json` end to end with a scripted provider: one result object, the NDJSON stream,
stderr rules, every exit path, redaction, the task tool, a closed stdout."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.config.settings import get_settings
from src.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from src.headless import EXIT_FAIL, EXIT_OK, HeadlessOptions
from src.headless.run import run_headless

from tests.fakes import ScriptedProvider

TOKEN = "ghp_" + "b" * 36


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    config = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.delenv("AGENT_SEARCH_URL", raising=False)
    get_settings.cache_clear()
    yield config
    get_settings.cache_clear()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    path.mkdir()
    return path


def say(text: str) -> list:
    return [TextDelta(text), Done(10, 5)]


def call(tool: str, call_id: str = "c1", **args: object) -> list:
    return [ToolCallEvent(ToolCall(call_id, tool, dict(args))), Done(10, 5)]


def go(ws: Path, provider: ScriptedProvider, mode: str = "json", prompt: str = "hi", **opts: object) -> int:
    return run_headless(ws, HeadlessOptions(prompt, output=mode, **opts), provider=provider)  # ty: ignore[invalid-argument-type]


def lines(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines()]


def settings(config: Path, **data: object) -> None:
    (config / "agent").mkdir(parents=True, exist_ok=True)
    (config / "agent" / "settings.json").write_text(json.dumps(data))


def two_calls() -> ScriptedProvider:
    return ScriptedProvider([call("read", "c1", path="a.txt"), call("glob", "c2", pattern="*.txt"), say("done")], context_window=100_000)


def test_the_step_limit_is_exit_1_step_limit(ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AGENT_MAX_STEPS", "1")
    get_settings.cache_clear()
    (ws / "a.txt").write_text("alpha")

    code = go(ws, ScriptedProvider([call("read", "c1", path="a.txt"), say("never")]))

    (obj,) = lines(capsys.readouterr().out)
    assert code == EXIT_FAIL
    assert (obj["exit_code"], obj["stop_reason"], obj["result"]) == (1, "step_limit", "")


def test_a_user_prompt_submit_block_is_exit_1_blocked_with_the_reason(
    ws: Path, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings(env, hooks={"UserPromptSubmit": [{"command": "echo not today >&2; exit 2"}]})

    code = go(ws, ScriptedProvider([say("x")]), "stream-json")

    events = lines(capsys.readouterr().out)
    assert code == EXIT_FAIL
    assert events[-1]["type"] == "result"
    assert (events[-1]["exit_code"], events[-1]["stop_reason"], events[-1]["error"]) == (1, "blocked", "not today")


def test_an_internal_error_is_exit_1_error(ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import src.headless.run as headless_run

    def boom(*_: object, **__: object):
        raise RuntimeError("boom")

    monkeypatch.setattr(headless_run, "_parse_allow", boom)

    code = go(ws, ScriptedProvider([say("x")]))

    out, err = capsys.readouterr()
    (obj,) = lines(out)
    assert code == EXIT_FAIL
    assert (obj["exit_code"], obj["stop_reason"]) == (1, "error") and "boom" in obj["error"]
    assert "internal error: boom" in err


def test_a_failure_inside_the_turn_is_exit_1_error(ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from src.engine import Agent

    async def boom(self: Agent, *_: object, **__: object) -> None:
        raise RuntimeError("turn broke")

    monkeypatch.setattr(Agent, "run_turn", boom)

    code = go(ws, ScriptedProvider([say("x")]))

    (obj,) = lines(capsys.readouterr().out)
    assert code == EXIT_FAIL and obj["stop_reason"] == "error" and "turn broke" in obj["error"]


def test_a_closed_stdout_is_no_traceback_and_the_turn_exit_code(ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import io
    import sys

    class Closed(io.StringIO):
        def write(self, text: str) -> int:
            raise BrokenPipeError

        def fileno(self) -> int:
            return 99

    import src.headless.output as output

    monkeypatch.setattr(output.os, "dup2", lambda *a: None)
    monkeypatch.setattr(sys, "stdout", Closed())

    code = go(ws, ScriptedProvider([say("done")]))

    assert code == EXIT_OK
    assert "Traceback" not in capsys.readouterr().err


def _one_result(out: str) -> dict:
    (obj,) = [e for e in lines(out) if e["type"] == "result"]
    return obj


def test_a_keyboard_interrupt_in_the_final_write_still_gives_one_result(
    ws: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.headless.run as run_mod

    real, hits = run_mod.write_stdout, []

    def flaky(text: str, stream: object = None) -> bool:
        if not hits:
            hits.append(1)
            raise KeyboardInterrupt
        return real(text)

    monkeypatch.setattr(run_mod, "write_stdout", flaky)

    code = go(ws, ScriptedProvider([say("done")]))

    obj = _one_result(capsys.readouterr().out)
    assert code == EXIT_OK and obj["exit_code"] == 0 and obj["stop_reason"] == "done"


def test_a_custom_subagent_call_is_one_task_entry_with_its_agent_argument(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = ws / ".agent" / "agents"
    folder.mkdir(parents=True)
    (folder / "w.md").write_text("---\ndescription: Reads\ntools: read\n---\nYou read.\n")
    (ws / "a.txt").write_text("alpha")
    provider = ScriptedProvider(
        [call("task", "t1", description="d", prompt="p", agent="w"), call("read", "r1", path="a.txt"), say("report"), say("done")],
        context_window=100_000,
    )

    code = go(ws, provider)

    (obj,) = lines(capsys.readouterr().out)
    assert code == 0 and obj["result"] == "done"
    (entry,) = obj["tool_calls"]  # the child's `read` is not listed
    assert entry["name"] == "task" and entry["args"]["agent"] == "w"
