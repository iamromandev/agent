"""`agent -p --output json|stream-json` end to end with a scripted provider: one result object, the NDJSON stream,
stderr rules, every exit path, redaction, the task tool, a closed stdout."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.config.settings import get_settings
from src.engine.provider.base import Done, ProviderError, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import SessionStore, sessions_root
from src.headless import EXIT_FAIL, EXIT_OK, EXIT_USAGE, HeadlessOptions
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


def test_json_prints_one_result_object(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (ws / "a.txt").write_text("alpha")

    code = go(ws, two_calls())

    out, _ = capsys.readouterr()
    (obj,) = lines(out)
    assert out.endswith("\n") and code == EXIT_OK
    assert (obj["type"], obj["result"], obj["is_error"], obj["exit_code"], obj["stop_reason"], obj["error"]) == ("result", "done", False, 0, "done", None)
    assert [(t["id"], t["name"]) for t in obj["tool_calls"]] == [("c1", "read"), ("c2", "glob")]
    assert obj["tool_calls"][0]["args"] == {"path": "a.txt"} and all(t["summary"] for t in obj["tool_calls"])
    assert obj["model"] == "fake" and obj["steps"] == 3
    assert obj["usage"] == {"input_tokens": 30, "output_tokens": 15}
    (info,) = SessionStore(sessions_root(get_settings().sessions_dir), ws.resolve()).list()
    assert obj["session_id"] == info.id


def test_stream_json_events_come_in_order(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (ws / "a.txt").write_text("alpha")

    go(ws, two_calls(), "stream-json")

    events = lines(capsys.readouterr().out)
    launch = 1
    while events[launch]["type"] == "notice":  # launch warnings come right after `start` (on Linux: no sandbox)
        launch += 1
    events = events[:1] + events[launch:]
    assert [e["type"] for e in events] == ["start", "tool_call", "tool_result", "tool_call", "tool_result", "result"]
    assert events[0]["model"] == "fake" and events[0]["cwd"] == str(ws.resolve())
    assert events[1]["name"] == "read" and events[2]["id"] == "c1" and events[2]["name"] == "read"
    assert events[-1]["result"] == "done" and events[-1]["exit_code"] == 0


def test_json_stderr_has_no_progress_lines_but_keeps_warnings(
    ws: Path, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (ws / "a.txt").write_text("alpha")
    settings(env, hooks={"Bogus": [{"command": "true"}]})

    go(ws, two_calls())

    _, err = capsys.readouterr()
    assert "tool " not in err and "saved" not in err
    assert "warning:" in err


def test_text_mode_flag_is_the_same_as_the_default(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    go(ws, ScriptedProvider([say("done")]), "text")

    out, err = capsys.readouterr()
    assert out == "done\n" and "saved" in err


def test_a_usage_error_ends_with_a_result_and_exit_2(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = go(ws, ScriptedProvider([say("x")]), allow=("bash(",))

    out, err = capsys.readouterr()
    (obj,) = lines(out)
    assert code == EXIT_USAGE
    assert (obj["is_error"], obj["exit_code"], obj["stop_reason"]) == (True, 2, "usage")
    assert "invalid --allow rule" in obj["error"]
    assert "invalid --allow rule" in err


def test_a_slash_command_usage_error_in_stream_mode_still_ends_with_a_result(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = go(ws, ScriptedProvider([say("x")]), "stream-json", prompt="/compact")

    events = lines(capsys.readouterr().out)
    assert code == EXIT_USAGE and events[-1]["type"] == "result"
    assert (events[-1]["exit_code"], events[-1]["stop_reason"]) == (2, "usage")


def test_a_provider_error_is_exit_1_error(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = go(ws, ScriptedProvider([[ProviderError("model fell over")]]))

    out, err = capsys.readouterr()
    (obj,) = lines(out)
    assert code == EXIT_FAIL
    assert (obj["exit_code"], obj["stop_reason"], obj["result"]) == (1, "error", "")
    assert "model fell over" in err


def test_a_failed_provider_check_is_exit_1_with_the_problem(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = go(ws, ScriptedProvider([say("x")], check_result="ollama is down"))

    (obj,) = lines(capsys.readouterr().out)
    assert code == EXIT_FAIL
    assert (obj["exit_code"], obj["stop_reason"], obj["error"]) == (1, "error", "ollama is down")


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


def test_secrets_and_escape_codes_are_redacted_and_escaped(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (ws / "a.txt").write_text(f"{TOKEN}\x1b[2J")
    provider = ScriptedProvider([call("read", "c1", path="a.txt", note=f"{TOKEN}\x1b[2J"), say("ok")])

    go(ws, provider, "stream-json")

    raw = capsys.readouterr().out
    assert TOKEN not in raw and "\x1b" not in raw
    assert "[redacted:github-token]" in raw and "\\u001b[2J" in raw
    call_event = next(e for e in lines(raw) if e["type"] == "tool_call")
    assert call_event["args"]["note"] == "[redacted:github-token]\x1b[2J"
    result = lines(raw)[-1]
    assert TOKEN not in json.dumps(result["tool_calls"])


def test_a_task_call_is_listed_once_with_no_child_calls(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name in "abc":
        (ws / f"{name}.txt").write_text(name)
    provider = ScriptedProvider(
        [
            call("task", "t1", description="look", prompt="read them"),
            call("read", "k1", path="a.txt"),
            call("read", "k2", path="b.txt"),
            call("read", "k3", path="c.txt"),
            say("found"),
            say("done"),
        ]
    )

    go(ws, provider)

    (obj,) = lines(capsys.readouterr().out)
    assert [t["name"] for t in obj["tool_calls"]] == ["task"]
    assert obj["result"] == "done" and obj["steps"] == 2
    assert obj["usage"] == {"input_tokens": 20, "output_tokens": 10}


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


def test_a_keyboard_interrupt_during_cleanup_gives_one_result_and_no_traceback(
    ws: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.runtime import Runtime

    def boom(self: Runtime) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(Runtime, "kill_all", boom)

    code = go(ws, ScriptedProvider([say("done")]))

    assert code == EXIT_OK and _one_result(capsys.readouterr().out)["stop_reason"] == "done"


def test_a_failing_result_build_falls_back_to_a_minimal_ascii_result(
    ws: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.headless.run as run_mod

    def broken(obj: object) -> str:
        if isinstance(obj, dict) and "tool_calls" in obj:
            raise RuntimeError("boom")
        return run_mod.json.dumps(obj, ensure_ascii=True)

    monkeypatch.setattr(run_mod, "json_line", broken)

    code = go(ws, ScriptedProvider([say("done")]))

    obj = _one_result(capsys.readouterr().out)
    assert code == EXIT_OK and obj["exit_code"] == 0 and obj["stop_reason"] == "done" and obj["is_error"] is False


def test_a_token_in_a_tool_summary_is_redacted_in_stream_and_result() -> None:
    from src.engine.events import ToolFinished, ToolStarted
    from src.engine.tools.base import ToolResult
    from src.headless.output import JsonCollector

    out: list[str] = []
    collector = JsonCollector(None, lambda t: t.replace(TOKEN, "[redacted]"), streaming=True)
    collector.flush()
    collector._emit = lambda obj: out.append(json.dumps(obj))  # ty: ignore[invalid-assignment]
    collector(ToolStarted("c1", "read", False, "read", {}))
    collector(ToolFinished("c1", ToolResult("ok", False, f"saw {TOKEN}"), None, True))
    assert TOKEN not in "".join(out) and TOKEN not in json.dumps(collector._calls)
