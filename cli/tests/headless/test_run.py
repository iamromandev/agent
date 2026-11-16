"""`run_headless` end to end with a scripted provider: stdout, stderr, exit codes, approvals, sessions, hooks."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from src.config.settings import get_settings
from src.engine.provider.base import Done, ProviderError, TextDelta, ToolCall, ToolCallEvent
from src.engine.session import SessionStore
from src.headless import EXIT_FAIL, EXIT_OK, EXIT_USAGE, HEADLESS_DENY, HeadlessOptions
from src.headless.output import write_stdout
from src.headless.run import run_headless
from src.runtime import StartChoice

from tests.fakes import ScriptedProvider

CLI = Path(__file__).parent.parent.parent


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """An isolated config folder (global settings), and settings read fresh from the environment."""
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


def settings(config: Path, **data: object) -> None:
    (config / "agent").mkdir(parents=True, exist_ok=True)
    (config / "agent" / "settings.json").write_text(json.dumps(data))


def go(ws: Path, provider: ScriptedProvider, prompt: str = "hi", **opts: object) -> int:
    return run_headless(ws, HeadlessOptions(prompt, **opts), provider=provider)  # ty: ignore[invalid-argument-type]


def tool_results(provider: ScriptedProvider, request: int = -1) -> list[str]:
    return [m.content for m in provider.requests[request][0] if m.role == "tool"]


def test_the_final_reply_is_all_of_stdout_and_the_exit_code_is_0(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([say("done")])

    code = go(ws, provider)

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_OK, "done\n")
    assert err.rstrip().endswith("saved") and "session " in err
    assert provider.requests[0][0][-1].content == "hi"


def test_two_tool_steps_and_a_stop_hook_continuation_leave_only_the_last_text_on_stdout(
    ws: Path, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (ws / "a.txt").write_text("alpha")
    marker = ws / "marker"
    stop = f"[ -f {marker} ] || {{ touch {marker}; echo more >&2; exit 2; }}"
    settings(env, hooks={"Stop": [{"command": stop}]})
    provider = ScriptedProvider(
        [
            [TextDelta("step one "), *call("read", path="a.txt")],
            [TextDelta("step two "), *call("glob", "c2", pattern="*.txt")],
            say("first answer"),
            say("final answer"),
        ]
    )

    code = go(ws, provider)

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_OK, "final answer\n")
    assert "tool read: a.txt" in err and "tool glob: *.txt" in err
    assert "hook Stop" in err
    assert len(provider.requests) == 4


def test_a_provider_error_exits_1_with_its_message_and_empty_stdout(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = go(ws, ScriptedProvider([[ProviderError("model fell over")]]))

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_FAIL, "")
    assert "error: model fell over" in err


def test_the_step_limit_exits_1_with_empty_stdout(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AGENT_MAX_STEPS", "1")
    get_settings.cache_clear()
    (ws / "a.txt").write_text("x")

    code = go(ws, ScriptedProvider([call("read", path="a.txt"), say("never")]))

    out, _ = capsys.readouterr()
    assert (code, out) == (EXIT_FAIL, "")


def test_an_unready_provider_exits_1_with_its_message(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([say("x")], check_result="Ollama is not reachable at http://x")

    code = go(ws, provider)

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_FAIL, "")
    assert "error: Ollama is not reachable at http://x" in err
    assert provider.requests == []


def test_an_unapproved_call_is_denied_with_the_note_and_the_turn_goes_on(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ScriptedProvider([call("bash", command="echo hi > made.txt"), say("ok")])

    code = go(ws, provider)

    assert code == EXIT_OK
    assert tool_results(provider) == [f"User denied this call. Note: {HEADLESS_DENY}"]
    assert not (ws / "made.txt").exists()
    assert "tool bash: echo hi > made.txt (denied)" in capsys.readouterr().err


def test_an_allow_flag_lets_exactly_that_command_run_and_nothing_else(ws: Path) -> None:
    provider = ScriptedProvider(
        [
            [*call("bash", "c1", command="echo hi > made.txt")[:1], *call("bash", "c2", command="echo no > other.txt")],
            say("ok"),
        ]
    )

    code = go(ws, provider, allow=("bash(echo hi > made.txt)",))

    assert code == EXIT_OK
    assert (ws / "made.txt").read_text() == "hi\n"
    assert not (ws / "other.txt").exists()
    first, second = tool_results(provider)
    assert not first.startswith("User denied")
    assert second.startswith("User denied this call")


def test_a_deny_rule_beats_an_allow_flag(ws: Path, env: Path) -> None:
    settings(env, permissions={"deny": ["bash(echo*)"]})
    provider = ScriptedProvider([call("bash", command="echo hi > made.txt"), say("ok")])

    go(ws, provider, allow=("bash",))

    assert tool_results(provider)[0].startswith("denied by rule bash(echo*) (global)")
    assert not (ws / "made.txt").exists()


def test_a_protected_read_is_refused_even_with_an_allow_flag(ws: Path) -> None:
    (ws / ".env").write_text("TOKEN=abc")
    provider = ScriptedProvider([call("read", path=".env"), say("ok")])

    go(ws, provider, allow=("read",))

    assert tool_results(provider) == [f"User denied this call. Note: {HEADLESS_DENY}"]


def test_an_invalid_allow_rule_exits_2_naming_it_before_anything_runs(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ScriptedProvider([say("x")])

    code = go(ws, provider, allow=("bash(a*b)",))

    assert code == EXIT_USAGE
    assert "bash(a*b)" in capsys.readouterr().err
    assert provider.requests == []


def test_an_empty_prompt_is_a_usage_error(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = go(ws, ScriptedProvider([say("x")]), prompt="   ")

    assert code == EXIT_USAGE
    assert "prompt" in capsys.readouterr().err


def test_the_terminal_tools_are_not_offered_and_a_forged_call_is_refused(ws: Path) -> None:
    provider = ScriptedProvider([call("term_start", name="a", command="sleep 30"), say("ok")])

    go(ws, provider)

    offered = provider.requests[0][1]
    assert not {"term_start", "term_read", "term_stop"} & set(offered)
    assert "read" in offered
    (result,) = tool_results(provider)
    assert "term_start" in result and "unknown" in result.lower()


def test_tool_results_are_redacted(ws: Path) -> None:
    (ws / "notes.txt").write_text("key ghp_" + "a" * 36)
    provider = ScriptedProvider([call("read", path="notes.txt"), say("ok")])

    go(ws, provider)

    (result,) = tool_results(provider)
    assert "[redacted:github-token]" in result and "ghp_aaaa" not in result


def test_the_session_is_saved_and_opens_in_the_stores_list(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    go(ws, ScriptedProvider([say("one")]), prompt="first")

    (info,) = SessionStore(_sessions(), ws.resolve()).list()
    assert info.message_count == 2
    assert f"session {info.id} saved" in capsys.readouterr().err


def _sessions() -> Path:
    from src.engine.session import sessions_root

    return sessions_root(get_settings().sessions_dir)


def test_continue_resumes_the_latest_session_and_starts_new_when_there_is_none(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fresh = ScriptedProvider([say("zero")])
    go(ws, fresh, prompt="solo", start=StartChoice("continue"))
    assert "No previous session" in capsys.readouterr().err
    go(ws, ScriptedProvider([say("one")]), prompt="first")
    provider = ScriptedProvider([say("two")])

    code = go(ws, provider, prompt="second", start=StartChoice("continue"))

    assert code == EXIT_OK
    texts = [m.content for m in provider.requests[0][0] if m.role == "user"]
    assert texts[-2:] == ["first", "second"]
    store = SessionStore(_sessions(), ws.resolve())
    assert sorted(i.message_count for i in store.list()) == [2, 4]


def test_resume_takes_a_session_by_id_prefix_and_an_unknown_one_exits_2(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    go(ws, ScriptedProvider([say("one")]), prompt="first")
    (info,) = SessionStore(_sessions(), ws.resolve()).list()
    provider = ScriptedProvider([say("two")])

    assert go(ws, provider, prompt="again", start=StartChoice("resume", info.id[:8])) == EXIT_OK
    assert [m.content for m in provider.requests[0][0] if m.role == "user"] == ["first", "again"]

    capsys.readouterr()
    bad = ScriptedProvider([say("x")])
    assert go(ws, bad, start=StartChoice("resume", "ZZZZZZZZ")) == EXIT_USAGE
    assert "ZZZZZZZZ" in capsys.readouterr().err
    assert bad.requests == []


def test_session_start_runs_with_source_startup_then_resume(ws: Path, env: Path) -> None:
    log = ws / "starts.log"
    settings(env, hooks={"SessionStart": [{"command": f"python3 -c 'import json,sys; print(json.load(sys.stdin)[\"source\"])' >> {log}"}]})

    go(ws, ScriptedProvider([say("one")]))
    go(ws, ScriptedProvider([say("two")]), start=StartChoice("continue"))

    assert log.read_text().split() == ["startup", "resume"]


def test_a_user_prompt_submit_block_exits_1_prints_the_reason_and_asks_nothing(
    ws: Path, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings(env, hooks={"UserPromptSubmit": [{"command": "echo not today >&2; exit 2"}]})
    provider = ScriptedProvider([say("x")])

    code = go(ws, provider)

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_FAIL, "")
    assert "blocked: not today" in err
    assert provider.requests == []


def test_a_quiet_run_keeps_progress_lines_off_stderr(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (ws / "a.txt").write_text("x")

    go(ws, ScriptedProvider([call("read", path="a.txt"), say("ok")], context_window=100_000), quiet=True)

    assert capsys.readouterr().err == ""


class _Closed:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None: ...


def test_a_closed_stdout_is_not_an_error() -> None:
    assert write_stdout("text", _Closed()) is False  # ty: ignore[invalid-argument-type]


def test_headless_mode_does_not_import_textual() -> None:
    code = (
        "import sys, src.main, src.headless.run, src.runtime\n"
        "assert 'textual' not in sys.modules, 'textual was imported'"
    )

    done = subprocess.run([sys.executable, "-c", code], cwd=CLI, capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


# --- internal errors, cleanup, final text ---------------------------------------------------------------------


class Spy:
    """Wraps `build_runtime` to see the `Runtime` and to count `close`, `abort` and `kill_all` calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.headless.run as headless_run

        self.rt = None
        self.closed = 0
        self.aborted = 0
        self.killed = 0
        self.agents: list = []
        real = headless_run.build_runtime

        def build(*args: Any, **kwargs: Any):
            rt = real(*args, **kwargs)
            self.rt = rt
            close, abort, kill_all, factory = rt.close, rt.abort, rt.kill_all, rt.agent_factory

            async def counted_close() -> None:
                self.closed += 1
                await close()

            def counted_abort() -> None:
                self.aborted += 1
                abort()

            def counted_kill() -> None:
                self.killed += 1
                kill_all()

            def counted_factory(approver: Any):
                agent = factory(approver)
                self.agents.append(agent)
                return agent

            rt: Any = rt
            rt.close, rt.abort, rt.kill_all, rt.agent_factory = counted_close, counted_abort, counted_kill, counted_factory
            return rt

        monkeypatch.setattr(headless_run, "build_runtime", build)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Spy:
    return Spy(monkeypatch)


def test_an_unexpected_exception_is_exit_1_with_a_one_line_message_and_no_traceback(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], spy: Spy
) -> None:
    import src.headless.run as headless_run

    def boom(*_: object, **__: object):
        raise RuntimeError("boom\x1b[2J")

    monkeypatch.setattr(headless_run, "_parse_allow", boom)

    code = go(ws, ScriptedProvider([say("x")]))

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_FAIL, "")
    assert err.strip() == "agent: internal error: boom\\x1b[2J"
    assert "Traceback" not in err


def test_a_failing_agent_factory_is_exit_1_and_the_runtime_is_still_closed(
    ws: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import types

    import src.headless.run as headless_run

    spy = types.SimpleNamespace(closed=0)

    real = headless_run.build_runtime

    def build(*args: Any, **kwargs: Any):
        rt = real(*args, **kwargs)
        close = rt.close

        async def counted() -> None:
            spy.closed += 1
            await close()

        def factory(_approver: object):
            raise RuntimeError("no agent")

        rt.close, rt.agent_factory = counted, factory
        return rt

    monkeypatch.setattr(headless_run, "build_runtime", build)

    code = go(ws, ScriptedProvider([say("x")]))

    assert code == EXIT_FAIL
    assert "agent: internal error: no agent" in capsys.readouterr().err
    assert spy.closed == 1


def test_a_failure_before_the_loop_ran_aborts_the_runtime_instead_of_closing_it(
    ws: Path, monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    import src.headless.run as headless_run

    real_run = asyncio.run
    calls: list[int] = []

    def boom(coro: object, **kwargs: object):
        if calls:  # `abort` runs the clients' close through the same function
            return real_run(coro, **kwargs)  # ty: ignore[invalid-argument-type]
        calls.append(1)
        coro.close()  # ty: ignore[unresolved-attribute]
        raise RuntimeError("late")

    monkeypatch.setattr(headless_run.asyncio, "run", boom)

    code = go(ws, ScriptedProvider([say("x")]))

    assert code == EXIT_FAIL
    assert (spy.aborted, spy.closed, spy.killed) == (1, 0, 1)


@pytest.mark.parametrize(
    "script",
    [[say("fine")], [[ProviderError("bad")]], [call("read", path="nope"), say("ok")]],
    ids=["ok", "provider-error", "tool-error"],
)
def test_the_runtime_is_closed_exactly_once_and_killed_on_every_exit_path(ws: Path, spy: Spy, script: list) -> None:
    go(ws, ScriptedProvider(script))

    assert (spy.closed, spy.aborted, spy.killed) == (1, 0, 1)


def test_the_runtime_is_closed_when_the_provider_is_not_ready_and_when_a_hook_blocks(ws: Path, env: Path, spy: Spy) -> None:
    go(ws, ScriptedProvider([say("x")], check_result="down"))
    assert (spy.closed, spy.killed) == (1, 1)
    settings(env, hooks={"UserPromptSubmit": [{"command": "exit 2"}]})
    go(ws, ScriptedProvider([say("x")]))
    assert (spy.closed, spy.killed) == (2, 2)


def test_the_runtime_is_closed_on_a_usage_error_from_resume(ws: Path, spy: Spy) -> None:
    code = go(ws, ScriptedProvider([say("x")]), start=StartChoice("resume", "ZZZZZZZZ"))

    assert code == EXIT_USAGE
    assert (spy.closed, spy.killed) == (1, 1)


def test_an_empty_final_reply_prints_just_a_newline_and_exits_0(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = go(ws, ScriptedProvider([[Done(10, 5)]]))

    out, _ = capsys.readouterr()
    assert (code, out) == (EXIT_OK, "\n")


def test_an_empty_final_reply_does_not_print_an_earlier_turns_text_on_a_continued_session(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    go(ws, ScriptedProvider([say("the old answer")]), prompt="first")
    capsys.readouterr()

    code = go(ws, ScriptedProvider([[Done(10, 5)]]), prompt="second", start=StartChoice("continue"))

    out, _ = capsys.readouterr()
    assert (code, out) == (EXIT_OK, "\n")


# --- approvals ------------------------------------------------------------------------------------------------


def test_a_prefix_allow_rule_does_not_cover_a_chained_command(ws: Path) -> None:
    (ws / "y").write_text("keep")
    provider = ScriptedProvider([call("bash", command="make x; rm y"), say("ok")])

    go(ws, provider, allow=("bash(make*)",))

    assert tool_results(provider) == [f"User denied this call. Note: {HEADLESS_DENY}"]
    assert (ws / "y").read_text() == "keep"


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("write", {"path": ".agent/settings.json", "content": "{}"}),
        ("edit", {"path": ".agent/settings.json", "old_string": "a", "new_string": "b"}),
    ],
)
def test_allow_write_and_edit_cannot_change_the_agents_own_settings(ws: Path, tool: str, args: dict) -> None:
    (ws / ".agent").mkdir()
    (ws / ".agent" / "settings.json").write_text("a")
    provider = ScriptedProvider([call(tool, **args), say("ok")])

    go(ws, provider, allow=("write", "edit"))

    assert tool_results(provider) == [f"User denied this call. Note: {HEADLESS_DENY}"]
    assert (ws / ".agent" / "settings.json").read_text() == "a"


@pytest.mark.parametrize(("tool", "args"), [("glob", {"pattern": ".env"}), ("grep", {"pattern": "T", "path": ".env"})])
def test_a_targeted_glob_or_grep_of_a_protected_file_is_refused_even_with_an_allow_flag(
    ws: Path, tool: str, args: dict
) -> None:
    (ws / ".env").write_text("TOKEN=abc")
    provider = ScriptedProvider([call(tool, **args), say("ok")])

    go(ws, provider, allow=("glob", "grep", "read"))

    assert tool_results(provider) == [f"User denied this call. Note: {HEADLESS_DENY}"]


def test_an_allowed_call_is_not_remembered_as_a_grant(ws: Path, spy: Spy) -> None:
    provider = ScriptedProvider([call("bash", command="echo hi"), say("ok")])

    go(ws, provider, allow=("bash(echo hi)",))

    (agent,) = spy.agents
    assert not tool_results(provider)[0].startswith("User denied")
    assert (agent.grants.tools, agent.grants.exact) == (set(), set())


# --- model and sessions ---------------------------------------------------------------------------------------


class _FakeOllama:
    base_url = "http://ollama.test"

    def __init__(self, names: list[str]) -> None:
        self._names = names

    async def models(self) -> list[str]:
        return self._names


def test_a_model_that_is_not_installed_exits_1_with_the_message(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from src.engine.provider.ollama import OllamaProvider

    provider = OllamaProvider(_FakeOllama(["other:latest"]), "ghost", 1000)  # ty: ignore[invalid-argument-type]

    code = go(ws, provider)  # ty: ignore[invalid-argument-type]

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_FAIL, "")
    assert "error: Model ghost is not pulled" in err and "ollama pull ghost" in err


def test_an_ambiguous_resume_prefix_exits_2(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    go(ws, ScriptedProvider([say("one")]), prompt="first")
    go(ws, ScriptedProvider([say("two")]), prompt="second")
    first, second = SessionStore(_sessions(), ws.resolve()).list()
    common = 0
    while first.id[common].upper() == second.id[common].upper():
        common += 1
    prefix = first.id[:common]
    capsys.readouterr()
    if len(prefix) < 4:
        pytest.skip("ids share no usable prefix")  # ids are time-ordered: two runs in one test share their start

    code = go(ws, ScriptedProvider([say("x")]), start=StartChoice("resume", prefix))

    assert code == EXIT_USAGE
    assert "matches 2 sessions" in capsys.readouterr().err


def test_allow_rules_that_can_never_apply_are_warned_about_on_stderr(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = go(ws, ScriptedProvider([say("ok")]), allow=("read(x)", "nosuch", "term_start"))

    err = capsys.readouterr().err
    assert code == EXIT_OK
    assert "warning: --allow read(x): read has no argument to match" in err
    assert "warning: --allow nosuch: no tool named nosuch" in err
    assert "warning: --allow term_start: no tool named term_start" in err  # terminals are off in headless mode
