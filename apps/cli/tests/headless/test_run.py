"""`run_headless` end to end with a scripted provider: stdout, stderr, exit codes, approvals, sessions, hooks."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from core.config.settings import get_settings
from core.engine.provider.base import Done, ProviderError, TextDelta, ToolCall, ToolCallEvent
from core.engine.session import SessionStore
from core.runtime import StartChoice
from core.testing.fakes import ScriptedProvider
from core.testing.helpers import session_held_elsewhere
from src.headless import EXIT_FAIL, EXIT_OK, EXIT_USAGE, HEADLESS_DENY, HeadlessOptions
from src.headless.run import run_headless

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


def test_a_provider_error_exits_1_with_its_message_and_empty_stdout(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = go(ws, ScriptedProvider([[ProviderError("model fell over")]]))

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_FAIL, "")
    assert "error: model fell over" in err


def test_an_invalid_allow_rule_exits_2_naming_it_before_anything_runs(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ScriptedProvider([say("x")])

    code = go(ws, provider, allow=("bash(a*b)",))

    assert code == EXIT_USAGE
    assert "bash(a*b)" in capsys.readouterr().err
    assert provider.requests == []


def _sessions() -> Path:
    from core.engine.session import sessions_root

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


def test_headless_mode_never_touches_the_daemon_run_folder(ws: Path) -> None:
    import os

    assert go(ws, ScriptedProvider([say("one")])) == EXIT_OK
    assert not (Path(os.environ["XDG_DATA_HOME"]) / "agent" / "run").exists()


def test_resume_of_a_session_open_in_another_process_exits_2(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    go(ws, ScriptedProvider([say("one")]), prompt="first")
    (info,) = SessionStore(_sessions(), ws.resolve()).list()
    capsys.readouterr()
    provider = ScriptedProvider([say("x")])

    with session_held_elsewhere(info.path.parent, info.id):
        code = go(ws, provider, start=StartChoice("resume", info.id[:8]))

    assert code == EXIT_USAGE and provider.requests == []
    assert f"session {info.id} is open in another agent process" in capsys.readouterr().err


class _Closed:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None: ...


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


# --- approvals ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("tool", "args"), [("glob", {"pattern": ".env"})])
def test_a_targeted_glob_or_grep_of_a_protected_file_is_refused_even_with_an_allow_flag(
    ws: Path, tool: str, args: dict
) -> None:
    (ws / ".env").write_text("TOKEN=abc")
    provider = ScriptedProvider([call(tool, **args), say("ok")])

    go(ws, provider, allow=("glob", "grep", "read"))

    assert tool_results(provider) == [f"User denied this call. Note: {HEADLESS_DENY}"]


# --- model and sessions ---------------------------------------------------------------------------------------


class _FakeOllama:
    base_url = "http://ollama.test"

    def __init__(self, names: list[str]) -> None:
        self._names = names

    async def models(self) -> list[str]:
        return self._names


def test_a_model_that_is_not_installed_exits_1_with_the_message(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from core.engine.provider.ollama import OllamaProvider

    provider = OllamaProvider(_FakeOllama(["other:latest"]), "ghost", 1000)  # ty: ignore[invalid-argument-type]

    code = go(ws, provider)  # ty: ignore[invalid-argument-type]

    out, err = capsys.readouterr()
    assert (code, out) == (EXIT_FAIL, "")
    assert "error: Model ghost is not pulled" in err and "ollama pull ghost" in err


def test_a_session_error_with_a_file_derived_id_is_written_out_escaped(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from core.engine.session import SessionError

    go(ws, ScriptedProvider([say("one")]), prompt="first")
    (info,) = SessionStore(_sessions(), ws.resolve()).list()

    def bad_load(self: SessionStore, session_id: str) -> None:  # what `read_header` says of a renamed file
        raise SessionError(f"Session {session_id} was copied or renamed: its header says evil\x1b[2J")

    monkeypatch.setattr(SessionStore, "load", bad_load)
    capsys.readouterr()

    code = go(ws, ScriptedProvider([say("x")]), start=StartChoice("resume", info.id[:8]))

    err = capsys.readouterr().err
    assert code == EXIT_USAGE, err
    assert "\x1b" not in err and "evil\\x1b[2J" in err


def test_mcp_server_names_and_failure_reasons_are_written_out_escaped(
    ws: Path, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from core.testing.fakes import FakeConnector

    (env / "agent").mkdir(parents=True)
    (env / "agent" / "mcp.json").write_text(json.dumps({"servers": {"srv": {"command": "x"}}}))
    connector = FakeConnector(failures={"srv": RuntimeError("bad\x1b[2J")})

    run_headless(ws, HeadlessOptions("hi"), provider=ScriptedProvider([say("ok")]), connector=connector)

    err = capsys.readouterr().err
    assert "\x1b" not in err and "bad\\x1b[2J" in err


# --- stdin, built-ins, project trust, MCP -----------------------------------------------------------------------


def _trust_files() -> list[Path]:
    import os

    return sorted(Path(os.environ["XDG_DATA_HOME"]).glob("agent/*trust*.json"))


def _project(ws: Path, hook_log: Path) -> None:
    (ws / ".agent").mkdir()
    (ws / ".agent" / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {"allow": ["bash(echo hi > made.txt)"]},
                "hooks": {"UserPromptSubmit": [{"command": f"echo ran >> {hook_log}"}]},
            }
        )
    )
    (ws / ".agent" / "mcp.json").write_text(json.dumps({"servers": {"proj": {"command": "x"}}}))


def test_an_untrusted_project_warns_once_per_kind_and_none_of_it_is_active(
    ws: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from core.testing.fakes import FakeConnector

    log = tmp_path / "hook.log"
    _project(ws, log)
    connector = FakeConnector()
    provider = ScriptedProvider([call("bash", command="echo hi > made.txt"), say("ok")])

    code = run_headless(ws, HeadlessOptions("hi"), provider=provider, connector=connector)

    err = capsys.readouterr().err
    assert code == EXIT_OK
    assert [line for line in err.splitlines() if "untrusted" in line and "--trust-project" in line].__len__() == 3
    for kind in ("allow rule", "hook", "MCP server"):
        assert sum(kind in line and "--trust-project" in line for line in err.splitlines()) == 1
    assert tool_results(provider)[0].startswith("User denied")
    assert not log.exists() and not (ws / "made.txt").exists()
    assert connector.connects == []
    assert _trust_files() == []


def test_trust_project_activates_rules_hooks_and_servers_without_writing_a_trust_store(
    ws: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from core.testing.fakes import FakeConnector

    log = tmp_path / "hook.log"
    _project(ws, log)
    connector = FakeConnector()
    provider = ScriptedProvider([call("bash", command="echo hi > made.txt"), say("ok")])

    code = run_headless(ws, HeadlessOptions("hi", trust_project=True), provider=provider, connector=connector)

    err = capsys.readouterr().err
    assert code == EXIT_OK
    assert "--trust-project" not in err
    assert not tool_results(provider)[0].startswith("User denied")
    assert (ws / "made.txt").exists() and log.read_text() == "ran\n"
    assert [name for name, _ in connector.connects] == ["proj"]
    assert _trust_files() == []


def test_a_stdin_without_a_byte_buffer_is_no_input_not_a_crash(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from io import StringIO

    monkeypatch.setattr(sys, "stdin", StringIO("ignored"))

    code = run_headless(ws, HeadlessOptions(None), provider=ScriptedProvider([say("x")]))

    assert code == EXIT_USAGE
    assert "the prompt is empty" in capsys.readouterr().err


def test_a_sigterm_during_the_runtime_build_exits_143_and_restores_the_handler(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import signal
    import time

    import src.headless.run as headless_run

    def slow_build(*_: object, **__: object):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(2)  # the handler raises before this ends
        raise AssertionError("SIGTERM was ignored")

    monkeypatch.setattr(headless_run, "build_runtime", slow_build)
    before = signal.getsignal(signal.SIGTERM)

    code = go(ws, ScriptedProvider([say("x")]))

    assert code == 143
    assert signal.getsignal(signal.SIGTERM) is before


def test_an_agent_file_warning_is_shown_at_launch_and_valid_agents_are_offered(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = ws / ".agent" / "agents"
    folder.mkdir(parents=True)
    (folder / "bad.md").write_text("no front matter")
    (folder / "reviewer.md").write_text("---\ndescription: Reviews code\n---\nYou review.\n")
    provider = ScriptedProvider([call("task", description="d", prompt="p", agent="reviewer"), say("report"), say("done")])

    code = go(ws, provider)

    assert code == EXIT_OK
    assert "bad.md" in capsys.readouterr().err
    assert "- reviewer: Reviews code" in provider.requests[0][0][0].content
    child_system = provider.requests[1][0][0].content
    assert child_system.startswith("You review.")
    assert "Available agents" not in child_system


def _writer_agent(ws: Path, tools: str = "edit, bash") -> None:
    folder = ws / ".agent" / "agents"
    folder.mkdir(parents=True)
    (folder / "w.md").write_text(f"---\ndescription: Edits files\ntools: {tools}\n---\nYou edit.\n")
    (ws / "a.txt").write_text("alpha\n")


def test_ac22_an_allow_flag_lets_a_custom_childs_edit_run_and_its_bash_is_refused(ws: Path) -> None:
    _writer_agent(ws)
    provider = ScriptedProvider(
        [
            call("task", "t1", description="d", prompt="p", agent="w"),
            call("edit", "e1", path="a.txt", old_string="alpha", new_string="beta"),
            call("bash", "b1", command="echo hi"),
            say("report"),
            say("done"),
        ]
    )

    code = go(ws, provider, allow=("edit",))

    assert code == EXIT_OK
    assert (ws / "a.txt").read_text() == "beta\n"
    child_results = tool_results(provider, 3)
    assert HEADLESS_DENY in child_results[-1]  # the child's bash, refused like the parent's would be
    assert "report" in tool_results(provider)[-1]


def test_a_builtin_command_prompt_such_as_agents_is_exit_2(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([say("x")])

    code = go(ws, provider, prompt="/agents")

    assert code == EXIT_USAGE
    assert "/agents" in capsys.readouterr().err
    assert provider.requests == []


def test_workflow_prompt_is_exit_2_like_other_builtins(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([say("x")])

    code = go(ws, provider, prompt="/workflow")

    assert code == EXIT_USAGE
    assert "/workflow" in capsys.readouterr().err
    assert provider.requests == []


def test_images_prompt_is_exit_2_like_other_builtins(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([say("x")])

    code = go(ws, provider, prompt="/images")

    assert code == EXIT_USAGE
    assert "/images" in capsys.readouterr().err
    assert provider.requests == []


def test_headless_never_offers_generate_image_even_when_mflux_is_found(
    ws: Path, spy: Spy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.testing.image_fakes import fake_mflux

    command = fake_mflux(tmp_path)
    monkeypatch.setattr("core.engine.imagegen.find_mflux", lambda: command)
    provider = ScriptedProvider([say("x")])

    assert go(ws, provider) == EXIT_OK

    assert "generate_image" not in provider.requests[0][1]
