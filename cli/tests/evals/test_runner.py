"""The runner: options, reachability, isolation (AC3), timeout and errors (AC6)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from evals.runner import FIXTURE, Options, options_from_env, reachable, run_all, run_task
from evals.tasks import TASKS, Task
from src.client import OllamaUnreachable
from src.config.settings import Settings

from tests.evals.conftest import call, say
from tests.fakes import ScriptedProvider

FIND_ANSWER = "parse_config is in app/config.py, called by run and reload"


def _settings() -> Settings:
    return Settings()


def _snapshot(root: Path) -> dict[str, tuple[bool, float]]:
    return {str(p.relative_to(root)): (p.is_dir(), p.stat().st_mtime_ns) for p in root.rglob("*")}


def _find_script() -> list[list]:
    return [call("task", description="look", prompt="find parse_config"), say("report: app/config.py"), say(FIND_ANSWER)]


class FakeClient:
    def __init__(self, models: list[str] | Exception) -> None:
        self._models = models

    async def models(self) -> list[str]:
        if isinstance(self._models, Exception):
            raise self._models
        return self._models


# options


def test_options_defaults() -> None:
    assert options_from_env({}) == Options(runs=3, tasks=(), timeout=300.0, keep=False)


def test_options_read_the_environment() -> None:
    env = {"EVAL_RUNS": "2", "EVAL_TASKS": "find, rules", "EVAL_TIMEOUT": "5.5", "EVAL_KEEP": "1"}
    assert options_from_env(env) == Options(runs=2, tasks=("find", "rules"), timeout=5.5, keep=True)


@pytest.mark.parametrize("env", [{"EVAL_RUNS": "0"}, {"EVAL_RUNS": "x"}, {"EVAL_TIMEOUT": "-1"}, {"EVAL_TIMEOUT": "y"}])
def test_invalid_options_raise(env: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        options_from_env(env)


# reachable


async def test_reachable_is_none_when_the_model_is_there() -> None:
    assert await reachable(FakeClient(["qwen:8b"]), "qwen:8b", "http://x") is None


async def test_reachable_accepts_a_model_without_tag() -> None:
    assert await reachable(FakeClient(["llama3:latest"]), "llama3", "http://x") is None


async def test_reachable_says_when_ollama_is_down() -> None:
    line = await reachable(FakeClient(OllamaUnreachable("down")), "m", "http://localhost:11434")
    assert line == "eval skipped: Ollama not reachable at http://localhost:11434"


async def test_reachable_says_when_the_model_is_missing() -> None:
    line = await reachable(FakeClient(["other:1b"]), "m:1", "http://x")
    assert line == "eval skipped: model m:1 is not available in Ollama at http://x"


# run_task


async def test_a_scripted_find_run_passes_and_is_recorded() -> None:
    provider = ScriptedProvider(_find_script(), context_window=8192)
    record = await run_task(TASKS["find"], 1, provider, _settings(), timeout=30, keep=False)
    assert record.passed, record.reasons
    assert (record.task, record.run) == ("find", 1)
    assert record.steps == 2
    assert record.tokens == 45
    assert record.tools["task"] == 1
    assert record.seconds > 0
    assert record.first_prompt == TASKS["find"].prompt


async def test_the_temp_dir_is_removed_and_the_fixture_is_untouched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    before = _snapshot(FIXTURE)
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    write = call("write", path="x.txt", content="hi")
    task = Task("w", "write a file", lambda r: True, (), setup=None)
    record = await run_task(task, 1, ScriptedProvider([write, say("ok")]), _settings(), timeout=30, keep=False)
    assert record.passed
    assert list(tmp_root.iterdir()) == []
    assert _snapshot(FIXTURE) == before


async def test_keep_leaves_the_workspace_and_prints_its_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    task = Task("w", "write a file", lambda r: True, ())
    await run_task(task, 1, ScriptedProvider([call("write", path="x.txt", content="hi"), say("ok")]), _settings(), timeout=30, keep=True)
    kept = list(tmp_path.glob("agent-eval-w-*"))
    assert len(kept) == 1
    assert (kept[0] / "ws" / "x.txt").read_text() == "hi"
    assert str(kept[0]) in capsys.readouterr().out


async def test_home_and_xdg_are_neither_written_nor_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    (sentinel / ".config" / "agent").mkdir(parents=True)
    # a real-looking global rule that would deny `write` if it were read
    (sentinel / ".config" / "agent" / "settings.json").write_text('{"permissions": {"deny": ["write"]}}')
    monkeypatch.setenv("HOME", str(sentinel))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(sentinel / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(sentinel / ".local" / "share"))
    before = _snapshot(sentinel)
    task = Task("w", "write a file", lambda r: True, ())
    provider = ScriptedProvider([call("write", path="x.txt", content="hi"), say("ok")])
    record = await run_task(task, 1, provider, _settings(), timeout=30, keep=False)
    assert not record.tool_results[0][2].startswith("denied")
    assert _snapshot(sentinel) == before
    assert not (sentinel / ".local").exists()


async def test_os_environ_is_not_changed() -> None:
    before = dict(os.environ)
    await run_task(TASKS["find"], 1, ScriptedProvider(_find_script()), _settings(), timeout=30, keep=False)
    assert dict(os.environ) == before


async def test_a_slow_provider_times_out_and_is_recorded() -> None:
    provider = ScriptedProvider(_find_script(), delay=0.5)
    record = await run_task(TASKS["find"], 1, provider, _settings(), timeout=0.05, keep=False)
    assert record.error == "timeout"
    assert not record.passed
    assert "timeout" in record.reasons


async def test_a_provider_error_is_recorded() -> None:
    from src.engine.provider.base import ProviderError

    record = await run_task(TASKS["find"], 1, ScriptedProvider([[ProviderError("model gone")]]), _settings(), timeout=30, keep=False)
    assert record.error == "model gone"
    assert not record.passed


async def test_a_crash_in_one_run_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        model = "x"
        context_window = 10

        def stream(self, messages: object, tools: object) -> object:
            raise RuntimeError("kaput")

    record = await run_task(TASKS["find"], 1, Boom(), _settings(), timeout=30, keep=False)  # ty: ignore[invalid-argument-type]
    assert record.error is not None and "kaput" in record.error


async def test_a_command_task_expands_the_command() -> None:
    provider = ScriptedProvider([say("the function average divides by len(xs) + 1")])
    record = await run_task(TASKS["command"], 1, provider, _settings(), timeout=30, keep=False)
    assert record.passed, record.reasons
    assert "Review app/calc.py for bugs" in record.first_prompt


async def test_an_unknown_command_is_an_error() -> None:
    task = Task("c", "/nope x", lambda r: False, ())
    record = await run_task(task, 1, ScriptedProvider([]), _settings(), timeout=30, keep=False)
    assert record.error is not None and "nope" in record.error


# run_all


async def test_run_all_runs_each_selected_task_the_given_number_of_times() -> None:
    made: list[ScriptedProvider] = []

    def factory() -> ScriptedProvider:
        made.append(ScriptedProvider(_find_script()))
        return made[-1]

    records = await run_all(Options(runs=2, tasks=("find",), timeout=30), factory, _settings())
    assert [(r.task, r.run) for r in records] == [("find", 1), ("find", 2)]
    assert len(made) == 2


async def test_a_timed_out_run_does_not_stop_the_next() -> None:
    providers = iter([ScriptedProvider(_find_script(), delay=0.5), ScriptedProvider(_find_script())])
    records = await run_all(Options(runs=2, tasks=("find",), timeout=0.2), lambda: next(providers), _settings())
    assert [r.error for r in records] == ["timeout", None]
    assert records[1].passed


async def test_run_all_appends_to_the_given_list_even_when_interrupted() -> None:
    into: list = []

    def factory() -> ScriptedProvider:
        if into:
            raise KeyboardInterrupt
        return ScriptedProvider(_find_script())

    with pytest.raises(KeyboardInterrupt):
        await run_all(Options(runs=2, tasks=("find",), timeout=30), factory, _settings(), into)
    assert len(into) == 1


def test_fixture_exists() -> None:
    assert (FIXTURE / "app" / "calc.py").is_file()


async def test_a_timeout_raised_inside_the_turn_is_an_error_not_the_run_timeout() -> None:
    class Inner:
        model = "x"
        context_window = 10

        def stream(self, messages: object, tools: object) -> object:
            raise TimeoutError("socket timed out")

    record = await run_task(TASKS["find"], 1, Inner(), _settings(), timeout=30, keep=False)  # ty: ignore[invalid-argument-type]
    assert record.error is not None and "socket timed out" in record.error
    assert "timeout" not in record.reasons


async def test_an_interrupted_run_is_kept_in_the_given_list() -> None:
    class Stop:
        model = "x"
        context_window = 10

        def stream(self, messages: object, tools: object) -> object:
            raise KeyboardInterrupt

    into: list = []
    with pytest.raises(KeyboardInterrupt):
        await run_all(Options(runs=1, tasks=("find",), timeout=30), lambda: Stop(), _settings(), into)  # ty: ignore[invalid-argument-type]
    assert len(into) == 1
    assert into[0].error == "interrupted"
    assert not into[0].passed


async def test_a_missing_python_is_a_harness_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.runner.shutil.which", lambda name, *a, **k: None if name == "python" else "/bin/" + name)
    record = await run_task(TASKS["rules"], 1, ScriptedProvider([say("x")]), _settings(), timeout=30, keep=False)
    assert record.error is not None and record.error.startswith("harness:")


async def test_a_failing_git_init_is_a_warning_in_the_run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import subprocess

    def boom(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr("evals.runner.subprocess.run", boom)
    task = Task("w", "hi", lambda r: True, ())
    record = await run_task(task, 1, ScriptedProvider([say("ok")]), _settings(), timeout=30, keep=False)
    assert any("git init" in note for note in record.notes)
    assert "git init" in capsys.readouterr().err


async def test_without_usage_steps_come_from_the_history_and_tokens_stay_unknown() -> None:
    from src.engine.provider.base import Done, TextDelta

    record = await run_task(TASKS["command"], 1, ScriptedProvider([[TextDelta("average"), Done(0, 0)]]), _settings(), timeout=30, keep=False)
    assert record.steps == 1
    assert record.tokens is None


def test_the_eval_agent_gets_a_real_guard_built_from_the_isolated_home(tmp_path: Path) -> None:
    from evals.record import RunRecord
    from evals.runner import _build_agent, _prepare

    task = TASKS["find"]
    record = RunRecord(task.name, 1)
    workspace, env = _prepare(task, tmp_path, record)

    agent = _build_agent(task, ScriptedProvider([]), _settings(), workspace, env, record)

    guard = agent.ctx.guard
    assert guard is not None
    home = Path(env["HOME"])
    assert guard.paths.home_folders == [home]  # the isolated home, not the real one
    assert guard.paths.is_protected(home / ".ssh" / "id_rsa")
    assert guard.paths.is_protected(workspace / ".env")
