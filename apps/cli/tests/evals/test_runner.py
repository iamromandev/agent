"""The runner: options, reachability, isolation (AC3), timeout and errors (AC6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from evals.runner import Options, run_all, run_task
from evals.tasks import Task
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


# reachable


# run_task


# run_all


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


async def test_a_failing_git_init_is_a_warning_in_the_run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import subprocess

    def boom(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr("evals.runner.subprocess.run", boom)
    task = Task("w", "hi", lambda r: True, ())
    record = await run_task(task, 1, ScriptedProvider([say("ok")]), _settings(), timeout=30, keep=False)
    assert any("git init" in note for note in record.notes)
    assert "git init" in capsys.readouterr().err
