"""`main`: the skip line (AC1), exit codes (AC8), the report on disk and the output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.runner import main
from src.client import OllamaUnreachable

from tests.evals.conftest import call, say
from tests.fakes import ScriptedProvider


class FakeClient:
    def __init__(self, models: list[str] | Exception) -> None:
        self._models = models

    async def models(self) -> list[str]:
        if isinstance(self._models, Exception):
            raise self._models
        return self._models

    async def aclose(self) -> None:
        pass


def _find_provider() -> ScriptedProvider:
    return ScriptedProvider(
        [call("task", description="look", prompt="p"), say("r"), say("app/config.py: run, reload")], context_window=8192
    )


def test_main_prints_the_skip_line_and_exits_0_when_ollama_is_down(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_OLLAMA_URL", "http://localhost:11434")  # Settings reads the real environment and .env
    code = main([], env={}, client=FakeClient(OllamaUnreachable("x")), report_dir=tmp_path)
    assert code == 0
    out = capsys.readouterr().out
    assert out.strip() == "eval skipped: Ollama not reachable at http://localhost:11434"
    assert list(tmp_path.iterdir()) == []


def test_main_reports_and_exits_0_even_when_runs_fail(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    env = {"EVAL_RUNS": "2", "EVAL_TASKS": "find"}
    providers = iter([_find_provider(), ScriptedProvider([say("no idea")])])
    code = main([], env=env, client=FakeClient(["qwen2.5:7b-instruct"]), provider_factory=lambda: next(providers), report_dir=tmp_path)
    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("model qwen2.5:7b-instruct · ctx 8192 · 2 runs/task")
    assert "find" in out and "1/2" in out
    (path,) = tmp_path.glob("*.json")
    assert f"report: {path}" in out
    data = json.loads(path.read_text())
    assert [r["passed"] for r in data["results"]] == [True, False]
    assert data["model"] == "qwen2.5:7b-instruct" and data["commit"]


def test_main_exits_2_for_a_bad_option(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main([], env={"EVAL_RUNS": "zero"}, client=FakeClient([]), report_dir=tmp_path)
    assert code == 2
    assert "EVAL_RUNS" in capsys.readouterr().err


def test_main_exits_2_for_an_unknown_task(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main([], env={"EVAL_TASKS": "find,nope"}, client=FakeClient([]), report_dir=tmp_path)
    assert code == 2
    assert "nope" in capsys.readouterr().err


def test_main_writes_a_partial_report_and_exits_130_on_ctrl_c(tmp_path: Path) -> None:
    calls = 0

    def factory() -> ScriptedProvider:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _find_provider()

    env = {"EVAL_RUNS": "3", "EVAL_TASKS": "find"}
    code = main([], env=env, client=FakeClient(["qwen2.5:7b-instruct"]), provider_factory=factory, report_dir=tmp_path)
    assert code == 130
    (path,) = tmp_path.glob("*.json")
    assert len(json.loads(path.read_text())["results"]) == 1


def test_main_keeps_the_in_flight_run_when_interrupted_mid_run(tmp_path: Path) -> None:
    class Stop:
        model = "m"
        context_window = 8192

        def stream(self, messages: object, tools: object) -> object:
            raise KeyboardInterrupt

    env = {"EVAL_RUNS": "2", "EVAL_TASKS": "find"}
    code = main([], env=env, client=FakeClient(["qwen2.5:7b-instruct"]), provider_factory=lambda: Stop(), report_dir=tmp_path)  # ty: ignore[invalid-argument-type]
    assert code == 130
    (path,) = tmp_path.glob("*.json")
    (result,) = json.loads(path.read_text())["results"]
    assert result["error"] == "interrupted" and result["passed"] is False
