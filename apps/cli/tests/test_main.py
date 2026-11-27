from __future__ import annotations

from pathlib import Path

import pytest
import src.tui
import typer.rich_utils
from src.main import app
from src.tui import StartChoice
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def plain_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typer forces colour at import when GITHUB_ACTIONS/FORCE_COLOR is set, which splits the
    text the tests look for with ANSI codes; render plain so the tests read the same everywhere."""
    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", False)


class Opened(list):
    """What `src.tui.run` was called with: `(workspace, start)` items, and the `--model` values in `models`."""

    models: list[str | None]


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[Path, StartChoice]]:
    """What `src.tui.run` was called with."""
    calls = Opened()
    calls.models = []

    def fake_run(workspace: Path, start: StartChoice, model: str | None = None) -> None:
        calls.append((workspace, start))
        calls.models.append(model)

    monkeypatch.setattr(src.tui, "run", fake_run)
    monkeypatch.chdir(tmp_path)
    return calls


def test_version_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.stdout.strip() == "0.0.1"


def test_resume_without_a_value_opens_the_picker(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--resume"])

    assert result.exit_code == 0
    assert opened == [(Path.cwd(), StartChoice("pick"))]


def test_resume_with_an_id_prefix_resumes_that_session(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--resume", "01JF3K"])

    assert result.exit_code == 0
    assert opened == [(Path.cwd(), StartChoice("resume", "01JF3K"))]


def test_resume_without_a_value_before_another_option_still_conflicts(
    opened: list[tuple[Path, StartChoice]],
) -> None:
    result = runner.invoke(app, ["--resume", "--continue"])

    assert result.exit_code == 2
    assert opened == []


def test_without_model_run_gets_none(opened: Opened) -> None:
    runner.invoke(app, [])

    assert opened.models == [None]


def test_a_blank_model_is_a_usage_error(opened: Opened) -> None:
    result = runner.invoke(app, ["--model", "  "])

    assert result.exit_code != 0
    assert opened == []


def test_model_before_a_subcommand_runs_the_subcommand_not_the_agent(opened: Opened) -> None:
    result = runner.invoke(app, ["--model", "x", "health", "check"])

    assert "No such command" not in result.output
    assert opened == []
    assert opened.models == []


class Headless(list):
    """What `run_headless` was called with, as `(workspace, options)`; the TUI must not have been opened."""


@pytest.fixture
def headless(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, opened: list[tuple[Path, StartChoice]]) -> Headless:
    import src.headless.run as headless_run
    from src.headless import HeadlessOptions

    calls = Headless()

    def fake(workspace: Path, opts: HeadlessOptions, **_: object) -> int:
        calls.append((workspace, opts))
        return 0

    monkeypatch.setattr(headless_run, "run_headless", fake)
    return calls


def _opts(headless: Headless):
    (_, opts) = headless[-1]
    return opts


def test_the_prompt_may_come_after_the_other_flags(headless: Headless) -> None:
    runner.invoke(app, ["-p", "--continue", "now add a test"])

    opts = _opts(headless)
    assert (opts.prompt, opts.start) == ("now add a test", StartChoice("continue"))


def test_p_with_a_bare_resume_is_a_usage_error(headless: Headless) -> None:
    result = runner.invoke(app, ["-p", "go", "--resume"])

    assert result.exit_code == 2
    assert headless == []


def test_extra_words_after_the_prompt_are_a_usage_error_naming_the_problem(headless: Headless) -> None:
    result = runner.invoke(app, ["-p", "summarise", "the", "code"])

    assert result.exit_code == 2
    assert "the" in result.output and "quote" in result.output
    assert headless == []


def test_a_prompt_after_double_dash_is_a_usage_error_naming_the_problem(headless: Headless) -> None:
    result = runner.invoke(app, ["-p", "--", "x"])

    assert result.exit_code == 2
    assert "--" in result.output and "-p" in result.output and "prompt" in result.output
    assert headless == []


def test_an_unknown_output_value_is_a_usage_error(headless: Headless) -> None:
    result = runner.invoke(app, ["-p", "go", "--output", "xml"])

    assert result.exit_code == 2
    assert headless == []


def test_output_json_without_p_is_a_usage_error(headless: Headless, opened: list) -> None:
    result = runner.invoke(app, ["--output", "json"])

    assert result.exit_code == 2
    assert opened == [] and headless == []


class DaemonCalls(list):
    """What `core.server.serve.serve` and `stop` were called with, as `(name, workspace)`; `running` is what `stop`
    answers."""

    running = True


@pytest.fixture
def daemon_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DaemonCalls:
    import core.server.serve as served

    calls = DaemonCalls()

    def fake_serve(workspace: Path) -> int:
        calls.append(("serve", workspace))
        return 1

    def fake_stop(workspace: Path) -> bool:
        calls.append(("stop", workspace))
        return calls.running

    monkeypatch.setattr(served, "serve", fake_serve)
    monkeypatch.setattr(served, "stop", fake_stop)
    monkeypatch.chdir(tmp_path)
    return calls


def test_serve_runs_the_daemon_for_this_folder_and_passes_its_exit_code(
    daemon_calls: DaemonCalls, opened: Opened, tmp_path: Path
) -> None:
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1 and daemon_calls == [("serve", tmp_path)] and opened == []

    other = tmp_path / "other"
    other.mkdir()
    runner.invoke(app, ["serve", "--workspace", str(other)])
    assert daemon_calls[-1] == ("serve", other)


def test_stop_says_when_no_daemon_runs_and_exits_0(daemon_calls: DaemonCalls, tmp_path: Path) -> None:
    daemon_calls.running = False
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0 and "no agent daemon for this folder" in result.output
    assert daemon_calls == [("stop", tmp_path)]


def test_a_word_after_resume_is_its_value_even_when_it_is_serve(daemon_calls: DaemonCalls, opened: Opened) -> None:
    runner.invoke(app, ["--resume", "serve"])
    assert daemon_calls == [] and opened  # `serve` is the session id here: the agent opened, no daemon started
