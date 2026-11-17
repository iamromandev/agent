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


def test_bare_agent_opens_the_tui_in_the_cwd_with_a_new_session(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert opened == [(Path.cwd(), StartChoice("new"))]


@pytest.mark.parametrize("args", [["--version"], ["--help"], ["health", "--help"]])
def test_other_invocations_do_not_open_the_tui(opened: list[tuple[Path, StartChoice]], args: list[str]) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert opened == []


def test_version_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.stdout.strip() == "0.0.1"


def test_continue_starts_with_the_latest_session(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--continue"])

    assert result.exit_code == 0
    assert opened == [(Path.cwd(), StartChoice("continue"))]


def test_resume_without_a_value_opens_the_picker(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--resume"])

    assert result.exit_code == 0
    assert opened == [(Path.cwd(), StartChoice("pick"))]


def test_resume_with_an_id_prefix_resumes_that_session(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--resume", "01JF3K"])

    assert result.exit_code == 0
    assert opened == [(Path.cwd(), StartChoice("resume", "01JF3K"))]


def test_resume_with_an_equals_value_resumes_that_session(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--resume=01JF3K"])

    assert result.exit_code == 0
    assert opened == [(Path.cwd(), StartChoice("resume", "01JF3K"))]


def test_continue_and_resume_together_is_a_usage_error(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--continue", "--resume"])

    assert result.exit_code == 2
    assert "use either --continue or --resume" in result.output
    assert opened == []


def test_continue_and_resume_with_an_id_is_a_usage_error(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--continue", "--resume", "01JF3K"])

    assert result.exit_code == 2
    assert opened == []


def test_resume_without_a_value_before_another_option_still_conflicts(
    opened: list[tuple[Path, StartChoice]],
) -> None:
    result = runner.invoke(app, ["--resume", "--continue"])

    assert result.exit_code == 2
    assert opened == []


def test_help_lists_the_session_flags() -> None:
    result = runner.invoke(app, ["--help"])

    assert "--continue" in result.output
    assert "--resume" in result.output


def test_resume_after_a_subcommand_is_not_rewritten(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["health", "check", "--resume"])

    assert result.exit_code == 2
    assert "No such option: --resume" in result.output
    assert "__pick__" not in result.output
    assert opened == []


def test_resume_after_a_double_dash_is_not_rewritten(opened: list[tuple[Path, StartChoice]]) -> None:
    result = runner.invoke(app, ["--", "--resume"])

    assert "__pick__" not in result.output
    assert opened == []


def test_model_is_passed_to_run(opened: Opened) -> None:
    result = runner.invoke(app, ["--model", "qwen3:14b"])

    assert result.exit_code == 0
    assert opened == [(Path.cwd(), StartChoice("new"))]
    assert opened.models == ["qwen3:14b"]


def test_without_model_run_gets_none(opened: Opened) -> None:
    runner.invoke(app, [])

    assert opened.models == [None]


def test_model_combines_with_continue_and_resume(opened: Opened) -> None:
    runner.invoke(app, ["--model", "a", "--continue"])
    runner.invoke(app, ["--model", "b", "--resume"])
    runner.invoke(app, ["--resume", "01JF3K", "--model", "c"])
    runner.invoke(app, ["--resume", "--model", "d"])

    assert opened == [
        (Path.cwd(), StartChoice("continue")),
        (Path.cwd(), StartChoice("pick")),
        (Path.cwd(), StartChoice("resume", "01JF3K")),
        (Path.cwd(), StartChoice("pick")),
    ]
    assert opened.models == ["a", "b", "c", "d"]


def test_a_blank_model_is_a_usage_error(opened: Opened) -> None:
    result = runner.invoke(app, ["--model", "  "])

    assert result.exit_code != 0
    assert opened == []


def test_model_before_a_subcommand_runs_the_subcommand_not_the_agent(opened: Opened) -> None:
    result = runner.invoke(app, ["--model", "x", "health", "check"])

    assert "No such command" not in result.output
    assert opened == []
    assert opened.models == []


def test_a_subcommand_named_like_the_model_value_is_not_taken_for_the_command(opened: Opened) -> None:
    runner.invoke(app, ["--model", "health", "--continue"])

    assert opened.models == ["health"]


def test_model_with_resume_value_and_continue_conflict_still_errors(opened: Opened) -> None:
    result = runner.invoke(app, ["--model", "x", "--continue", "--resume", "01JF3K"])

    assert result.exit_code != 0
    assert opened == []


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


def test_p_with_a_prompt_runs_headless_not_the_app(headless: Headless, opened: list) -> None:
    result = runner.invoke(app, ["-p", "summarise src"])

    assert result.exit_code == 0
    assert opened == []
    workspace, opts = headless[0]
    assert workspace == Path.cwd()
    assert (opts.prompt, opts.allow, opts.quiet, opts.trust_project, opts.model) == ("summarise src", (), False, False, None)
    assert opts.start == StartChoice("new")


def test_long_print_flag_and_equals_form_give_the_same_prompt(headless: Headless) -> None:
    runner.invoke(app, ["--print", "a b"])
    assert _opts(headless).prompt == "a b"
    runner.invoke(app, ["--print=-dash first"])
    assert _opts(headless).prompt == "-dash first"


def test_p_without_a_prompt_reads_stdin(headless: Headless) -> None:
    assert runner.invoke(app, ["-p"]).exit_code == 0
    assert _opts(headless).prompt is None
    runner.invoke(app, ["-p", "--quiet"])
    assert _opts(headless).prompt is None and _opts(headless).quiet


def test_the_prompt_may_come_after_the_other_flags(headless: Headless) -> None:
    runner.invoke(app, ["-p", "--continue", "now add a test"])

    opts = _opts(headless)
    assert (opts.prompt, opts.start) == ("now add a test", StartChoice("continue"))


def test_allow_is_repeatable_and_keeps_its_value_apart_from_the_prompt(headless: Headless) -> None:
    runner.invoke(app, ["-p", "go", "--allow", "bash(make test)", "--allow", "edit", "--quiet", "--trust-project"])

    opts = _opts(headless)
    assert (opts.prompt, opts.allow, opts.quiet, opts.trust_project) == ("go", ("bash(make test)", "edit"), True, True)


def test_p_combines_with_model_and_resume_by_id(headless: Headless) -> None:
    runner.invoke(app, ["-p", "go", "--model", " qwen ", "--resume", "01JF3K"])

    opts = _opts(headless)
    assert (opts.model, opts.start) == ("qwen", StartChoice("resume", "01JF3K"))


def test_p_with_a_bare_resume_is_a_usage_error(headless: Headless) -> None:
    result = runner.invoke(app, ["-p", "go", "--resume"])

    assert result.exit_code == 2
    assert headless == []


@pytest.mark.parametrize("flag", [["--allow", "bash"], ["--quiet"], ["--trust-project"]])
def test_headless_only_flags_need_p(headless: Headless, opened: list, flag: list[str]) -> None:
    result = runner.invoke(app, flag)

    assert result.exit_code == 2
    assert "needs -p" in result.output
    assert headless == [] and opened == []


def test_the_exit_code_of_the_headless_run_is_the_exit_code_of_the_command(
    headless: Headless, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.headless.run as headless_run

    monkeypatch.setattr(headless_run, "run_headless", lambda *a, **k: 130)

    assert runner.invoke(app, ["-p", "x"]).exit_code == 130


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


def test_output_defaults_to_text_and_is_passed_to_headless(headless: Headless) -> None:
    runner.invoke(app, ["-p", "go"])
    assert _opts(headless).output == "text"
    runner.invoke(app, ["-p", "go", "--output", "text"])
    assert _opts(headless).output == "text"


@pytest.mark.parametrize("value", ["json", "stream-json"])
def test_output_json_modes_are_passed_to_headless(headless: Headless, value: str) -> None:
    assert runner.invoke(app, ["-p", "go", "--output", value]).exit_code == 0
    assert _opts(headless).output == value


def test_an_unknown_output_value_is_a_usage_error(headless: Headless) -> None:
    result = runner.invoke(app, ["-p", "go", "--output", "xml"])

    assert result.exit_code == 2
    assert headless == []


def test_output_json_without_p_is_a_usage_error(headless: Headless, opened: list) -> None:
    result = runner.invoke(app, ["--output", "json"])

    assert result.exit_code == 2
    assert opened == [] and headless == []


def test_output_text_without_p_is_accepted(opened: list) -> None:
    assert runner.invoke(app, ["--output", "text"]).exit_code == 0
    assert len(opened) == 1
