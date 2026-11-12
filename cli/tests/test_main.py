from __future__ import annotations

from pathlib import Path

import pytest
import src.tui
from src.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_bare_agent_opens_the_tui_in_the_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(src.tui, "run", calls.append)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert calls == [Path.cwd()]


@pytest.mark.parametrize("args", [["--version"], ["--help"], ["health", "--help"]])
def test_other_invocations_do_not_open_the_tui(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(src.tui, "run", calls.append)

    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert calls == []


def test_version_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.stdout.strip() == "0.0.1"
