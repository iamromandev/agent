from __future__ import annotations

from typing import Any

import pytest
from src.client import ApiError
from src.command import health
from src.main import app
from typer.testing import CliRunner

runner = CliRunner()


class _Stub:
    def __init__(self, report: dict[str, Any] | None = None, error: str | None = None) -> None:
        self._report = report or {}
        self._error = error

    def health(self) -> dict[str, Any]:
        if self._error:
            raise ApiError(self._error)
        return self._report


def test_check_prints_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {"version": "1.2.3", "environment": "local", "uptime": "5s", "db": {"status": "success", "version": "18"}}
    monkeypatch.setattr(health, "get_api_client", lambda: _Stub(report))

    result = runner.invoke(app, ["health", "check"])

    assert result.exit_code == 0
    assert "1.2.3" in result.output
    assert "success (18)" in result.output


def test_check_exits_non_zero_when_the_api_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "get_api_client", lambda: _Stub(error="Cannot reach http://x"))

    result = runner.invoke(app, ["health", "check"])

    assert result.exit_code == 1
    assert "Cannot reach" in result.output
