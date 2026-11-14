from __future__ import annotations

import pytest
from src.agent.env import child_environment


def test_the_child_environment_asks_for_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("KEEP_ME", "yes")

    env = child_environment()

    assert "NO_COLOR" not in env
    assert env["FORCE_COLOR"] == "1"
    assert env["CLICOLOR_FORCE"] == "1"
    assert env["KEEP_ME"] == "yes"
