from __future__ import annotations

import pytest
from src.engine.process import shell_argv


def test_the_shell_is_the_users_with_a_c_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/zsh")

    assert shell_argv("ls -l") == ["/bin/zsh", "-c", "ls -l"]


@pytest.mark.parametrize("value", [None, ""])
def test_the_shell_falls_back_to_bin_sh(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("SHELL", raising=False)
    else:
        monkeypatch.setenv("SHELL", value)

    assert shell_argv("x") == ["/bin/sh", "-c", "x"]
