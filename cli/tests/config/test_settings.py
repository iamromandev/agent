from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from src.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.model == "qwen2.5:7b-instruct"
    assert settings.num_ctx == 8192
    assert settings.bash_timeout == 120.0
    assert settings.max_steps == 25


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODEL", "foo")
    monkeypatch.setenv("AGENT_NUM_CTX", "4096")
    settings = get_settings()
    assert settings.model == "foo"
    assert settings.num_ctx == 4096


@pytest.mark.parametrize("name", ["AGENT_NUM_CTX", "AGENT_BASH_TIMEOUT", "AGENT_MAX_STEPS"])
def test_non_positive_values_are_rejected(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
