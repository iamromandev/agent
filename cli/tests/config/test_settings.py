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


def test_session_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_SESSIONS_DIR")
    settings = Settings(_env_file=None)
    assert settings.sessions_dir is None
    assert settings.compact_at == 0.8
    assert settings.compact_keep_turns == 2


def test_session_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SESSIONS_DIR", "/somewhere")
    monkeypatch.setenv("AGENT_COMPACT_AT", "0.5")
    monkeypatch.setenv("AGENT_COMPACT_KEEP_TURNS", "4")
    settings = Settings(_env_file=None)
    assert settings.sessions_dir == "/somewhere"
    assert settings.compact_at == 0.5
    assert settings.compact_keep_turns == 4


@pytest.mark.parametrize("value", ["0", "1.5", "-0.1"])
def test_compact_at_outside_the_range_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AGENT_COMPACT_AT", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_compact_at_of_one_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_COMPACT_AT", "1")
    assert Settings(_env_file=None).compact_at == 1.0


def test_compact_keep_turns_of_zero_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_COMPACT_KEEP_TURNS", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_instructions_share_defaults_to_a_quarter() -> None:
    assert Settings(_env_file=None).instructions_share == 0.25


def test_instructions_share_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_INSTRUCTIONS_SHARE", "0.1")
    share = Settings(_env_file=None).instructions_share
    assert share == 0.1
    assert int(share * 8192) == 819


@pytest.mark.parametrize("value", ["0", "0.6", "abc"])
def test_instructions_share_outside_the_range_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AGENT_INSTRUCTIONS_SHARE", value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_example_lists_the_instructions_share() -> None:
    from pathlib import Path

    example = (Path(__file__).parents[2] / ".env.example").read_text()
    assert "AGENT_INSTRUCTIONS_SHARE=0.25" in example.splitlines()


def test_fetch_defaults_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None)
    assert settings.fetch_timeout == 15.0
    assert settings.fetch_max_bytes == 2_000_000
    monkeypatch.setenv("AGENT_FETCH_TIMEOUT", "3")
    monkeypatch.setenv("AGENT_FETCH_MAX_BYTES", "100")
    settings = Settings(_env_file=None)
    assert (settings.fetch_timeout, settings.fetch_max_bytes) == (3.0, 100)


def test_fetch_limits_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, fetch_timeout=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, fetch_max_bytes=0)
