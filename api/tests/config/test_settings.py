from __future__ import annotations

import pytest
from src.config import Settings

_REQUIRED = {
    "ENV": "local",
    "DEBUG": "false",
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_NAME": "test",
    "DB_USER": "test",
    "DB_PASSWORD": "test",
}


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name, value in (_REQUIRED | env).items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_comma_lists_are_split_and_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch, CORS_ORIGINS=" http://a.test , ,http://b.test", TRUSTED_PROXIES="10.0.0.0/8, 127.0.0.1"
    )

    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]
    assert settings.trusted_proxy_list == ["10.0.0.0/8", "127.0.0.1"]


def test_blank_lists_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, CORS_ORIGINS="", TRUSTED_PROXIES="")

    assert settings.cors_origin_list == []
    assert settings.trusted_proxy_list == []
