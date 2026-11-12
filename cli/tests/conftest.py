from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class NetworkAccessError(AssertionError):
    """A test tried to use a real network transport."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any real HTTP call. Tests use httpx.MockTransport instead (NFR2)."""

    async def deny_async(self: Any, request: httpx.Request) -> httpx.Response:
        raise NetworkAccessError(f"real network call blocked: {request.url}")

    def deny_sync(self: Any, request: httpx.Request) -> httpx.Response:
        raise NetworkAccessError(f"real network call blocked: {request.url}")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_sync)


@pytest.fixture
def fixture_path() -> Callable[[str], Path]:
    def resolve(name: str) -> Path:
        return FIXTURES / name

    return resolve
