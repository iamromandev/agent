from __future__ import annotations

import httpx
import pytest
from src.client import ApiClient, ApiError


def _client(handler) -> ApiClient:
    return ApiClient(base_url="http://api.test", timeout=1, transport=httpx.MockTransport(handler))


def test_health_returns_the_envelope_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health/check"
        return httpx.Response(200, json={"status": "success", "code": 200, "data": {"version": "1.2.3"}})

    assert _client(handler).health() == {"version": "1.2.3"}


def test_an_error_envelope_raises_with_its_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"status": "error", "code": 503, "message": "db down"})

    with pytest.raises(ApiError, match="db down"):
        _client(handler).health()


def test_an_unreachable_api_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError, match="Cannot reach"):
        _client(handler).health()
