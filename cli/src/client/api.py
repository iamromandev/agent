"""The only module that talks HTTP. Commands call this, never ``httpx`` directly."""

from __future__ import annotations

from typing import Any

import httpx

from src.config import get_settings


class ApiError(Exception):
    """The API could not be reached, or answered with its error envelope."""


class ApiClient:
    def __init__(self, base_url: str, timeout: float, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _get(self, path: str) -> Any:
        try:
            response = self._http.get(path)
        except httpx.HTTPError as error:
            raise ApiError(f"Cannot reach {self._http.base_url}: {error}") from error
        body = response.json() if response.content else {}
        if response.is_error or body.get("status") == "error":
            raise ApiError(body.get("message") or f"HTTP {response.status_code}")
        return body.get("data")

    def health(self) -> dict[str, Any]:
        return self._get("/health/check") or {}


def get_api_client() -> ApiClient:
    settings = get_settings()
    return ApiClient(base_url=settings.api_url, timeout=settings.api_timeout)
