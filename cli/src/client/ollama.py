"""Async Ollama HTTP client. Part of the only package that talks HTTP."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.config import get_settings

_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class OllamaError(Exception):
    """Talking to Ollama failed. The message is meant for the user."""


class OllamaUnreachable(OllamaError):
    """Ollama could not be connected to at all."""


class OllamaClient:
    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0),
            transport=transport,
        )

    def _unreachable(self, error: httpx.HTTPError) -> OllamaUnreachable:
        return OllamaUnreachable(f"Cannot reach Ollama at {self.base_url}: {error}")

    @staticmethod
    def _http_error(response: httpx.Response, body: bytes) -> OllamaError:
        try:
            message = json.loads(body).get("error")
        except (ValueError, AttributeError):
            message = None
        return OllamaError(message or f"HTTP {response.status_code}")

    async def chat(self, payload: dict[str, Any]) -> AsyncGenerator[dict[str, Any]]:
        finished = False
        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as response:
                if response.is_error:
                    raise self._http_error(response, await response.aread())
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError as error:
                        raise OllamaError(f"Ollama sent invalid JSON: {line[:200]}") from error
                    if not isinstance(chunk, dict):
                        raise OllamaError(f"Ollama sent invalid JSON: {line[:200]}")
                    if "error" in chunk:
                        raise OllamaError(str(chunk["error"]))
                    finished = finished or bool(chunk.get("done"))
                    yield chunk
        except _CONNECT_ERRORS as error:
            raise self._unreachable(error) from error
        except httpx.HTTPError as error:
            raise OllamaError(f"Connection to Ollama failed: {error}") from error
        if not finished:
            raise OllamaError("Ollama closed the stream early")

    async def models(self) -> list[str]:
        try:
            response = await self._http.get("/api/tags")
        except _CONNECT_ERRORS as error:
            raise self._unreachable(error) from error
        except httpx.HTTPError as error:
            raise OllamaError(f"Connection to Ollama failed: {error}") from error
        if response.is_error:
            raise self._http_error(response, response.content)
        try:
            return [model["name"] for model in response.json().get("models", [])]
        except (ValueError, AttributeError, KeyError, TypeError) as error:
            raise OllamaError("Ollama sent an invalid model list") from error

    async def aclose(self) -> None:
        await self._http.aclose()


def get_ollama_client() -> OllamaClient:
    return OllamaClient(get_settings().ollama_url)
