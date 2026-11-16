"""Async Ollama HTTP client. Part of the only package that talks HTTP."""

from __future__ import annotations

import asyncio
import builtins
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.config import get_settings
from src.engine.provider.models import SHOW_TIMEOUT, CatalogError, ModelInfo
from src.engine.text import escape

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

    async def _get(self, path: str) -> httpx.Response:
        try:
            response = await self._http.get(path)
        except _CONNECT_ERRORS as error:
            raise self._unreachable(error) from error
        except httpx.HTTPError as error:
            raise OllamaError(f"Connection to Ollama failed: {error}") from error
        if response.is_error:
            raise self._http_error(response, response.content)
        return response

    async def tags(self) -> list[dict[str, Any]]:
        """The entries of `/api/tags`, each with at least a `name`."""
        response = await self._get("/api/tags")
        try:
            entries = response.json().get("models", [])
            if not all(isinstance(entry, dict) and isinstance(entry["name"], str) for entry in entries):
                raise TypeError("entry")
            return list(entries)
        except (ValueError, AttributeError, KeyError, TypeError) as error:
            raise OllamaError("Ollama sent an invalid model list") from error

    async def models(self) -> list[str]:
        return [entry["name"] for entry in await self.tags()]

    async def show(self, name: str, timeout: float) -> dict[str, Any]:
        """`POST /api/show` for one model, with its own time limit."""
        try:
            response = await self._http.post("/api/show", json={"model": name}, timeout=timeout)
        except _CONNECT_ERRORS as error:
            raise self._unreachable(error) from error
        except httpx.HTTPError as error:
            raise OllamaError(f"Connection to Ollama failed: {error}") from error
        if response.is_error:
            raise self._http_error(response, response.content)
        try:
            body = response.json()
        except ValueError as error:
            raise OllamaError("Ollama sent an invalid model description") from error
        if not isinstance(body, dict):
            raise OllamaError("Ollama sent an invalid model description")
        return body

    async def aclose(self) -> None:
        await self._http.aclose()


def _number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _context_length(info: Any) -> int | None:
    """`<architecture>.context_length` of `model_info`, else any `*.context_length`; None when there is none."""
    if not isinstance(info, dict):
        return None
    arch = info.get("general.architecture")
    if isinstance(arch, str) and (length := _number(info.get(f"{arch}.context_length"))):
        return length
    for key, value in info.items():
        if isinstance(key, str) and key.endswith(".context_length") and (length := _number(value)):
            return length
    return None


class OllamaCatalog:
    """`ModelCatalog` over `OllamaClient`: every failure is a `CatalogError` with escaped text."""

    def __init__(self, client: OllamaClient, timeout: float = SHOW_TIMEOUT) -> None:
        self._client = client
        self._timeout = timeout

    async def _bounded(self, call: Any) -> Any:
        try:
            async with asyncio.timeout(self._timeout):
                return await call
        except TimeoutError as error:
            raise CatalogError(f"Ollama did not answer within {self._timeout:g} s") from error
        except OllamaError as error:
            raise CatalogError(escape(str(error))) from error

    async def list(self) -> builtins.list[ModelInfo]:
        entries = await self._bounded(self._client.tags())
        models: list[ModelInfo] = []
        for entry in entries:
            details = entry.get("details")
            params = details.get("parameter_size") if isinstance(details, dict) else None
            models.append(ModelInfo(entry["name"], _number(entry.get("size")), params if isinstance(params, str) else None))
        return models

    async def context_length(self, name: str) -> int | None:
        body = await self._bounded(self._client.show(name, self._timeout))
        return _context_length(body.get("model_info"))


def get_ollama_client() -> OllamaClient:
    return OllamaClient(get_settings().ollama_url)
