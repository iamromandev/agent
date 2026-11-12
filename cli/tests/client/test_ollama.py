from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from src.client import OllamaClient, OllamaError, OllamaUnreachable

URL = "http://ollama.test"


def _client(handler) -> OllamaClient:
    return OllamaClient(URL, transport=httpx.MockTransport(handler))


def _ndjson(*lines: dict) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


async def _collect(client: OllamaClient) -> list[dict]:
    return [chunk async for chunk in client.chat({"model": "m"})]


async def test_chat_yields_lines_in_order_and_skips_blanks() -> None:
    body = b'{"n":1,"done":false}\n\n{"n":2,"done":false}\n{"n":3,"done":true}\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert json.loads(request.content) == {"model": "m"}
        return httpx.Response(200, content=body)

    chunks = await _collect(_client(handler))
    assert [c["n"] for c in chunks] == [1, 2, 3]


async def test_chat_streams_a_recorded_fixture(fixture_path: Callable[[str], Path]) -> None:
    data = fixture_path("ollama/text.ndjson").read_bytes()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=data)

    chunks = await _collect(_client(handler))
    assert chunks[-1]["done"] is True


async def test_unreachable_server_raises_with_the_url() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaError, match=f"Cannot reach Ollama at {URL}"):
        await _collect(_client(handler))


async def test_http_error_carries_the_body_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'x' not found"})

    with pytest.raises(OllamaError, match="model 'x' not found"):
        await _collect(_client(handler))


async def test_http_error_without_json_body_reports_the_code() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    with pytest.raises(OllamaError, match="HTTP 500"):
        await _collect(_client(handler))


async def test_error_line_mid_stream_raises() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson({"done": False}, {"error": "out of memory"}))

    with pytest.raises(OllamaError, match="out of memory"):
        await _collect(_client(handler))


async def test_invalid_json_line_raises() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"done":false}\nnot json\n')

    with pytest.raises(OllamaError, match="invalid"):
        await _collect(_client(handler))


async def test_stream_without_done_raises() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson({"done": False}))

    with pytest.raises(OllamaError, match="closed the stream early"):
        await _collect(_client(handler))


async def test_dropped_connection_mid_stream_raises() -> None:
    async def body() -> AsyncIterator[bytes]:
        yield b'{"done":false}\n'
        raise httpx.ReadError("dropped")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    with pytest.raises(OllamaError):
        await _collect(_client(handler))


async def test_models_returns_names() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "a:latest"}, {"name": "b:7b"}]})

    assert await _client(handler).models() == ["a:latest", "b:7b"]


async def test_models_unreachable_raises() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaError, match="Cannot reach Ollama"):
        await _client(handler).models()


async def test_chat_connect_timeout_raises_unreachable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(OllamaUnreachable, match=f"Cannot reach Ollama at {URL}"):
        await _collect(_client(handler))


async def test_models_connect_timeout_raises_unreachable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(OllamaUnreachable):
        await _client(handler).models()


async def test_models_connect_error_raises_unreachable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaUnreachable):
        await _client(handler).models()


async def test_models_read_timeout_is_not_unreachable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(OllamaError) as info:
        await _client(handler).models()
    assert not isinstance(info.value, OllamaUnreachable)


async def test_models_http_500_is_not_unreachable() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    with pytest.raises(OllamaError, match="HTTP 500") as info:
        await _client(handler).models()
    assert not isinstance(info.value, OllamaUnreachable)
