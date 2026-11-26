from __future__ import annotations

import json
from collections.abc import AsyncIterator

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


# tags, show and the catalog


async def test_tags_returns_the_raw_entries() -> None:
    entries = [{"name": "a:latest", "size": 5, "details": {"parameter_size": "7B"}}, {"name": "b"}]

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": entries})

    assert await _client(handler).tags() == entries


async def test_tags_rejects_an_entry_without_a_name() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"size": 1}]})

    with pytest.raises(OllamaError, match="invalid model list"):
        await _client(handler).tags()


async def test_show_posts_the_model_and_returns_the_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        assert json.loads(request.content) == {"model": "a:latest"}
        return httpx.Response(200, json={"model_info": {"x": 1}})

    assert await _client(handler).show("a:latest", timeout=5.0) == {"model_info": {"x": 1}}


async def test_show_http_error_carries_the_body_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'x' not found"})

    with pytest.raises(OllamaError, match="not found"):
        await _client(handler).show("x", timeout=5.0)


async def test_show_unreachable_raises() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaUnreachable):
        await _client(handler).show("x", timeout=5.0)


async def test_show_rejects_a_body_that_is_not_an_object() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1])

    with pytest.raises(OllamaError, match="invalid"):
        await _client(handler).show("x", timeout=5.0)


async def test_catalog_lists_names_sizes_and_parameters_in_order() -> None:
    from src.client.ollama import OllamaCatalog
    from src.engine.provider.models import ModelInfo

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": "b", "size": 7, "details": {"parameter_size": "7B"}}, {"name": "a", "size": "x", "details": 3}]},
        )

    assert await OllamaCatalog(_client(handler)).list() == [ModelInfo("b", 7, "7B"), ModelInfo("a")]


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"general.architecture": "qwen2", "qwen2.context_length": 32768, "llama.context_length": 1}, 32768),
        ({"general.architecture": "qwen2", "llama.context_length": 4096}, 4096),
        ({"general.architecture": "qwen2"}, None),
        ({"x.context_length": "big"}, None),
        ({"x.context_length": True}, None),
        ({"x.context_length": 0}, None),
        ({}, None),
    ],
)
async def test_catalog_reads_the_context_length(info: dict, expected: int | None) -> None:
    from src.client.ollama import OllamaCatalog

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model_info": info})

    assert await OllamaCatalog(_client(handler)).context_length("m") == expected


async def test_catalog_without_model_info_has_no_length() -> None:
    from src.client.ollama import OllamaCatalog

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    assert await OllamaCatalog(_client(handler)).context_length("m") is None


async def test_catalog_errors_become_escaped_catalog_errors() -> None:
    from src.client.ollama import OllamaCatalog
    from src.engine.provider.models import CatalogError

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "bad \x1b[31m thing‮"})

    catalog = OllamaCatalog(_client(handler))
    for call in (catalog.list(), catalog.context_length("m")):
        with pytest.raises(CatalogError) as info:
            await call
        assert "\x1b" not in str(info.value)
        assert "‮" not in str(info.value)
        assert "\\x1b" in str(info.value)


async def test_catalog_gives_up_on_a_slow_answer() -> None:
    import asyncio

    from src.client.ollama import OllamaCatalog
    from src.engine.provider.models import CatalogError

    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    catalog = OllamaCatalog(_client(handler), timeout=0.05)
    with pytest.raises(CatalogError, match="did not answer"):
        await catalog.context_length("m")
    with pytest.raises(CatalogError, match="did not answer"):
        await catalog.list()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"capabilities": ["completion", "vision"]}, ("completion", "vision")),
        ({"capabilities": ["completion", 3, None, "tools"]}, ("completion", "tools")),
        ({"capabilities": "vision"}, ()),
        ({}, ()),
    ],
)
async def test_catalog_reads_the_capabilities(body: dict, expected: tuple[str, ...]) -> None:
    from src.client.ollama import OllamaCatalog

    async def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"model": "m"}
        return httpx.Response(200, json=body)

    assert await OllamaCatalog(_client(handler)).capabilities("m") == expected


async def test_catalog_capabilities_fail_as_catalog_errors() -> None:
    import asyncio

    from src.client.ollama import OllamaCatalog
    from src.engine.provider.models import CatalogError

    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    with pytest.raises(CatalogError, match="did not answer"):
        await OllamaCatalog(_client(handler), timeout=0.05).capabilities("m")

    async def broken(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with pytest.raises(CatalogError):
        await OllamaCatalog(_client(broken)).capabilities("m")
