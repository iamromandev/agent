from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from src.client import SearxClient
from src.engine.search_web import SearchError, SearchHit

BASE = "http://searx.test:8888"


def _client(handler: Callable[[httpx.Request], httpx.Response], base: str = BASE, timeout: float = 15.0) -> SearxClient:
    return SearxClient(base, timeout, transport=httpx.MockTransport(handler))


def _json(results: Any) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, json={"results": results})


async def test_hits_are_parsed_in_order_and_cut_to_limit() -> None:
    results = [{"title": f"T{n}", "url": f"https://e.com/{n}", "content": f"c{n}"} for n in range(7)]
    hits = await _client(_json(results)).search("q", 3)
    assert hits == [SearchHit(f"T{n}", f"https://e.com/{n}", f"c{n}") for n in range(3)]


async def test_missing_url_is_skipped_and_missing_title_and_content_fall_back() -> None:
    results = [
        {"title": "no url", "content": "x"},
        {"title": "empty url", "url": ""},
        "not a dict",
        {"url": "https://e.com/a"},
        {"title": 5, "url": "https://e.com/b", "content": None},
        {"title": "ok", "url": "https://e.com/c", "content": "snip"},
    ]
    hits = await _client(_json(results)).search("q", 10)
    assert hits == [
        SearchHit("https://e.com/a", "https://e.com/a", ""),
        SearchHit("https://e.com/b", "https://e.com/b", ""),
        SearchHit("ok", "https://e.com/c", "snip"),
    ]


async def test_connect_error_is_a_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(SearchError) as info:
        await _client(handler).search("q", 5)
    assert str(info.value) == f"search failed: cannot connect ({BASE})" and info.value.summary == "unreachable"


async def test_other_http_errors_name_their_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("bad")

    with pytest.raises(SearchError, match="search failed: RemoteProtocolError"):
        await _client(handler).search("q", 5)


async def test_status_500_is_a_search_error() -> None:
    with pytest.raises(SearchError) as info:
        await _client(lambda request: httpx.Response(500)).search("q", 5)
    assert str(info.value) == "search failed: HTTP 500 Internal Server Error"


async def test_status_403_hints_at_the_json_format_setting() -> None:
    with pytest.raises(SearchError) as info:
        await _client(lambda request: httpx.Response(403)).search("q", 5)
    assert "403" in str(info.value) and "search.formats" in str(info.value)


async def test_a_redirect_is_not_followed() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url.host))
        return httpx.Response(302, headers={"location": "http://evil.test/search"})

    with pytest.raises(SearchError) as info:
        await _client(handler).search("q", 5)
    assert seen == ["searx.test"]
    assert "redirected (HTTP 302)" in str(info.value) and "AGENT_SEARCH_URL" in str(info.value)


@pytest.mark.parametrize("body", [b"not json", b'"x"', b'{"results": "nope"}'])
async def test_unexpected_bodies_are_search_errors(body: bytes) -> None:
    with pytest.raises(SearchError) as info:
        await _client(lambda request: httpx.Response(200, content=body)).search("q", 5)
    assert str(info.value) == "search failed: unexpected response from SearXNG"


async def test_an_oversized_response_is_a_search_error() -> None:
    padding = "x" * (2_000_000 + 10)
    body = json.dumps({"results": [], "pad": padding})
    with pytest.raises(SearchError) as info:
        await _client(lambda request: httpx.Response(200, content=body.encode())).search("q", 5)
    assert str(info.value) == "search failed: response too large"


async def test_a_stalled_body_hits_one_overall_deadline() -> None:
    class Stalls(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b"{"
            await asyncio.sleep(10)
            yield b"}"

    with pytest.raises(SearchError) as info:
        await _client(lambda request: httpx.Response(200, stream=Stalls()), timeout=0.05).search("q", 5)
    assert info.value.summary == "timeout" and "timeout after 0.05s" in str(info.value)
