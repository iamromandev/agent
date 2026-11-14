from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import httpx
import pytest
from src.agent.web import FetchError, FetchTooLarge
from src.client import WebClient

URL = "http://site.test/page"


def _client(handler: Callable[[httpx.Request], httpx.Response], max_bytes: int = 1000) -> WebClient:
    return WebClient(15.0, max_bytes, transport=httpx.MockTransport(handler))


async def test_a_page_comes_back_with_type_charset_and_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<p>hi</p>", headers={"content-type": "Text/HTML; charset=UTF-8"})

    r = await _client(handler).fetch(URL)
    assert (r.status, r.reason, r.content_type, r.charset) == (200, "OK", "text/html", "utf-8")
    assert r.body == b"<p>hi</p>" and r.size == 9 and r.url == URL


async def test_missing_content_type_is_an_empty_string() -> None:
    r = await _client(lambda request: httpx.Response(200, content=b"x")).fetch(URL)
    assert r.content_type == "" and r.charset is None


async def test_the_user_agent_names_the_agent() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        return httpx.Response(200)

    await _client(handler).fetch(URL)
    assert seen[0].startswith("agent-cli/")


async def test_404_comes_back_as_a_response_not_an_error() -> None:
    r = await _client(lambda request: httpx.Response(404)).fetch(URL)
    assert (r.status, r.reason) == (404, "Not Found")


async def test_an_oversized_body_stops_the_read_at_the_limit() -> None:
    chunks_read = 0

    async def body():
        nonlocal chunks_read
        for _ in range(1000):
            chunks_read += 1
            yield b"x" * 100

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    with pytest.raises(FetchTooLarge) as info:
        await _client(handler, max_bytes=500).fetch(URL)
    assert "larger than 500 bytes" in str(info.value)
    assert info.value.summary == "too large"
    assert chunks_read < 20


async def test_a_body_exactly_at_the_limit_is_fine() -> None:
    r = await _client(lambda request: httpx.Response(200, content=b"x" * 500), max_bytes=500).fetch(URL)
    assert r.size == 500


async def test_a_timeout_says_so() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(FetchError) as info:
        await _client(handler).fetch(URL)
    assert str(info.value) == "timeout after 15s" and info.value.summary == "timeout after 15s"


async def test_a_connection_failure_is_a_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("name or service not known", request=request)

    with pytest.raises(FetchError, match="cannot connect: name or service not known"):
        await _client(handler).fetch(URL)


async def test_other_http_errors_are_request_failed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("bad", request=request)

    with pytest.raises(FetchError, match="request failed: bad"):
        await _client(handler).fetch(URL)


async def test_a_redirect_is_followed_and_the_final_url_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/page":
            return httpx.Response(301, headers={"location": "/new"})
        return httpx.Response(200, content=b"new")

    r = await _client(handler).fetch(URL)
    assert r.url == "http://site.test/new" and r.body == b"new"


async def test_six_redirects_in_a_row_are_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        n = int(request.url.path.strip("/") or 0)
        return httpx.Response(302, headers={"location": f"/{n + 1}"})

    with pytest.raises(FetchError, match="too many redirects"):
        await _client(handler).fetch("http://site.test/0")


async def test_five_redirects_are_allowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        n = int(request.url.path.strip("/") or 0)
        if n < 5:
            return httpx.Response(302, headers={"location": f"/{n + 1}"})
        return httpx.Response(200, content=b"end")

    assert (await _client(handler).fetch("http://site.test/0")).body == b"end"


async def test_a_redirect_body_is_never_read() -> None:
    chunks_read = 0

    async def body():
        nonlocal chunks_read
        for _ in range(1000):
            chunks_read += 1
            yield b"x" * 100

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/page":
            return httpx.Response(301, headers={"location": "/new"}, content=body())
        return httpx.Response(200, content=b"new")

    r = await _client(handler, max_bytes=500).fetch(URL)
    assert r.body == b"new"
    assert chunks_read == 0


async def test_the_cap_applies_to_the_final_body_after_a_redirect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/page":
            return httpx.Response(302, headers={"location": "/big"})
        return httpx.Response(200, content=b"x" * 600)

    with pytest.raises(FetchTooLarge):
        await _client(handler, max_bytes=500).fetch(URL)


async def test_a_relative_and_an_absolute_location_are_both_resolved() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        match request.url.path:
            case "/a/b":
                return httpx.Response(302, headers={"location": "../c"})
            case "/c":
                return httpx.Response(302, headers={"location": "http://other.test/d"})
        return httpx.Response(200, content=b"end")

    r = await _client(handler).fetch("http://site.test/a/b")
    assert r.url == "http://other.test/d" and r.redirected


async def test_a_redirect_without_a_location_is_returned_as_the_response() -> None:
    r = await _client(lambda request: httpx.Response(304)).fetch(URL)
    assert r.status == 304 and not r.redirected


async def test_a_page_that_was_not_redirected_says_so_even_when_httpx_normalises_the_url() -> None:
    r = await _client(lambda request: httpx.Response(200, content=b"x")).fetch("https://EXAMPLE.com/a b")
    assert not r.redirected


async def test_a_slow_drip_body_hits_the_overall_deadline() -> None:
    async def drip():
        for _ in range(1000):
            await asyncio.sleep(0.05)
            yield b"x"

    client = WebClient(0.3, 1000, transport=httpx.MockTransport(lambda request: httpx.Response(200, content=drip())))
    start = time.perf_counter()
    with pytest.raises(FetchError) as info:
        await client.fetch(URL)
    assert str(info.value) == "timeout after 0.3s" and info.value.summary == "timeout after 0.3s"
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(
    "start,target",
    [
        ("https://site.test/", "http://site.test/"),
        ("https://site.test/", "https://127.0.0.1/admin"),
        ("https://site.test/", "https://localhost/admin"),
        ("https://site.test/", "https://sub.localhost/admin"),
        ("https://site.test/", "https://169.254.169.254/latest/meta-data"),
        ("https://site.test/", "https://10.1.2.3/"),
        ("https://site.test/", "https://192.168.0.1/"),
        ("https://site.test/", "https://[::1]/"),
        ("https://site.test/", "https://[::ffff:127.0.0.1]/"),
        ("https://site.test/", "https://0.0.0.0/"),
    ],
)
async def test_an_unsafe_redirect_hop_is_refused_before_it_is_requested(start: str, target: str) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": target})

    with pytest.raises(FetchError, match="refused redirect to") as info:
        await _client(handler).fetch(start)
    assert info.value.summary == "refused redirect"
    assert requested == [start]


async def test_an_http_to_https_hop_is_fine() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.scheme == "http":
            return httpx.Response(301, headers={"location": "https://site.test/page"})
        return httpx.Response(200, content=b"secure")

    assert (await _client(handler).fetch(URL)).body == b"secure"


@pytest.mark.parametrize("start", ["http://localhost:8000/a", "http://127.0.0.1/a", "http://10.0.0.5/a"])
async def test_a_local_start_may_redirect_to_local_hosts(start: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:9000/b"})
        return httpx.Response(200, content=b"local")

    assert (await _client(handler).fetch(start)).body == b"local"
