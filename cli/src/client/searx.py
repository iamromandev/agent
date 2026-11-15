"""HTTP client for the web_search tool: the only place the search contract meets httpx."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import PackageNotFoundError, version

import httpx

from src.engine.search_web import SearchError, SearchHit

MAX_BYTES = 2_000_000
_UNEXPECTED = "search failed: unexpected response from SearXNG"


def _version() -> str:
    try:
        return version("agent-cli")
    except PackageNotFoundError:
        return "dev"


class SearxClient:
    """Implements `src.engine.search_web.Searcher` against a SearXNG instance: one GET, never redirected."""

    def __init__(
        self, base_url: str, timeout: float, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._base_url = base_url
        self._url = base_url.rstrip("/") + "/search"
        self._timeout = timeout
        self._http = httpx.AsyncClient(
            follow_redirects=False,
            timeout=timeout,
            headers={"User-Agent": f"agent-cli/{_version()}"},
            transport=transport,
        )

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        try:
            async with asyncio.timeout(self._timeout):  # one deadline for connect, headers and body
                body = await self._get(query)
        except httpx.ConnectError as e:
            raise SearchError(f"search failed: cannot connect ({self._shown_url()})", "unreachable") from e
        except (httpx.TimeoutException, TimeoutError) as e:
            raise SearchError(f"search failed: timeout after {self._timeout:g}s", "timeout") from e
        except httpx.HTTPError as e:
            raise SearchError(f"search failed: {type(e).__name__}") from e
        except httpx.InvalidURL as e:  # a malformed AGENT_SEARCH_URL
            raise SearchError(f"search failed: {type(e).__name__}") from e
        return self._parse(body, limit)

    def _shown_url(self) -> str:
        """The base URL without userinfo and query, safe to put in a message."""
        try:
            url = httpx.URL(self._base_url)
            return str(url.copy_with(userinfo=b"", query=None, fragment=None))
        except httpx.InvalidURL:
            return "AGENT_SEARCH_URL"

    async def _get(self, query: str) -> bytes:
        request = self._http.build_request("GET", self._url, params={"q": query, "format": "json"})
        response = await self._http.send(request, stream=True)
        try:
            status = response.status_code
            if 300 <= status < 400:
                raise SearchError(
                    f"search failed: SearXNG redirected (HTTP {status}); set AGENT_SEARCH_URL to the final address"
                )
            if status == 403:
                raise SearchError(
                    "search failed: HTTP 403 Forbidden; enable json in SearXNG's search.formats setting"
                )
            if status != 200:
                raise SearchError(f"search failed: HTTP {status} {response.reason_phrase}".rstrip())
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise SearchError("search failed: response too large")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            await response.aclose()

    @staticmethod
    def _parse(body: bytes, limit: int) -> list[SearchHit]:
        try:
            data = json.loads(body)
        except (ValueError, RecursionError) as e:  # ValueError includes UnicodeDecodeError
            raise SearchError(_UNEXPECTED) from e
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise SearchError(_UNEXPECTED)
        hits: list[SearchHit] = []
        for item in results:
            if len(hits) >= limit:
                break
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url:
                continue
            title = item.get("title")
            content = item.get("content")
            hits.append(
                SearchHit(
                    title if isinstance(title, str) and title else url,
                    url,
                    content if isinstance(content, str) else "",
                )
            )
        return hits

    async def aclose(self) -> None:
        await self._http.aclose()
