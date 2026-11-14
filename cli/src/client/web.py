"""HTTP client for the web_fetch tool: the only place the fetch contract meets httpx."""

from __future__ import annotations

import asyncio
import ipaddress
from importlib.metadata import PackageNotFoundError, version

import httpx

from src.agent.web import FetchError, FetchResponse, FetchTooLarge

MAX_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _version() -> str:
    try:
        return version("agent-cli")
    except PackageNotFoundError:
        return "dev"


def _is_local(host: str) -> bool:
    """True for "localhost" and literal loopback, link-local and private addresses.

    The check is on the text of the URL only: a public name that resolves to a private address
    (DNS rebinding, an attacker's A record for 127.0.0.1) is not caught, as no lookup is made.
    """
    host = host.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_link_local or address.is_private or address.is_unspecified


class WebClient:
    """Implements `src.agent.web.Fetcher`: GET with redirects, a timeout and a byte cap."""

    def __init__(
        self, timeout: float, max_bytes: int, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._timeout = timeout
        self._max_bytes = max_bytes
        # No automatic redirects: httpx reads a 3xx body in full, which would bypass the byte cap.
        self._http = httpx.AsyncClient(
            follow_redirects=False,
            timeout=timeout,
            headers={"User-Agent": f"agent-cli/{_version()}"},
            transport=transport,
        )

    async def fetch(self, url: str) -> FetchResponse:
        try:
            async with asyncio.timeout(self._timeout):  # the whole fetch, not just each socket read
                return await self._fetch(url)
        except TimeoutError as e:
            raise FetchError(f"timeout after {self._timeout:g}s") from e
        # Most specific first: TimeoutException, TooManyRedirects and ConnectError are all
        # HTTPError subclasses, so HTTPError must come after them. InvalidURL is not an HTTPError.
        except httpx.TimeoutException as e:
            message = f"timeout after {self._timeout:g}s"
            raise FetchError(message) from e
        except httpx.TooManyRedirects as e:
            raise FetchError("too many redirects") from e
        except httpx.ConnectError as e:
            raise FetchError(f"cannot connect: {e}", "connection error") from e
        except httpx.HTTPError as e:
            raise FetchError(f"request failed: {e}", "request failed") from e
        except httpx.InvalidURL as e:  # a malformed URL that httpx rejects
            raise FetchError(f"request failed: {e}", "request failed") from e

    async def _fetch(self, url: str) -> FetchResponse:
        request = self._http.build_request("GET", url)
        start_local = _is_local(request.url.host)
        hops = 0
        while True:
            response = await self._http.send(request, stream=True)
            try:
                location = response.headers.get("location")
                if response.status_code not in _REDIRECT_STATUSES or not location:
                    return await self._read(response, redirected=hops > 0)
            finally:
                await response.aclose()  # a redirect's body is closed unread
            if hops == MAX_REDIRECTS:
                raise httpx.TooManyRedirects("too many redirects", request=request)
            target = request.url.join(location)
            self._check_hop(request.url, target, start_local)
            request = self._http.build_request("GET", target)
            hops += 1

    @staticmethod
    def _check_hop(source: httpx.URL, target: httpx.URL, start_local: bool) -> None:
        downgrade = source.scheme == "https" and target.scheme == "http"
        to_local = _is_local(target.host) and not start_local
        if downgrade or to_local:
            raise FetchError(f"refused redirect to {target}", "refused redirect")

    async def _read(self, response: httpx.Response, redirected: bool) -> FetchResponse:
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > self._max_bytes:
                raise FetchTooLarge(f"response larger than {self._max_bytes} bytes", "too large")
            chunks.append(chunk)
        content_type = response.headers.get("content-type", "")
        media = content_type.split(";")[0].strip().lower()
        return FetchResponse(
            url=str(response.url),
            status=response.status_code,
            reason=response.reason_phrase,
            content_type=media,
            charset=response.charset_encoding,
            body=b"".join(chunks),
            size=size,
            redirected=redirected,
        )

    async def aclose(self) -> None:
        await self._http.aclose()
