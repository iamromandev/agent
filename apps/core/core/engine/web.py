"""The fetch contract: what the web_fetch tool needs from an HTTP client (no HTTP library here)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FetchResponse:
    url: str  # final URL after redirects
    status: int
    reason: str  # "OK", "Not Found"
    content_type: str  # media type, lowercased, without parameters; "" if missing
    charset: str | None  # from the Content-Type header
    body: bytes  # at most max_bytes
    size: int  # bytes read
    redirected: bool = False  # True if at least one redirect was followed to reach `url`


class FetchError(Exception):
    """Fetching failed; str(exc) is a one-line reason such as 'timeout after 15s'."""

    def __init__(self, message: str, summary: str | None = None) -> None:
        super().__init__(message)
        self.summary = summary or message


class FetchTooLarge(FetchError):
    """The response body went past the byte limit."""


class Fetcher(Protocol):
    async def fetch(self, url: str) -> FetchResponse: ...
