"""The search contract: what the web_search tool needs from a search client (no HTTP library here)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str  # may be empty


class SearchError(Exception):
    """Searching failed; str(exc) is a one-line reason such as 'search failed: timeout after 15s'."""

    def __init__(self, message: str, summary: str | None = None) -> None:
        super().__init__(message)
        self.summary = summary or message


class Searcher(Protocol):
    async def search(self, query: str, limit: int) -> list[SearchHit]: ...
