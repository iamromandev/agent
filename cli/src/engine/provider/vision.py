"""Whether a model has vision, from its capabilities (UI-free). One answer per model per run, once one is known."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from src.engine.text import escape

from .models import SHOW_TIMEOUT, ModelCatalog, shown_name


class VisionCheck:
    """Asks the catalog once per model and keeps the answer. A failure, a timeout or no catalog is "no vision" for
    that message; a failure is not kept, so the next message asks again (a passing Ollama hiccup does not drop
    images for the whole run), and it warns once per model."""

    def __init__(self, catalog: ModelCatalog | None, timeout: float = SHOW_TIMEOUT) -> None:
        self._catalog = catalog
        self._timeout = timeout
        self._cache: dict[str, bool] = {}
        self._warned: set[str] = set()  # models whose lookup failed and was reported

    async def has(self, model: str, warn: Callable[[str], None] | None = None) -> bool:
        if model in self._cache:
            return self._cache[model]
        if self._catalog is None:
            return False
        try:
            async with asyncio.timeout(self._timeout):
                has = "vision" in await self._catalog.capabilities(model)
        except Exception as error:  # a CatalogError, a timeout or a bug: never stops a turn
            if warn is not None and model not in self._warned:
                self._warned.add(model)
                text = escape(str(error)) or type(error).__name__
                warn(f"could not read the capabilities of '{shown_name(model)}' ({text}); images are not sent")
            return False
        self._cache[model] = has
        return has

    def set(self, model: str, value: bool) -> None:
        self._cache[model] = value
