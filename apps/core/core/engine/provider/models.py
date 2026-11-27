"""Which models are installed and how large a window each has: the contract, with no HTTP (see `client/ollama.py`)."""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from core.engine.text import escape

SHOW_TIMEOUT = 5.0  # seconds one details lookup may take
SLOW_AFTER = 1.0  # seconds before the launch lookup is announced on stderr
NAME_SHOWN = 200  # characters of a model name shown in a warning or notice


def shown_name(name: str) -> str:
    """`name` escaped for a warning or notice and cut to `NAME_SHOWN` characters (then `…`): a name that came from a
    file or the command line must not flood the transcript."""
    cut = name[:NAME_SHOWN]
    return escape(cut) + ("…" if len(name) > NAME_SHOWN else "")


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size: int | None = None  # bytes, from /api/tags
    parameters: str | None = None  # details.parameter_size, e.g. "7.6B"
    context_length: int | None = None  # from /api/show


class CatalogError(Exception):
    """Listing models or reading one's details failed. The message is escaped and meant for the user."""


class ModelCatalog(Protocol):
    async def list(self) -> builtins.list[ModelInfo]:
        """The installed models in Ollama's order, without `context_length`. Raises `CatalogError`."""
        ...

    async def context_length(self, name: str) -> int | None:
        """The trained context length of `name`; None when Ollama reports none. Raises `CatalogError`."""
        ...

    async def capabilities(self, name: str) -> tuple[str, ...]:
        """The capabilities Ollama lists for `name` (`completion`, `vision`, ...); empty when it lists none.
        Raises `CatalogError`."""
        ...


def resolve_window(length: int | None, cap: int) -> int:
    """min(the model's length, the cap); the cap when the length is unknown."""
    return min(length, cap) if length else cap


def installed_name(name: str, installed: list[str]) -> str | None:
    """The installed name `name` stands for: exactly as listed, or `name:latest` when it has no tag."""
    if name in installed:
        return name
    if ":" not in name.rsplit("/", 1)[-1] and f"{name}:latest" in installed:
        return f"{name}:latest"
    return None


def pick_launch_model(
    flag: str | None, env: str | None, project: object, default: str, installed: list[str] | None
) -> tuple[str, list[str]]:
    """The model to start with and the launch warnings.

    Order: the flag, the environment, the project key (a string naming an installed model), the default. When
    `installed` is None (Ollama cannot be reached) a project key is taken as it is.
    """
    if flag:
        return flag, []
    if env is not None:
        return env, []
    if project is None:
        return default, []
    if not isinstance(project, str) or not project.strip():
        return default, [f'.agent/settings.json: "model" must be a model name, not {escape(repr(project)[:80])}; using {shown_name(default)}']
    project = project.strip()
    if installed is not None and installed_name(project, installed) is None:
        return default, [f"Model {shown_name(project)} from .agent/settings.json is not installed; using {shown_name(default)}"]
    return project, []


async def launch_model(
    settings: Any,
    flag: str | None,
    project_data: Mapping[str, object] | None,
    catalog: ModelCatalog,
    timeout: float = SHOW_TIMEOUT,
    notices: list[str] | None = None,
    on_slow: Callable[[str], None] | None = None,
    slow_after: float = SLOW_AFTER,
) -> tuple[str, int, list[str]]:
    """`_launch_model`, calling `on_slow(name)` once when it takes more than `slow_after` seconds, with the model
    being asked about (the flag, the environment's, the project's key or the default), so the user sees why the
    window is still empty."""
    task = asyncio.ensure_future(_launch_model(settings, flag, project_data, catalog, timeout, notices))
    try:
        if on_slow is not None:
            await asyncio.wait({task}, timeout=slow_after)
            if not task.done():
                project = project_data.get("model") if project_data is not None else None
                explicit = "model" in settings.model_fields_set and settings.model.strip()
                asked = flag or (settings.model.strip() if explicit else None)
                if not asked and isinstance(project, str) and project.strip():
                    asked = project.strip()
                on_slow(asked or settings.model)
        return await task
    finally:
        if not task.done():
            task.cancel()


async def _launch_model(
    settings: Any,
    flag: str | None,
    project_data: Mapping[str, object] | None,
    catalog: ModelCatalog,
    timeout: float,
    notices: list[str] | None,
) -> tuple[str, int, list[str]]:
    """The model to start with, its window and the launch warnings.

    `settings` has `model`, `num_ctx` and `model_fields_set` (an explicit `AGENT_MODEL`, from the environment or
    `.env`, is in it). The installed models are asked for only when the project key decides the choice; each call
    is bounded by `timeout`. The window is `resolve_window` of the model's length; when the lookup fails (or the
    catalog already did not answer) it is `num_ctx` and a warning says so. A blank explicit `AGENT_MODEL` counts
    as unset. When the project key decided the model, an info line for the user is appended to `notices`.
    """
    explicit = "model" in settings.model_fields_set and bool(settings.model.strip())
    project = project_data.get("model") if project_data is not None else None
    installed: list[str] | None = None
    reachable = True
    if not flag and not explicit and project is not None:
        try:
            async with asyncio.timeout(timeout):
                installed = [info.name for info in await catalog.list()]
        except (CatalogError, TimeoutError):
            reachable = False
    model, warnings = pick_launch_model(flag, settings.model.strip() if explicit else None, project, settings.model, installed)
    if not flag and not explicit and notices is not None and isinstance(project, str) and project.strip() == model:
        notices.append(f"Model {shown_name(model)} (from .agent/settings.json)")
    length: int | None = None
    failed = not reachable
    if reachable:
        try:
            async with asyncio.timeout(timeout):
                length = await catalog.context_length(model)
        except (CatalogError, TimeoutError):
            failed = True
    if failed:
        warnings.append(
            f"Could not read the context length of {shown_name(model)}; using AGENT_NUM_CTX ({settings.num_ctx})"
        )
    return model, resolve_window(length, settings.num_ctx), warnings
