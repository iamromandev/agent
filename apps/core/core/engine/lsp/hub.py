"""The language servers of a run: chosen by file extension, started on the first matching file, one process per name."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from core.engine.events import EventSink, Notice
from core.engine.lsp.config import LspSetup, ServerSpec
from core.engine.lsp.render import Rendered, fit, render
from core.engine.lsp.server import LspServer
from core.engine.text import escape_display


class LspHub:
    def __init__(
        self,
        setup: LspSetup,
        workspace: Path,
        env: Mapping[str, str],
        *,
        project_active: Callable[[], bool] = lambda: True,
        global_active: Callable[[], bool] = lambda: True,
        wait: float,
        start_wait: float,
        max_lines: int,
    ) -> None:
        self._setup = setup
        self._workspace = workspace
        self._env = dict(env)
        self._project_active = project_active
        self._global_active = global_active
        self._wait = wait
        self._start_wait = start_wait
        self._max_lines = max_lines
        self._servers: dict[str, LspServer] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._withdrawn: dict[str, str] = {}
        self._told: set[tuple[str, str]] = set()  # (server, reason): each warning once per run

    def _active(self, spec: ServerSpec) -> bool:
        if spec.name in self._withdrawn:
            return False
        return self._trusted(spec)

    def _shown(self, path: Path) -> str:
        try:
            return path.relative_to(self._workspace).as_posix()
        except ValueError:
            return str(path)

    async def block(self, path: Path, text: str, emit: EventSink, budget: int) -> str:
        """The diagnostics of `path` (its new `text`) as one block per matching server; "" when there are none, the
        budget is spent, or nothing matches. Never raises but for a cancel."""
        suffix = path.suffix.lower()
        blocks: list[Rendered] = []
        for spec in self._setup.servers:
            if suffix not in spec.extensions or not self._active(spec):
                continue
            server = self._servers.get(spec.name)
            if server is None:
                server = self._servers[spec.name] = LspServer(spec, self._workspace, self._env)
            lock = self._locks.setdefault(spec.name, asyncio.Lock())
            async with lock:
                if not self._active(spec):  # withdrawn (or no longer trusted) while it waited for the lock
                    continue
                found = await server.diagnostics(path, text, wait=self._wait, start_wait=self._start_wait)
            self._tell(spec.name, server, emit)
            if found:
                block = render(spec.name, self._shown(path), found, warnings=self._setup.warnings, max_lines=self._max_lines)
                if block is not None:
                    blocks.append(block)
        return fit(blocks, budget)

    def _tell(self, name: str, server: LspServer, emit: EventSink) -> None:
        if server.state in ("exited", "timed out") and server.reason and (name, server.reason) not in self._told:
            self._told.add((name, server.reason))
            emit(Notice("warn", f"lsp {escape_display(name)}: {server.reason}"))

    @property
    def warnings(self) -> tuple[str, ...]:
        """The launch warnings of the setup (already escaped)."""
        return self._setup.warnings_text

    def listing(self) -> list[tuple[str, str, str, str | None]]:
        rows: list[tuple[str, str, str, str | None]] = []
        for spec in self._setup.servers:
            server = self._servers.get(spec.name)
            if spec.name in self._withdrawn:
                rows.append((spec.name, spec.source, "off", self._withdrawn[spec.name]))
            elif not self._trusted(spec):
                rows.append((spec.name, spec.source, "untrusted", "untrusted"))
            elif server is None:
                rows.append((spec.name, spec.source, "idle", None))
            else:
                rows.append((spec.name, spec.source, server.state, server.reason))
        return rows

    def _trusted(self, spec: ServerSpec) -> bool:
        return self._project_active() if spec.source == "project" else self._global_active()

    def untrusted(self, source: str = "project") -> list[ServerSpec]:
        """The servers of `source` (project, or global while a launch marker holds it) that wait for the trust prompt:
        none once trusted or withdrawn."""
        return [s for s in self._setup.servers if s.source == source and s.name not in self._withdrawn and not self._trusted(s)]

    def sources(self) -> dict[str, str]:
        return {spec.name: spec.source for spec in self._setup.servers}

    def withdraw(self, names: Sequence[str], reason: str) -> None:
        """Stop these servers now and never start them again in this run."""
        for name in names:
            self._withdrawn[name] = reason
            if (server := self._servers.get(name)) is not None:
                server.withdraw(reason)  # safe from the config guard's thread: see `LspServer.withdraw`

    async def close(self) -> None:
        await asyncio.gather(*(s.stop() for s in self._servers.values()), return_exceptions=True)

    def kill_all(self) -> None:
        for server in self._servers.values():
            with contextlib.suppress(Exception):
                server.kill()
