"""What the agent does when the config watch sees a change a child process made: revoke project trust, switch off
global rules and hooks, stop the MCP servers whose file changed, leave a marker for the next launch, and tell the
listeners. UI-free; `react` never raises, and may be called from any thread (a trust write runs in one): the
switches flip at once, the MCP servers' tools are withdrawn at once, and closing them and the listeners run on the
event loop the guard knows."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from core.engine.configwatch import Change, ConfigWatch, TrustRevoked
from core.engine.fileio import parse_json, read_regular
from core.engine.mcp.trust import TrustStore, trust_store_path
from core.engine.permissions.setup import Rules
from core.engine.text import escape

MARKER = "config-changed.json"  # in the agent's data folder: {"marker": 1, "at": <iso time>, "files": [...]}
MAX_SHOWN = 10  # files named in one warning
MAX_SUMMARY = 20  # changes listed by `/guardrails`, then a count
STOP_REASON = "config changed"
PROJECT_REACTION = "project trust revoked — the next launch will ask"
GLOBAL_REACTION = "global allow rules and hooks are off until you trust them at the next launch"
MAX_NESTING = 4  # reactions to changes seen by the guard's own accepts, one inside the other
MAX_MARKER_BYTES = 64 << 10  # the guard writes at most 100 short names
NO_CHANGES = "Config watch: no changes this run"
HELD_PREFIX = "global config held until trusted"
NEXT_LAUNCH = "keepEnv and the global commands, skills, agents and workflows return at the next launch (they are read once at launch)"


def _count(n: int, what: str) -> str:
    return f"{n} {what}{'s' * (n != 1)}"


@dataclass(frozen=True)
class HeldGlobal:
    """What the marker holds besides the global allow rules, hooks and MCP servers (those come back when the user
    trusts the global config): the global `keepEnv` and the global commands, skills, agents and workflows, which are
    read once at launch and so return only at the next one. Names are raw: whoever shows them escapes them."""

    keep_env: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    workflows: tuple[str, ...] = ()

    @property
    def any(self) -> bool:
        return bool(self.keep_env or self.commands or self.skills or self.agents or self.workflows)

    def kinds(self) -> list[tuple[str, tuple[str, ...]]]:
        """(singular name, names) of each loaded kind that is held."""
        pairs = (("command", self.commands), ("skill", self.skills), ("agent", self.agents), ("workflow", self.workflows))
        return [(kind, names) for kind, names in pairs if names]

    def notices(self) -> list[str]:
        """The app's launch notices: one for keepEnv and one per kind, each with its count."""
        later = "at the next launch after you trust it"
        lines: list[str] = []
        if self.keep_env:
            lines.append(
                f"{HELD_PREFIX}: global keepEnv ignored ({_count(len(self.keep_env), 'name')}), so those variables are "
                f"removed from child environments; it returns {later}"
            )
        for kind, names in self.kinds():
            lines.append(
                f"{HELD_PREFIX}: {_count(len(names), 'global ' + kind)} not loaded; "
                f"{'they return' if len(names) != 1 else 'it returns'} {later}"
            )
        return lines


def held_warning(held: HeldGlobal) -> str:
    """Headless mode's one warning when the marker is set at launch (it never asks): everything that is held."""
    parts = ["global allow rules, hooks and MCP servers are off"]
    if held.keep_env:
        parts.append(f"global keepEnv is ignored ({_count(len(held.keep_env), 'name')})")
    if kinds := held.kinds():
        parts.append("not loaded: " + ", ".join(_count(len(names), "global " + kind) for kind, names in kinds))
    return (
        "global config was changed by a command in an earlier run; until you trust it in the app, "
        + "; ".join(parts)
    )


def marker_path(env: Mapping[str, str] = os.environ) -> Path:
    """The marker file: next to the trust stores (`$XDG_DATA_HOME/agent`, else `~/.local/share/agent`)."""
    return trust_store_path(env, MARKER)


def _problem(exc: BaseException) -> str:
    code = exc.errno if isinstance(exc, OSError) else None
    return errno.errorcode.get(code, type(exc).__name__) if code else type(exc).__name__


class McpControl(Protocol):
    """The hub's side of a reaction (`McpHub`)."""

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None: ...

    def sources(self) -> Mapping[str, str]: ...

    def withdraw(self, names: Sequence[str], reason: str) -> Sequence[str]: ...

    async def stop(self, names: Sequence[str], reason: str) -> None: ...


class GlobalMarker:
    """A file that says the global config was changed by a child process: while it exists the next launch asks
    before using the global allow rules, hooks and MCP servers. `active`: set at launch or written in this run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.active = self.exists()
        # The files named so far (at launch and by this run's writes): a rewrite keeps them, whatever a command left
        # in the file meanwhile.
        info = self.read() if self.active else None
        self._files: list[str] = list(info.files) if info is not None else []

    def exists(self) -> bool:
        return os.path.lexists(self.path)

    def write(self, files: Sequence[str]) -> None:
        """Atomic (temp file and rename). The names already recorded come first, then the new ones, each once.
        Raises OSError."""
        merged = list(dict.fromkeys([*self._files, *files]))[: MAX_SHOWN * 10]
        body = {"marker": 1, "at": datetime.now(UTC).isoformat(timespec="seconds"), "files": merged}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.stem}-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(body, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        self._files = merged
        self.active = True

    def clear(self) -> None:
        """Remove the marker (the user trusted the global config). A missing marker is fine; other errors raise."""
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        self.active = False  # the names are kept: a marker written again (a refused trust) still lists them

    def read(self) -> MarkerInfo | None:
        """What the marker says (None: there is none). A marker that cannot be read or parsed is still set, with
        nothing known; odd values are left out. The strings are raw: whoever shows them escapes them."""
        if not self.exists():
            return None
        try:
            data = parse_json(read_regular(self.path, MAX_MARKER_BYTES).decode("utf-8"))
        except (OSError, ValueError):
            return MarkerInfo(None, ())
        if not isinstance(data, dict):
            return MarkerInfo(None, ())
        at: datetime | None = None
        if isinstance(raw := data.get("at"), str):
            with contextlib.suppress(ValueError):
                at = datetime.fromisoformat(raw)
                at = at if at.tzinfo is not None else at.replace(tzinfo=UTC)
        files = data.get("files")
        listed = tuple(f for f in files if isinstance(f, str)) if isinstance(files, list) else ()
        return MarkerInfo(at, listed[: MAX_SHOWN * 10])


@dataclass(frozen=True)
class MarkerInfo:
    at: datetime | None  # when the change was seen (aware); None when the marker does not say
    files: tuple[str, ...]  # as the warning showed them, raw


class ConfigGuard:
    """The reaction object: `guard.bind(watch)` makes the watch report to `react` and lets the guard accept its own
    writes to the trust stores and the marker (each right after it is written, so a child's write in between is
    still reported)."""

    def __init__(
        self,
        workspace: Path,
        rules: Rules,
        mcp: McpControl | None,
        stores: Sequence[TrustStore],
        marker: GlobalMarker,
        *,
        home: Path | None = None,
    ) -> None:
        self._workspace = workspace
        self._rules = rules
        self.mcp = mcp  # may be set later: the hub is built after the guard in some setups
        self._stores = tuple(stores)
        self._marker = marker
        self._home = home if home is not None else Path.home()
        self._watch: ConfigWatch | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._depth = threading.local()
        self._pending: set[asyncio.Task[None]] = set()
        self.listeners: list[Callable[[str], None]] = []  # told the warning text; a failing one is skipped
        self.changes: list[Change] = []  # every change seen in this run, for `/guardrails`
        self._backlog: list[str] = []  # texts told while nobody listened (a change seen during the launch)
        self._tell_lock = threading.Lock()

    def listen(self, listener: Callable[[str], None]) -> None:
        """Hear every warning from now on; the first listener also hears those said before there was one."""
        with self._tell_lock:
            self.listeners.append(listener)
            backlog, self._backlog = self._backlog, []
        for text in backlog:
            self._deliver([listener], text, None)

    def unlisten(self, listener: Callable[[str], None]) -> None:
        """Stop telling `listener` (the app's transcript is going away): what is said next is kept (`undelivered`)."""
        with self._tell_lock, contextlib.suppress(ValueError):
            self.listeners.remove(listener)

    def undelivered(self) -> list[str]:
        """The warnings no live listener heard (said with none, or every listener failed), once: the app prints them
        on stderr after it has quit, so a change seen by the quit checks is not lost."""
        with self._tell_lock:
            kept, self._backlog = self._backlog, []
        return kept

    def trust_global(self) -> None:
        """The user trusted the global config in the launch dialog: the marker goes (the agent's own write) and the
        global allow rules and hooks are on again. Raises `TrustRevoked` (the marker stays, everything stays off) when
        a change was seen in this run, also by the check that opens the write; `OSError` when the marker cannot be
        removed (then nothing is turned on)."""
        if self._watch is None:
            raise TrustRevoked("trusting the global config", reason="the config watch is not running")
        refused = TrustRevoked("trusting the global config", reason="the agent's config was changed by a command in this run")
        with self._watch.own_write([self._marker.path]):
            if self._rules.revoked or self._rules.global_revoked:  # seen by the check that opened the write
                raise refused
            try:
                self._marker.clear()
            except OSError as exc:  # a folder in its place, an unwritable data folder: one clear line, nothing on
                raise OSError(f"could not clear the marker: {_problem(exc)}; remove {self._marker.path} by hand") from exc
        try:
            self._rules.enable_global()
        except TrustRevoked:  # the check after the write saw a change: the marker is needed again
            if not self._marker.exists():
                with contextlib.suppress(Exception), self._watch.own_write([self._marker.path]):
                    self._marker.write([])
            raise refused from None

    def summary(self) -> list[str]:
        """The `/guardrails` lines: the changes of this run (local time, cause, files) and the marker's state."""
        if not self.changes:
            lines = [NO_CHANGES]
        else:
            count = len(self.changes)
            lines = [f"Config watch: {count} change{'s' * (count != 1)} this run"]
            for change in self.changes[:MAX_SUMMARY]:
                stamp = datetime.fromtimestamp(change.at).strftime("%H:%M:%S")
                lines.append(f"  {stamp} {change.cause.text}: {self._files(change.paths)}")
            if count > MAX_SUMMARY:
                lines.append(f"  and {count - MAX_SUMMARY} more")
        info = self._marker.read()
        if info is not None:
            since = info.at.astimezone().strftime("%Y-%m-%d %H:%M:%S") if info.at is not None else "an unknown time"
            lines.append(f"global marker: set (since {since})")
        return lines

    def _files(self, paths: Sequence[Path]) -> str:
        shown = [self.show(p) for p in paths[:MAX_SHOWN]]  # cut first, then escaped
        more = len(paths) - MAX_SHOWN
        return ", ".join(shown) + (f" and {more} more" if more > 0 else "")

    def bind(self, watch: ConfigWatch) -> None:
        self._watch = watch
        self._remember_loop()
        watch.listen(self.react)

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """The loop that owns the hub and the listeners (also learnt from any reaction that runs on a loop)."""
        self._loop = loop

    @property
    def marker_set(self) -> bool:
        return self._marker.exists()

    def marker_info(self) -> MarkerInfo | None:
        """What the global marker says, for the launch dialog (None: it is not set)."""
        return self._marker.read()

    async def settle(self) -> None:
        """Wait for the MCP stops started by `react` (the app awaits them at quit; tests await them)."""
        self._remember_loop()
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    def show(self, path: Path) -> str:
        """A path as the warning shows it: below the workspace relative, below the home folder as `~/`, else absolute."""
        for base, prefix in ((self._workspace, ""), (self._home, "~/")):
            with contextlib.suppress(ValueError):
                return escape(prefix + path.relative_to(base).as_posix())
        return escape(str(path))

    def react(self, change: Change) -> None:
        """Flip the switches, write what has to be written, tell the listeners. Never raises."""
        self._remember_loop()
        depth = getattr(self._depth, "value", 0)
        if depth == 0:
            self._depth.later = []
        self._depth.value = depth + 1
        try:
            try:
                text = self._react(change, nested=depth >= MAX_NESTING)
            except Exception as exc:  # a bug here must not cost the call that was being checked
                text = f"config changed by {change.cause.text}; the reaction failed ({_problem(exc)})"
        finally:
            self._depth.value = depth
        if depth > 0:  # seen by one of the guard's own accepts: told after the change that caused the reaction
            self._depth.later.append((change, text))
            return
        later, self._depth.later = self._depth.later, []
        for seen, said in ((change, text), *later):
            self.changes.append(seen)
            self._tell(said)

    def _react(self, change: Change, *, nested: bool) -> str:
        areas = change.areas
        project = bool(areas & {"project", "trust"})
        changed_global = "global" in areas
        problems: list[str] = []
        # The switches first: nothing below can keep them from flipping.
        if project:
            self._safely(self._rules.revoke_project, "could not revoke the project's rules", problems)
        if changed_global:
            self._safely(self._rules.disable_global, "could not switch off the global rules", problems)
        names: list[str] = []
        sources: Mapping[str, str] = {}
        with contextlib.suppress(Exception):
            sources = self.mcp.sources() if self.mcp is not None else {}
        if project:
            names += [n for n, source in sources.items() if source == "project"]
        global_mcp = self._watch.watched.root("global", "mcp.json") if self._watch is not None else None
        if changed_global and global_mcp is not None and global_mcp in change.paths:
            names += [n for n, source in sources.items() if source == "global"]
        if names:
            try:
                self._stop(names)
            except Exception as exc:
                problems.append(f"could not stop the MCP servers ({_problem(exc)})")
        reactions: list[str] = []
        if project:
            failed = self._forget(problems, nested)
            reactions.append(
                PROJECT_REACTION if not failed else "project trust revoked for this run; the next launch will NOT ask: " + "; ".join(failed)
            )
        if changed_global:
            failed_marker = self._write_marker(change, nested)
            reactions.append(
                GLOBAL_REACTION
                if failed_marker is None
                else "global allow rules and hooks are off for this run; the next launch will NOT ask about the global "
                f"config: could not write {MARKER} ({failed_marker})"
            )
        elif "marker" in areas:
            if self._marker.active:
                failed_marker = self._write_marker(change, nested)
                reactions.append(
                    "the global marker was changed by a command and is written again"
                    if failed_marker is None
                    else f"the global marker was changed by a command; could not write it again ({failed_marker})"
                )
            else:
                reactions.append("the global marker file was changed by a command")
        reaction = "; ".join(reactions)
        if problems:
            reaction += " (" + "; ".join(problems) + ")"
        return f"config changed by {change.cause.text}: {self._files(change.paths)}; {reaction}"

    @staticmethod
    def _safely(step: Callable[[], None], what: str, problems: list[str]) -> None:
        try:
            step()
        except Exception as exc:
            problems.append(f"{what} ({_problem(exc)})")

    def _accept(self, path: Path, nested: bool) -> None:
        """The guard wrote `path`: accept it (any other change seen meanwhile is reported, one level deeper). Past the
        nesting limit the next check reports instead."""
        if self._watch is not None and not nested:
            self._watch.accept([path])

    def _forget(self, problems: list[str], nested: bool) -> list[str]:
        """Remove the folder from both stores. Returns what failed so that the next launch will NOT ask; notes go to
        `problems`. A store that cannot be updated is deleted instead; one that a command wrote again meanwhile too."""
        failed: list[str] = []
        for store in self._stores:
            name = escape(store.path.name)
            try:
                wrote = store.forget(self._workspace)
            except Exception as exc:
                if self._delete(store.path):
                    problems.append(f"could not update {name} ({_problem(exc)}); deleted it instead")
                    self._accept(store.path, nested)
                else:
                    failed.append(f"could not update {name} ({_problem(exc)})")
                continue
            if wrote:
                self._accept(store.path, nested)
            try:
                again = store.holds(self._workspace)
            except Exception:
                again = False
            if again:  # written again between the guard's write and its accept
                if self._delete(store.path):
                    problems.append(f"{name} was written again during the revoke; deleted it")
                    self._accept(store.path, nested)
                else:
                    failed.append(f"{name} was written again during the revoke and could not be deleted")
        return failed

    @staticmethod
    def _delete(path: Path) -> bool:
        try:
            os.unlink(path)
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    def _write_marker(self, change: Change, nested: bool) -> str | None:
        """Write the marker; None, or the error's name."""
        try:
            self._marker.write([self.show(p) for p in change.paths[: MAX_SHOWN * 10]])
        except Exception as exc:
            return _problem(exc)
        self._accept(self._marker.path, nested)
        return None

    def _remember_loop(self) -> None:
        with contextlib.suppress(RuntimeError):
            self._loop = asyncio.get_running_loop()

    def _known_loop(self) -> asyncio.AbstractEventLoop | None:
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        hub_loop = self.mcp.loop if self.mcp is not None else None
        return hub_loop if hub_loop is not None and not hub_loop.is_closed() else None

    @staticmethod
    def _running() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def _stop(self, names: Sequence[str]) -> None:
        """Withdraw the servers now (any thread); close them on the loop that owns them."""
        mcp = self.mcp
        if mcp is None:
            return
        mcp.withdraw(names, STOP_REASON)
        running = self._running()
        if running is not None:
            self._spawn(names)
            return
        loop = self._known_loop()
        if loop is not None:
            loop.call_soon_threadsafe(self._spawn, list(names))

    def _spawn(self, names: Sequence[str]) -> None:
        if self.mcp is None:
            return
        task = asyncio.get_running_loop().create_task(self.mcp.stop(names, STOP_REASON))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def _tell(self, text: str) -> None:
        """Every listener hears the text: on the loop when one is known and this is another thread."""
        loop = None if self._running() is not None else self._known_loop()
        with self._tell_lock:
            listeners = list(self.listeners)
            if not listeners:  # nobody is listening (yet, or any more): the first `listen` or `undelivered` gets it
                self._backlog.append(text)
                return
        self._deliver(listeners, text, loop)

    def _deliver(self, listeners: Sequence[Callable[[str], None]], text: str, loop: asyncio.AbstractEventLoop | None) -> None:
        """Tell each listener (on `loop` when given); when every one of them fails, the text is kept."""
        left = [len(listeners)]  # listeners still to be told
        heard = [False]  # did one of them take it

        def call(listener: Callable[[str], None]) -> None:
            try:
                listener(text)
                ok = True
            except Exception:
                ok = False
            with self._tell_lock:
                heard[0] = heard[0] or ok
                left[0] -= 1
                if not left[0] and not heard[0]:
                    self._backlog.append(text)

        for listener in listeners:
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(call, listener)
                    continue
                except RuntimeError:  # the loop closed meanwhile
                    pass
            call(listener)
