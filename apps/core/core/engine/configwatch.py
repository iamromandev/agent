"""The config watch: a fingerprint of the agent's own configuration and trust files, compared around everything
that runs a user command, so a change a child process made is seen.

UI-free and never raises: a file that cannot be read is an `error` entry, a vanished folder is `missing`.
The fingerprint is stat-first and hashes a regular file only when its size, mtime, ctime or inode differ from the
cached entry (at most `MAX_CHECK_HASH_BYTES` per check; past that a file whose stat moved counts as changed).
Project paths are never followed (`lstat`: the project loaders refuse links, so a link is fingerprinted as one);
global and trust-store paths are followed (dotfile setups link them, and the loaders follow them): a link's entry
also holds its target's state, and a linked folder is walked, within the same bounds. A link loop is an error entry.
Inside a walk, `.git` folders and `node_modules` folders below the top level are skipped: the loaders never read them.
A trust store counts only by this workspace's entry, so another folder's trust decisions are not a change.
The workspace `.env` is not watched: it is not a trust input (no setting it holds can trust anything).
The reaction to a change lives in `configguard.py`."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import itertools
import os
import stat
import threading
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from core.engine.fileio import parse_json, read_regular
from core.engine.text import escape

MAX_ENTRIES = 2000  # entries recorded per watched folder; the rest only counts toward its overflow entry
MAX_DEPTH = 6  # levels below a watched folder
MAX_HASH_BYTES = 1 << 20  # a larger file is compared by its stat only
MAX_CHECK_HASH_BYTES = 32 << 20  # bytes hashed per check; past that a file whose stat moved counts as changed
OVERFLOW_SCAN = 50_000  # entries past the bound that one walk looks at (stat only); the rest is a "+" on the count
_COUNT_DEPTH = 32  # how deep the overflow scan looks
KEY_CHARS = 80  # a tool's key argument in a cause
OVERFLOW = "(more entries)"

Area = Literal["project", "global", "trust", "marker"]
Kind = Literal["file", "dir", "link", "missing", "error", "overflow", "store"]
FOLLOWED: frozenset[Area] = frozenset({"global", "trust"})  # areas whose links are followed


class TrustRevoked(OSError):
    """Project trust was revoked in this run (the config watch saw the agent's config change): nothing may trust
    the project again or write to its settings file until the next launch."""

    def __init__(
        self,
        what: str = "this",
        *,
        reason: str = "project trust was revoked in this run (the agent's config was changed by a command)",
    ) -> None:
        super().__init__(f"{what} was refused: {reason}")


@dataclass(frozen=True)
class Entry:
    """One path's fingerprint."""

    kind: Kind
    size: int = 0
    mtime_ns: int = 0
    ino: int = 0
    digest: str | None = None  # sha256 of a regular file up to MAX_HASH_BYTES
    detail: str = ""  # a link's target, an errno name, an overflow count and hash
    ctime_ns: int = 0
    target: Entry | None = None  # a followed link (global and trust areas): its target's state

    def same(self, other: Entry) -> bool:
        """Is `other` the same state? A touch that leaves the bytes alone is not a change."""
        if self.kind != other.kind or self.detail != other.detail:
            return False
        if self.kind == "file":
            if self.digest is not None and other.digest is not None:
                return self.digest == other.digest and self.size == other.size
            return self._stamp() == other._stamp() and self.digest == other.digest
        if self.kind == "store":  # a trust store: only this workspace's entry counts (`detail` is its state)
            return self.digest == other.digest
        if self.kind == "dir":
            return self.ino == other.ino
        if self.kind == "link":
            if self.target is None or other.target is None:
                return self.target is other.target
            return self.target.same(other.target)
        if self.kind == "error":
            return self._stamp() == other._stamp()
        return True

    def _stamp(self) -> tuple[int, int, int, int]:
        return (self.size, self.mtime_ns, self.ctime_ns, self.ino)


def _base(env: Mapping[str, str], key: str, fallback: Path) -> Path:
    value = env.get(key)
    return Path(value) if value and Path(value).is_absolute() else fallback


WALKED = ("commands", "skills", "agents", "workflows")


@dataclass(frozen=True)
class WatchedSet:
    roots: tuple[tuple[Path, Area, bool], ...]  # (path, area, is a folder walk)
    # The trust stores are shared by every folder: with a workspace, only its own entry in them is fingerprinted.
    workspace: Path | None = None

    @classmethod
    def for_run(cls, workspace: Path, env: Mapping[str, str], home: Path) -> WatchedSet:
        """The files and folders a run watches: the project's and the global config (settings, mcp.json, commands,
        skills, agents, workflows), both trust stores and the global marker. Missing ones count (appearing is a
        change). Not `.env`: it is not a trust input."""
        project = workspace / ".agent"
        config = _base(env, "XDG_CONFIG_HOME", home / ".config") / "agent"
        data = _base(env, "XDG_DATA_HOME", home / ".local" / "share") / "agent"
        roots: list[tuple[Path, Area, bool]] = []
        for folder, area in ((project, "project"), (config, "global")):
            roots.append((folder / "settings.json", area, False))
            roots.append((folder / "mcp.json", area, False))
            roots.extend((folder / name, area, True) for name in WALKED)
        roots.extend((data / name, "trust", False) for name in ("permission-trust.json", "mcp-trust.json"))
        roots.append((data / "config-changed.json", "marker", False))
        return cls(tuple(roots), workspace)

    def area_of(self, path: Path) -> Area:
        for root, area, _ in self.roots:
            if path == root or root in path.parents:
                return area
        return "project"  # not reachable: every key comes from a root

    def root(self, area: Area, name: str) -> Path | None:
        """The watched root of `area` with this file name (e.g. the global `mcp.json`)."""
        return next((root for root, a, _ in self.roots if a == area and root.name == name), None)


def _name(exc: BaseException) -> str:
    code = exc.errno if isinstance(exc, OSError) else None
    return errno.errorcode.get(code, type(exc).__name__) if code else type(exc).__name__


def _error(exc: BaseException, path: Path | None = None, info: os.stat_result | None = None) -> Entry:
    """An unreadable path, with a stamp that moves when something is done behind the error: the path's own stat when
    known, else that of the nearest folder above it that can be stat'ed (a chmod there changes its ctime)."""
    if info is None and path is not None:
        for place in (path, *path.parents):
            try:
                info = os.lstat(place)
                break
            except OSError:
                continue
    if info is None:
        return Entry("error", detail=_name(exc))
    return Entry("error", info.st_size, info.st_mtime_ns, info.st_ino, detail=_name(exc), ctime_ns=info.st_ctime_ns)


class _Budget:
    """Bytes one check may still hash."""

    def __init__(self) -> None:
        self.left = MAX_CHECK_HASH_BYTES


def _hash(path: Path, size: int, follow: bool, budget: _Budget) -> str | None:
    """sha256 of a regular file. None for a file over the cap or past the check's budget (then compared by stat)."""
    if size > MAX_HASH_BYTES or size > budget.left:
        return None
    flags = os.O_RDONLY | os.O_NONBLOCK | (0 if follow else os.O_NOFOLLOW)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.EINVAL, "not a regular file")
        digest = hashlib.sha256()
        remaining = MAX_HASH_BYTES + 1
        while remaining > 0 and (chunk := os.read(fd, min(65536, remaining))):
            digest.update(chunk)
            remaining -= len(chunk)
        budget.left -= MAX_HASH_BYTES + 1 - remaining
        return digest.hexdigest()
    finally:
        os.close(fd)


def _node(path: Path, info: os.stat_result, cached: Entry | None, follow: bool, budget: _Budget) -> Entry:
    """A file, a folder or something else, from its stat (of the path itself, or of a followed link's target)."""
    mode = info.st_mode
    if stat.S_ISDIR(mode):
        return Entry("dir", ino=info.st_ino)
    if stat.S_ISREG(mode):
        if (
            cached is not None
            and cached.kind == "file"
            and cached._stamp() == (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
        ):
            return cached  # stat-first: nothing moved, nothing is read
        try:
            digest = _hash(path, info.st_size, follow, budget)
        except OSError as exc:
            return _error(exc, info=info)
        return Entry("file", info.st_size, info.st_mtime_ns, info.st_ino, digest, ctime_ns=info.st_ctime_ns)
    # a fifo, a socket, a device: never opened
    return Entry("error", info.st_size, info.st_mtime_ns, info.st_ino, detail="NOTREGULAR", ctime_ns=info.st_ctime_ns)


DirId = tuple[int, int]


def _stat(path: Path, cache: Mapping[Path, Entry], follow: bool, budget: _Budget) -> tuple[Entry, DirId | None]:
    """The entry of one path, and the id of the folder it is (or, followed, points to), if it is one."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return Entry("missing"), None
    except OSError as exc:
        return _error(exc, path), None
    cached = cache.get(path)
    if not stat.S_ISLNK(info.st_mode):
        entry = _node(path, info, cached, False, budget)
        return entry, ((info.st_dev, info.st_ino) if entry.kind == "dir" else None)
    try:
        dest = os.readlink(path)
    except OSError as exc:
        return _error(exc, path, info), None
    if not follow:
        return Entry("link", detail=dest), None
    try:
        target_info = os.stat(path)
    except FileNotFoundError:
        return Entry("link", detail=dest, target=Entry("missing")), None
    except OSError as exc:  # a loop (ELOOP) or an unreadable folder on the way
        return _error(exc, path, info), None
    previous = cached.target if cached is not None and cached.kind == "link" else None
    target = _node(path, target_info, previous, True, budget)
    folder = (target_info.st_dev, target_info.st_ino) if target.kind == "dir" else None
    return Entry("link", detail=dest, target=target), folder


def _skipped(child: os.DirEntry[str], depth: int) -> bool:
    """A `.git` folder anywhere in a walk, or a `node_modules` folder below the top level: the loaders never read them
    (a skill folder itself may be called `node_modules`, so only deeper ones go). A link of that name is still kept."""
    if child.name != ".git" and (child.name != "node_modules" or depth == 0):
        return False
    try:
        return child.is_dir(follow_symlinks=False)
    except OSError:
        return False


def _walk(root: Path, cache: Mapping[Path, Entry], out: dict[Path, Entry], follow: bool, budget: _Budget) -> None:
    """Record `root` and what is below it, at most MAX_ENTRIES entries and MAX_DEPTH levels; `.git` and deeper
    `node_modules` folders are skipped (`_skipped`). What lies beyond goes
    into one overflow entry: the count and a running sha256 of each entry's name and stat (no content), so an edit
    or a rename out there changes it; that scan stops after OVERFLOW_SCAN entries (the count then ends in "+")."""
    top, top_id = _stat(root, cache, follow, budget)
    out[root] = top
    if top_id is None:
        return
    visited: set[DirId] = {top_id}
    recorded = overflow = 0
    truncated = False
    digest = hashlib.sha256()
    cap = MAX_ENTRIES + OVERFLOW_SCAN
    looked = 0
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and not truncated:
        folder, depth = stack.pop()
        try:
            with os.scandir(folder) as it:
                children = sorted(itertools.islice(it, cap - looked + 1), key=lambda e: e.name)
        except OSError as exc:
            out[folder] = _error(exc, folder)
            continue
        for child in children:
            if _skipped(child, depth):
                continue
            if looked >= cap:
                truncated = True
                break
            looked += 1
            path = folder / child.name
            child_id: DirId | None = None
            if recorded < MAX_ENTRIES and depth < MAX_DEPTH:
                recorded += 1
                entry, child_id = _stat(path, cache, follow, budget)
                if child_id is not None and child_id in visited:
                    entry, child_id = Entry("error", detail="ELOOP"), None  # a link back to a folder already walked
                out[path] = entry
            else:
                overflow += 1
                try:
                    info = child.stat(follow_symlinks=False)
                    kind = stat.S_IFMT(info.st_mode)
                    stamp = f"{kind}\0{info.st_size}\0{info.st_mtime_ns}\0{info.st_ctime_ns}\0{info.st_ino}"
                    if stat.S_ISDIR(info.st_mode):
                        child_id = (info.st_dev, info.st_ino)
                except OSError as exc:
                    stamp = _name(exc)
                digest.update(os.fsencode(path.relative_to(root)) + b"\0" + stamp.encode() + b"\n")
            if child_id is not None and depth < _COUNT_DEPTH:
                visited.add(child_id)
                stack.append((path, depth + 1))
    if overflow or truncated:
        out[root / OVERFLOW] = Entry("overflow", detail=f"{overflow}{'+' if truncated else ''}:{digest.hexdigest()}")


def _store(path: Path, workspace: Path, cache: Mapping[Path, Entry], budget: _Budget) -> Entry:
    """A trust store by `workspace`'s entry only (links followed, like the store itself): `absent` (no file, or no
    entry), `unreadable` (not a regular JSON object of strings, which trusts nothing) or `entry` with the digest of
    the value. Other folders' entries changing is not a change of this one. Stat-first like a file."""
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return Entry("store", detail="absent")
    except OSError as exc:
        return _error(exc, path)
    stamp = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
    cached = cache.get(path)
    if cached is not None and cached.kind == "store" and cached._stamp() == stamp:
        return cached
    state, digest = "unreadable", None
    try:
        raw = read_regular(path, MAX_HASH_BYTES)
        budget.left -= len(raw)
        data = parse_json(raw.decode("utf-8"))
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        value = data.get(str(workspace))
        state = "absent" if value is None else "entry"
        digest = hashlib.sha256(value.encode()).hexdigest() if value is not None else None
    return Entry("store", info.st_size, info.st_mtime_ns, info.st_ino, digest, detail=state, ctime_ns=info.st_ctime_ns)


def fingerprint(watched: WatchedSet, cache: Mapping[Path, Entry]) -> dict[Path, Entry]:
    """The state of every watched path. Never raises; follows links only in the global and trust areas."""
    out: dict[Path, Entry] = {}
    budget = _Budget()
    for root, area, walk in watched.roots:
        follow = area in FOLLOWED
        try:
            if area == "trust" and watched.workspace is not None:
                out[root] = _store(root, watched.workspace, cache, budget)
            elif walk:
                _walk(root, cache, out, follow, budget)
            else:
                out[root] = _stat(root, cache, follow, budget)[0]
        except Exception as exc:  # a bug or an odd file system: the path counts as unreadable, the check goes on
            out[root] = _error(exc)
    return out


@dataclass(frozen=True)
class Cause:
    text: str  # escaped and cut: what ran when the change was seen

    @staticmethod
    def tool(name: str, key: str | None) -> Cause:
        shown = escape(name)
        return Cause(f"{shown} ({escape(key[:KEY_CHARS])})" if key else shown)

    @staticmethod
    def hook(event: str) -> Cause:
        return Cause(f"hook {escape(event)}")

    @staticmethod
    def background() -> Cause:
        return Cause("a background process or terminal (during the turn)")


@dataclass(frozen=True)
class Change:
    at: float
    cause: Cause
    paths: tuple[Path, ...]
    areas: frozenset[Area]


def _alias(root: Path) -> Path:
    """`root` with its folder resolved (not its last part: a link there is itself watched)."""
    try:
        return root.parent.resolve() / root.name
    except (OSError, RuntimeError):
        return root


class ConfigWatch:
    """The baseline and the comparisons. Check points call `check`; the agent's own writes call `accept(paths)` or run
    inside `own_write(paths)`. Synchronous and cheap (stat only) so it can run between any two steps; safe to call
    from several threads (one comparison at a time). `on_change` is called outside the lock; a change seen before
    there is one is kept and handed over by `listen`."""

    def __init__(self, watched: WatchedSet, on_change: Callable[[Change], None] | None = None) -> None:
        self._watched = watched
        self._on_change = on_change
        self._lock = threading.RLock()
        self._unheard: list[Change] = []
        self._aliases = tuple((root, _alias(root)) for root, _, _ in watched.roots)
        self._baseline = fingerprint(watched, {})

    @property
    def watched(self) -> WatchedSet:
        return self._watched

    def listen(self, on_change: Callable[[Change], None]) -> None:
        """Hear every change from now on, and those seen before there was a listener."""
        with self._lock:
            self._on_change = on_change
            unheard, self._unheard = self._unheard, []
        for change in unheard:
            self._deliver(on_change, change)

    def check(self, cause: Cause) -> Change | None:
        """Compare with the baseline. A difference becomes the new baseline (so it is reported once) and goes to
        `on_change`; a failing listener never breaks the caller."""
        return self._compare(cause, set())

    def accept(self, paths: Iterable[Path]) -> Change | None:
        """The agent wrote these watched paths itself: their new state (and a folder the write created on the way)
        becomes the baseline unreported. Any other difference is reported, as a background change, by this call."""
        keys = {key for path in paths if (key := self._key(path)) is not None}
        return self._compare(Cause.background(), keys)

    @contextmanager
    def own_write(self, paths: Iterable[Path]) -> Generator[None]:
        """Around the agent's own write of `paths`: whatever changed before is checked first (as background), and only
        what the block wrote to `paths` is accepted afterwards."""
        named = tuple(paths)
        self.check(Cause.background())
        try:
            yield
        finally:
            self.accept(named)

    def watches(self, path: Path) -> bool:
        """Is `path` (or a spelling of it through a resolved parent folder) a watched path?"""
        return self._key(path) is not None

    def _key(self, path: Path) -> Path | None:
        for root, alias in self._aliases:
            for base in (root, alias):
                if path == base or base in path.parents:
                    return root / path.relative_to(base)
        return None

    def _compare(self, cause: Cause, adopt: set[Path]) -> Change | None:
        with self._lock:
            before = self._baseline
            now = fingerprint(self._watched, before)
            differ = {p for p in before.keys() | now.keys() if p not in before or p not in now or not before[p].same(now[p])}
            adopted = set(adopt)
            for key in adopt:
                for parent in key.parents:  # folders the own write created on its way to the file
                    made = parent in now and now[parent].kind == "dir"
                    if not made or (parent in before and before[parent].kind != "missing"):
                        break
                    adopted.add(parent)
            changed = sorted(differ - adopted, key=str)
            self._baseline = now
            if not changed:
                return None
            change = Change(time.time(), cause, tuple(changed), frozenset(self._watched.area_of(p) for p in changed))
            on_change = self._on_change
            if on_change is None:
                self._unheard.append(change)
                return change
        self._deliver(on_change, change)
        return change

    @staticmethod
    def _deliver(on_change: Callable[[Change], None], change: Change) -> None:
        with contextlib.suppress(Exception):
            on_change(change)


class Watch(Protocol):
    """What a check point needs: `ConfigWatch`, or `NullWatch` for an agent or a rule set built without one."""

    def check(self, cause: Cause) -> Change | None: ...

    def accept(self, paths: Iterable[Path]) -> Change | None: ...

    def own_write(self, paths: Iterable[Path]) -> AbstractContextManager[None]: ...

    def watches(self, path: Path) -> bool: ...


class NullWatch:
    """A watch that sees nothing: the default for an agent or a rule set built without one."""

    def check(self, cause: Cause) -> Change | None:
        return None

    def accept(self, paths: Iterable[Path]) -> Change | None:
        return None

    @contextmanager
    def own_write(self, paths: Iterable[Path]) -> Generator[None]:
        yield

    def watches(self, path: Path) -> bool:
        return False
