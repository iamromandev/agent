"""The turns of every session of one daemon, with when they ran: what `/undo` and `/diff` use to tell the files another
session changed meanwhile from the ones this turn changed.

UI-free. Sessions of a daemon run turns at the same time in one workspace, so a file a turn changed may also have been
changed by a turn of another session that overlapped it in time. Undoing the first must not revert the second's work,
so the paths of every overlapping turn (of another owner, still running or not) are skipped and named. Overlap is by
time only: a path the undone turn really changed alone may be skipped too (a skip is preferred to a wrong revert).

An entry outlives its owner's undo history (`retire`: `/new`, a switch, compaction, a closed session): the work it
records is still in the workspace, and the other sessions' undo must still leave it alone. It is removed once it has
ended and no live entry of another owner overlaps it. While the turn runs the entry holds the turn itself (the
before/after snapshots and the file log fill in as it goes); once it ends only a slim copy is kept (snapshot ids, the
logged paths and the digests of what the turn wrote), so no file bytes are kept alive."""

from __future__ import annotations

import hashlib
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from core.engine.checkpoints.store import CheckpointStore, DiffEntry
from core.engine.text import escape
from core.engine.undo import CHANGED_BY_OTHER, MAYBE_BY_OTHER, UndoTurn

LEDGER_CAP = 500
_UNKNOWN = object()


@dataclass(frozen=True)
class _Slim:
    before: str | None
    after: str | None
    paths: frozenset[Path]
    digests: Mapping[Path, bytes | None]


class LedgerEntry:
    def __init__(self, owner: str, turn: UndoTurn, start: float) -> None:
        self.owner = owner  # the session (id, or a host id while it is unsaved)
        self.start = start
        self.end: float | None = None  # None while the turn runs
        self.retired = False  # its owner can no longer undo it; it only guards the others
        self._turn: UndoTurn | None = turn  # strong while the turn runs
        self._ref = weakref.ref(turn)  # lets `find` and an evicted undo record be told apart without keeping it alive
        self._slim: _Slim | None = None

    def seal(self) -> None:
        """The turn ended: keep what the others need and let go of the turn (and the file bytes in it)."""
        turn = self._turn
        if turn is None:
            return
        self._slim = _Slim(
            turn.before,
            turn.after,
            frozenset(turn.changes),
            {path: change.after for path, change in turn.changes.items()},
        )
        self._turn = None

    @property
    def before(self) -> str | None:
        return self._turn.before if self._turn is not None else self._slim.before if self._slim else None

    @property
    def after(self) -> str | None:
        return self._turn.after if self._turn is not None else self._slim.after if self._slim else None

    @property
    def paths(self) -> frozenset[Path]:
        """The files its write/edit calls changed (absolute)."""
        if self._turn is not None:
            return frozenset(self._turn.changes)
        return self._slim.paths if self._slim else frozenset()

    @property
    def digests(self) -> Mapping[Path, bytes | None]:
        """The sha256 of what each logged file held after its last write (None: not known or absent)."""
        if self._turn is not None:
            return {path: change.after for path, change in self._turn.changes.items()}
        return self._slim.digests if self._slim else {}

    def released(self) -> bool:
        """Its undo record is gone (evicted, cleared): nobody can undo it any more."""
        return self._turn is None and self._ref() is None


class Foreign(dict[str, str]):
    """`{workspace-relative path: reason}` plus `failures`: what could not be read (the answer is then incomplete)."""

    def __init__(self) -> None:
        super().__init__()
        self.failures: list[str] = []


class TurnLedger:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic, cap: int = LEDGER_CAP) -> None:
        self._clock = clock
        self._cap = cap
        self.entries: list[LedgerEntry] = []

    def begin(self, owner: str, turn: UndoTurn) -> LedgerEntry:
        entry = LedgerEntry(owner, turn, self._clock())
        self.entries.append(entry)
        self._prune()
        return entry

    def finish(self, entry: LedgerEntry) -> None:
        if entry.end is None:
            entry.end = self._clock()
        entry.seal()
        self._prune()

    def drop(self, entry: LedgerEntry) -> None:
        """The turn was undone or discarded: it no longer matters to anyone."""
        if entry in self.entries:
            self.entries.remove(entry)
        self._prune()

    def undone(self, entry: LedgerEntry) -> None:
        """The turn was undone. What it changed and the undo left alone (a skipped path) is still in the workspace, so
        the entry stays for the sessions whose turns overlapped it, like a retired one."""
        entry.retired = True
        self._prune()

    def retire(self, owner: str) -> None:
        """The owner's undo history is gone (`/new`, a session switch, compaction, a closed session): its entries stay
        for the other sessions' sake, and go once nothing live overlaps them."""
        for entry in self.entries:
            if entry.owner == owner:
                entry.retired = True
        self._prune()

    def find(self, turn: UndoTurn) -> LedgerEntry | None:
        return next((e for e in self.entries if e._ref() is turn), None)

    @staticmethod
    def _meet(a: LedgerEntry, b: LedgerEntry) -> bool:
        inf = float("inf")
        return a.start < (b.end if b.end is not None else inf) and b.start < (a.end if a.end is not None else inf)

    def _prune(self) -> None:
        for entry in self.entries:
            if entry.released():
                entry.retired = True
        live = [e for e in self.entries if not e.retired]

        def needed(entry: LedgerEntry) -> bool:
            return entry.end is None or any(o.owner != entry.owner and self._meet(o, entry) for o in live)

        self.entries = [e for e in self.entries if not e.retired or needed(e)]
        excess = len(self.entries) - self._cap
        if excess > 0:  # the oldest ended entries first, the retired ones before the undoable ones; never a running one
            ended = sorted((e for e in self.entries if e.end is not None), key=lambda e: (not e.retired, e.start))
            gone = {id(e) for e in ended[:excess]}
            self.entries = [e for e in self.entries if id(e) not in gone]

    def overlapping(self, entry: LedgerEntry) -> list[LedgerEntry]:
        """The entries of other owners whose time span meets `entry`'s (a running one lasts until now)."""
        return [other for other in self.entries if other.owner != entry.owner and self._meet(other, entry)]

    async def foreign_paths(self, entry: LedgerEntry, store: CheckpointStore | None, workspace: Path) -> Foreign:
        """The paths an overlapping turn of another owner changed, with why: its file log (`write`/`edit`, exact) and
        its `before` to `after` snapshot diff (to a fresh snapshot when it has no `after`: still running, or its undo
        record was cleared), which also holds `bash` changes. That diff cannot tell the turns apart: it holds
        everything that changed in the workspace meanwhile, `entry`'s own files included. A path `entry`'s file log
        names is left out of it only when the other turn's end state is exactly what `entry` wrote (the other turn did
        not touch it beyond that); one that matches `entry`'s own end state without being logged may be `entry`'s own
        bash change (`MAYBE_BY_OTHER`, skipped to be safe). When a diff cannot be read, `failures` says why. Only a
        cancel raises."""
        found = Foreign()
        others = self.overlapping(entry)
        if not others:
            return found
        root = workspace.resolve()
        own = {name.casefold(): digest for name, digest in _relative(entry.digests, root).items()}
        own_after: dict[str, str | None] | None = None
        now: str | None = None
        for other in sorted(others, key=lambda o: o.after is None):  # known snapshots first: no fresh one if none needs it
            logged = _relative(dict.fromkeys(other.paths), root)
            for name in logged:
                found.setdefault(name, CHANGED_BY_OTHER)
            before, after = other.before, other.after
            if store is None or not before:
                continue
            if after is None:
                if now is None:
                    snapshot = await store.snapshot()
                    now = snapshot.commit if snapshot is not None else ""
                after = now
            if not after:
                found.failures.append("no snapshot of the workspace could be taken")
                continue
            try:
                diffed = await store.diff(before, after)
                held = {name.casefold() for name in logged}
                exact: list[DiffEntry] = []
                for d in diffed:
                    fold = d.path.casefold()
                    if fold in held:
                        continue
                    if fold in own:
                        exact.append(d)
                        continue
                    if own_after is None:
                        own_after = await self._own_after(entry, store)
                    mine = own_after.get(d.path, _UNKNOWN)
                    found.setdefault(d.path, MAYBE_BY_OTHER if mine == d.new_sha else CHANGED_BY_OTHER)
                if exact:
                    await self._classify_logged(found, exact, own, store)
            except Exception as exc:
                found.failures.append(f"{type(exc).__name__}: {escape(str(exc))}"[:200])
        return found

    @staticmethod
    async def _own_after(entry: LedgerEntry, store: CheckpointStore) -> dict[str, str | None]:
        """The blob each path ended with at `entry`'s own `after` snapshot (only the paths it changed)."""
        if not entry.before or not entry.after:
            return {}
        return {d.path: d.new_sha for d in await store.diff(entry.before, entry.after)}

    @staticmethod
    async def _classify_logged(
        found: Foreign, exact: list[DiffEntry], own: Mapping[str, bytes | None], store: CheckpointStore
    ) -> None:
        """Paths `entry` logged and another turn's diff also holds: exempt when the diff's end state is the content
        `entry` wrote, else the other turn changed it too."""
        blobs = await store.read_blobs([d.new_sha for d in exact if d.new_sha])
        for d in exact:
            written = own[d.path.casefold()]
            content = blobs.get(d.new_sha) if d.new_sha else None
            if written is not None and content is not None and hashlib.sha256(content).digest() == written:
                continue
            found.setdefault(d.path, CHANGED_BY_OTHER)


def _relative(paths: Mapping[Path, bytes | None], root: Path) -> dict[str, bytes | None]:
    out: dict[str, bytes | None] = {}
    for path, digest in paths.items():
        try:
            out[path.relative_to(root).as_posix()] = digest
        except ValueError:
            continue
    return out
