"""What the newest undoable turn changed in the workspace: a file list built without reading whole files, and the
unified diff of one file loaded on demand.

UI-free. The source is the turn's two checkpoint snapshots (`CheckpointStore.diff`, then blob sizes, a binary scan and
`numstat`, then `read_blobs` for the file that is opened); without a complete pair it is the file-level undo log
(`write`/`edit` changes only). Nothing outside the shadow repository and the log is read, except the current content of
a file the log recorded, and only when its digest is still the one the turn left.

The line-by-line comparison is pure Python and quadratic in the worst case, so it is bounded: sides over `DIFF_BYTES` or
`DIFF_LINES`, a work estimate over `WORK_LIMIT` and a deadline each end in `too large to compare`, and it runs in a
thread that stops at its next check when its caller is cancelled."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import itertools
import os
import stat
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.engine.checkpoints.git import GitError
from core.engine.checkpoints.store import CheckpointStore, DiffEntry
from core.engine.text import escape_display
from core.engine.undo import SNAPSHOT_FILE_LIMIT, FileChange, UndoTurn, read_nofollow

LINE_LIMIT = 5_000
FILE_LIST_LIMIT = 2_000
CONTEXT = 3
DIFF_BYTES = 1024 * 1024  # a side over this is not compared, and numstat does not count it
DIFF_LINES = 20_000  # lines per side
LINE_CHARS = 2_000  # characters of one displayed line
WORK_LIMIT = 2_000_000  # matcher steps estimated for the first search (about 0.15 s of difflib)
DIFF_SECONDS = 0.5  # one file
LIST_SECONDS = 2.0  # every file of a list built from the log
MAX_SKIPPED = 200  # paths excluded from one numstat; above this no counts are asked for
BANNER = "checkpoints off: bash and terminal changes are not shown"
NO_FILES = "The last turn changed no files"
NOTHING = "Nothing to diff"
TOO_BIG = "too large to compare"
NO_NEWLINE = "\\ No newline at end of file"

Status = Literal["A", "D", "M", "T"]  # T: type change (file <-> symlink)
LineKind = Literal["add", "del", "hunk", "meta", "ctx"]

SYMLINK = 0o120000


@dataclass(frozen=True)
class DiffItem:
    path: str  # workspace-relative posix, raw: escape when shown
    status: Status
    added: int | None  # None: unknown (binary, too large, unavailable, git could not count)
    removed: int | None
    note: str | None = None  # "binary" | "too large" | "too large to compare" | "unavailable" | "mode 100644 → 100755"
    old_mode: int = 0
    new_mode: int = 0
    old_sha: str | None = None  # checkpoint source
    new_sha: str | None = None
    old_size: int | None = None  # bytes, when known
    new_size: int | None = None
    old_bytes: bytes | None = None  # fallback source: what the log kept before the turn
    new_path: Path | None = None  # fallback: read now, only when its sha256 is still `new_digest`
    new_digest: bytes | None = None


@dataclass(frozen=True)
class TurnDiff:
    items: tuple[DiffItem, ...]  # sorted by path, at most FILE_LIST_LIMIT
    more: int  # files beyond the limit
    fallback: bool  # built from the file log: bash and terminal changes are missing
    store: CheckpointStore | None
    workspace: Path
    turn: UndoTurn  # the turn this is about: the caller checks it is still the newest before showing it


@dataclass(frozen=True)
class FileDiff:
    item: DiffItem
    meta: tuple[str, ...]  # raw lines about the file as a whole (binary sizes, symlink targets, mode, unavailable)
    lines: tuple[str, ...]  # unified diff lines without the line ending, at most LINE_LIMIT
    cut: int  # lines not shown


@dataclass(frozen=True)
class DiffLine:
    text: str  # escape_display'ed
    kind: LineKind


class TooLarge(Exception):
    """The comparison was refused: a side over a limit, too much work estimated, or the deadline passed."""


class Cancelled(Exception):
    """The caller gave up: the thread stops at its next check."""


class Budget:
    """When a bounded comparison must stop: a deadline, or the caller's cancel event."""

    def __init__(self, cancel: threading.Event, seconds: float) -> None:
        self._cancel = cancel
        self._deadline = time.monotonic() + seconds

    def within(self, seconds: float) -> Budget:
        """A budget for one file inside this one: the earlier of the two deadlines, the same cancel event."""
        inner = Budget(self._cancel, seconds)
        inner._deadline = min(inner._deadline, self._deadline)
        return inner

    def expired(self) -> bool:
        return time.monotonic() > self._deadline

    def check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled
        if self.expired():
            raise TooLarge("deadline")


async def _in_thread[T](work: Callable[[Budget], T], seconds: float) -> T:
    """`work(budget)` in a thread; cancelling the awaiting task sets the budget's cancel event, so the thread ends at its
    next check instead of running on (the app's exit waits for the default executor's threads)."""
    cancel = threading.Event()
    budget = Budget(cancel, seconds)
    try:
        return await asyncio.to_thread(work, budget)
    except asyncio.CancelledError:
        cancel.set()
        raise


def _kind(mode: int) -> int:
    return stat.S_IFMT(mode)


def _mode_note(old_mode: int, new_mode: int) -> str:
    return f"mode {old_mode:o} → {new_mode:o}"


@dataclass(frozen=True)
class _Facts:
    """What the list needs beyond the raw entries; every part may be missing when git could not answer."""

    sizes: dict[str, int]  # blob sha -> bytes
    binary: frozenset[str]  # blob shas with a NUL or invalid UTF-8
    counts: dict[str, tuple[int, int] | None]  # path -> added, removed (only the paths git counted)


def _item_from_entry(entry: DiffEntry, facts: _Facts) -> DiffItem:
    if entry.old_sha is None:
        status: Status = "A"
    elif entry.new_sha is None:
        status = "D"
    elif _kind(entry.old_mode) != _kind(entry.new_mode):
        status = "T"
    else:
        status = "M"
    old_size = facts.sizes.get(entry.old_sha) if entry.old_sha else None
    new_size = facts.sizes.get(entry.new_sha) if entry.new_sha else None
    sha_sides = [sha for sha in (entry.old_sha, entry.new_sha) if sha]
    note: str | None = None
    added: int | None = None
    removed: int | None = None
    if entry.old_sha is not None and entry.old_sha == entry.new_sha:  # same content, other mode or type
        note = _mode_note(entry.old_mode, entry.new_mode)
        added, removed = 0, 0
    else:
        links = SYMLINK in (_kind(entry.old_mode), _kind(entry.new_mode))
        if any(facts.sizes.get(sha, 0) > DIFF_BYTES for sha in sha_sides):
            note = TOO_BIG
        elif not links and any(sha in facts.binary for sha in sha_sides):
            note = "binary"
        else:
            found = facts.counts.get(entry.path)
            if found is not None:
                added, removed = found
        if status == "T":
            note = note or _mode_note(entry.old_mode, entry.new_mode)
    return DiffItem(
        entry.path, status, added, removed, note,
        entry.old_mode, entry.new_mode, entry.old_sha, entry.new_sha, old_size, new_size,
    )  # fmt: skip


def _finish(
    items: Sequence[DiffItem], fallback: bool, store: CheckpointStore | None, workspace: Path, turn: UndoTurn
) -> TurnDiff:
    ordered = sorted(items, key=lambda i: i.path)
    return TurnDiff(
        tuple(ordered[:FILE_LIST_LIMIT]), max(0, len(ordered) - FILE_LIST_LIMIT), fallback, store, workspace, turn
    )


async def build_diff(turn: UndoTurn, store: CheckpointStore | None, workspace: Path) -> TurnDiff:
    """The file list of `turn`. Whole files are read only by the fallback (the files its log recorded, to count).

    From the snapshots when the turn has both; if git fails and the file log has entries, from the log. Raises
    GitError (or OSError) when there is nothing to fall back on. A failure of the counting steps alone is not one: the
    files are listed with unknown counts."""
    if turn.before and turn.after and store is not None:
        try:
            return await _from_checkpoints(turn, store, workspace)
        except (GitError, OSError):
            if not turn.changes:
                raise
    return await _in_thread(lambda budget: _from_log(turn, workspace, budget), LIST_SECONDS)


async def _from_checkpoints(turn: UndoTurn, store: CheckpointStore, workspace: Path) -> TurnDiff:
    assert turn.before and turn.after
    entries = await store.diff(turn.before, turn.after)
    facts = await _facts(store, turn.before, turn.after, entries)
    items = [_item_from_entry(e, facts) for e in entries]
    listed = {i.path for i in items}
    large_before = set(turn.large_before)
    for name in turn.skipped_large:  # over the snapshot cap: not in the snapshots, but the turn made or changed it
        if name not in listed:
            items.append(DiffItem(name, "M" if name in large_before else "A", None, None, "too large"))
    return _finish(items, False, store, workspace, turn)


async def _facts(store: CheckpointStore, before: str, after: str, entries: Sequence[DiffEntry]) -> _Facts:
    """Sizes, binary blobs and line counts for the list. Any git failure here (a timeout included) leaves that part
    unknown: the files are still listed, and a file is never given counts that were not checked for binary content."""
    shas = sorted({sha for e in entries for sha in (e.old_sha, e.new_sha) if sha})
    try:
        sizes = await store.blob_sizes(shas)
    except (GitError, OSError):
        return _Facts({}, frozenset(), {})
    big = sorted(e.path for e in entries if any(sizes[sha] > DIFF_BYTES for sha in (e.old_sha, e.new_sha) if sha))
    big_paths = set(big)
    small = sorted(
        {
            sha
            for e in entries
            if e.path not in big_paths and e.old_sha != e.new_sha and SYMLINK not in (_kind(e.old_mode), _kind(e.new_mode))
            for sha in (e.old_sha, e.new_sha)
            if sha
        }
    )
    try:
        binary = await store.binary_blobs(small)
    except (GitError, OSError):
        return _Facts(sizes, frozenset(), {})
    counts: dict[str, tuple[int, int] | None] = {}
    if len(big) <= MAX_SKIPPED:
        try:
            counts = await store.numstat(before, after, big)
        except (GitError, OSError):
            counts = {}
    return _Facts(sizes, frozenset(binary), counts)


def _inside(path: Path, root: Path) -> bool:
    return os.path.realpath(path) == str(path) and path.is_relative_to(root)


def _current(path: Path, digest: bytes | None, root: Path) -> bytes | None:
    """The file as the turn left it: None when it changed since (or cannot be read safely, or is over the log's cap)."""
    if not _inside(path, root):
        return None
    try:
        data = read_nofollow(path, SNAPSHOT_FILE_LIMIT)
    except OSError:
        return None
    return data if hashlib.sha256(data).digest() == digest else None


def _from_log(turn: UndoTurn, workspace: Path, budget: Budget) -> TurnDiff:
    root = workspace.resolve()
    items: list[DiffItem] = []
    for change in turn.changes.values():
        found = _item_from_change(change, root, budget)
        if found is not None:
            items.append(found)
    return _finish(items, True, None, root, turn)


def _item_from_change(change: FileChange, root: Path, budget: Budget) -> DiffItem | None:
    """None when the turn left the file as it found it (written back) or made and removed it."""
    try:
        name = change.path.relative_to(root).as_posix()
    except ValueError:
        name = str(change.path)
    if not change.existed and change.after is None:
        return None
    if change.existed and change.before is not None and change.after == hashlib.sha256(change.before).digest():
        return None
    status: Status = "A" if not change.existed else "D" if change.after is None else "M"
    added: int | None = None
    removed: int | None = None
    note: str | None = None
    old_size = len(change.before) if change.before is not None else None
    new_size: int | None = None
    if change.too_large:
        note = "too large"
    elif change.broken is not None:
        note = "unavailable"
    else:
        new = b"" if change.after is None else _current(change.path, change.after, root)
        new_size = None if new is None else len(new)
        if new is None:
            note = "unavailable"
        elif budget.expired():  # the list's time is used up: the rest is listed without counts
            pass
        else:
            try:
                counts = _count(change.before or b"", new, budget.within(DIFF_SECONDS))
            except TooLarge:
                note = TOO_BIG
            else:
                if counts is None:
                    note = "binary"
                else:
                    added, removed = counts
    return DiffItem(
        name, status, added, removed, note,
        old_size=old_size, new_size=new_size,
        old_bytes=change.before, new_path=change.path, new_digest=change.after,
    )  # fmt: skip


def _decode(old: bytes, new: bytes) -> tuple[str, str] | None:
    if b"\0" in old or b"\0" in new:
        return None
    try:
        return old.decode("utf-8"), new.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _split(text: str) -> list[str]:
    """Lines with their `\\n` (and any `\\r` before it); only `\\n` ends a line."""
    parts = text.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


class _Search(difflib.SequenceMatcher[str]):
    """The matcher of the part of two files that differs; it looks at the budget before every search, so a deadline or a
    cancel ends the comparison between two searches (each is bounded by the `WORK_LIMIT` estimate)."""

    def __init__(self, a: list[str], b: list[str], budget: Budget) -> None:
        super().__init__(None, a, b, autojunk=False)
        self._budget = budget

    def find_longest_match(self, alo: int = 0, ahi: int | None = None, blo: int = 0, bhi: int | None = None) -> difflib.Match:
        self._budget.check()
        return super().find_longest_match(alo, ahi, blo, bhi)


class _Whole(difflib.SequenceMatcher[str]):
    """Two whole files with the matching blocks already found: `get_opcodes` and `get_grouped_opcodes` take them as
    they are."""

    def __init__(self, a: list[str], b: list[str], blocks: list[difflib.Match]) -> None:
        super().__init__(None, a, b, autojunk=False)
        self._blocks = blocks

    def get_matching_blocks(self) -> list[difflib.Match]:
        return self._blocks


def _work(a: list[str], b: list[str]) -> int:
    """The steps of the first search of difflib (no junk): each line of `a` against every equal line of `b`."""
    in_a, in_b = Counter(a), Counter(b)
    return sum(n * in_b[line] for line, n in in_a.items() if line in in_b)


def _matcher(a: list[str], b: list[str], budget: Budget) -> difflib.SequenceMatcher[str]:
    """The one comparison both the counts and the diff use: the common first and last lines are taken off (the usual
    edit is small), the rest is matched exactly (`autojunk` off) if it fits the limits. Raises TooLarge, or Cancelled."""
    if len(a) > DIFF_LINES or len(b) > DIFF_LINES:
        raise TooLarge("lines")
    budget.check()
    head = 0
    limit = min(len(a), len(b))
    while head < limit and a[head] == b[head]:
        head += 1
    tail = 0
    while tail < limit - head and a[-1 - tail] == b[-1 - tail]:
        tail += 1
    inner_a, inner_b = a[head : len(a) - tail], b[head : len(b) - tail]
    blocks: list[difflib.Match] = []
    if head:
        blocks.append(difflib.Match(0, 0, head))
    if inner_a and inner_b:
        if _work(inner_a, inner_b) > WORK_LIMIT:
            raise TooLarge("work")
        blocks.extend(
            difflib.Match(m.a + head, m.b + head, m.size) for m in _Search(inner_a, inner_b, budget).get_matching_blocks() if m.size
        )
    if tail:
        blocks.append(difflib.Match(len(a) - tail, len(b) - tail, tail))
    blocks.append(difflib.Match(len(a), len(b), 0))
    return _Whole(a, b, blocks)


def _count(old: bytes, new: bytes, budget: Budget) -> tuple[int, int] | None:
    """Lines added and removed, None for binary content. Raises TooLarge (a side over a limit, or no time)."""
    if max(len(old), len(new)) > DIFF_BYTES:
        raise TooLarge("bytes")
    texts = _decode(old, new)
    if texts is None:
        return None
    added = removed = 0
    for tag, i1, i2, j1, j2 in _matcher(_split(texts[0]), _split(texts[1]), budget).get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def _range(start: int, stop: int) -> str:
    """difflib's `@@` range: `start,length`, the length left out when it is 1, the line before when it is 0."""
    beginning, length = start + 1, stop - start
    if length == 1:
        return str(beginning)
    if not length:
        beginning -= 1
    return f"{beginning},{length}"


def _hunks(a: list[str], b: list[str], path: str, matcher: difflib.SequenceMatcher[str]) -> Iterator[str]:
    """`difflib.unified_diff` over `matcher`, plus git's marker after a last line that has no line ending."""
    yield f"--- a/{path}"
    yield f"+++ b/{path}"
    for group in matcher.get_grouped_opcodes(CONTEXT):
        yield f"@@ -{_range(group[0][1], group[-1][2])} +{_range(group[0][3], group[-1][4])} @@"
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                sides = [(" ", line) for line in a[i1:i2]]
            else:
                sides = []
                if tag in ("replace", "delete"):
                    sides += [("-", line) for line in a[i1:i2]]
                if tag in ("replace", "insert"):
                    sides += [("+", line) for line in b[j1:j2]]
            for sign, line in sides:
                yield sign + line.removesuffix("\n")
                if not line.endswith("\n"):
                    yield NO_NEWLINE


def _unified(
    a: list[str], b: list[str], path: str, matcher: difflib.SequenceMatcher[str]
) -> tuple[tuple[str, ...], int]:
    produced = _hunks(a, b, path, matcher)
    lines = tuple(itertools.islice(produced, LINE_LIMIT))
    cut = sum(1 for _ in produced)
    return lines, cut


def _compare(item: DiffItem, old: bytes, new: bytes, old_mode: int, new_mode: int, budget: Budget) -> FileDiff:
    meta: list[str] = []
    if old_mode and new_mode and old_mode != new_mode:
        meta.append(_mode_note(old_mode, new_mode))
    old_link, new_link = _kind(old_mode) == SYMLINK, _kind(new_mode) == SYMLINK
    if old_link or new_link:
        for label, data, link, mode in (("old", old, old_link, old_mode), ("new", new, new_link, new_mode)):
            if link:
                meta.append(f"{label} -> {data.decode('utf-8', 'replace')}")
            elif mode:
                meta.append(f"{label}: regular file, {len(data)} bytes")
        return FileDiff(item, tuple(meta), (), 0)
    if max(len(old), len(new)) > DIFF_BYTES:
        meta.append(f"{TOO_BIG}: {len(old)} → {len(new)} bytes")
        return FileDiff(item, tuple(meta), (), 0)
    texts = _decode(old, new)
    if texts is None:
        meta.append(f"binary: {len(old)} → {len(new)} bytes")
        return FileDiff(item, tuple(meta), (), 0)
    if old == new:
        return FileDiff(item, tuple(meta), (), 0)
    a, b = _split(texts[0]), _split(texts[1])
    try:
        matcher = _matcher(a, b, budget)
    except TooLarge:
        meta.append(f"{TOO_BIG}: {len(a)} → {len(b)} lines, {len(old)} → {len(new)} bytes")
        return FileDiff(item, tuple(meta), (), 0)
    lines, cut = _unified(a, b, item.path, matcher)
    return FileDiff(item, tuple(meta), lines, cut)


async def load_file(diff: TurnDiff, item: DiffItem) -> FileDiff:
    """The diff of one file. Raises GitError when git fails. Cancelling it stops the comparison thread."""
    if item.note == "too large":
        return FileDiff(item, ("too large: no diff",), (), 0)
    if item.note == TOO_BIG:
        return FileDiff(item, (f"{TOO_BIG}: {item.old_size or 0} → {item.new_size or 0} bytes",), (), 0)
    if diff.fallback or diff.store is None:
        return await _in_thread(lambda budget: _load_from_log(item, diff.workspace, budget), DIFF_SECONDS)
    wanted = [s for s in (item.old_sha, item.new_sha) if s]
    blobs = await diff.store.read_blobs(wanted)
    if any(s not in blobs for s in wanted):  # past the store's read budget
        return FileDiff(item, ("too large: no diff",), (), 0)
    old = blobs[item.old_sha] if item.old_sha else b""
    new = blobs[item.new_sha] if item.new_sha else b""
    return await _in_thread(
        lambda budget: _compare(item, old, new, item.old_mode, item.new_mode, budget), DIFF_SECONDS
    )


def _load_from_log(item: DiffItem, workspace: Path, budget: Budget) -> FileDiff:
    if item.note == "unavailable" and item.old_bytes is None and item.status != "A":
        return FileDiff(item, ("unavailable: the log did not keep the file's content",), (), 0)
    new = b"" if item.status == "D" else None
    if item.status != "D" and item.new_path is not None:
        new = _current(item.new_path, item.new_digest, workspace)
    if new is None:
        return FileDiff(item, ("unavailable: the file changed since the turn",), (), 0)
    return _compare(item, item.old_bytes or b"", new, 0, 0, budget)


def render_lines(file_diff: FileDiff) -> list[DiffLine]:
    """Display lines: meta first, then the hunks; every text goes through `escape_display` (tabs kept) after being cut
    to `LINE_CHARS`, the note about the cut ends the list."""
    out = [DiffLine(_show(text), "meta") for text in file_diff.meta]
    for number, line in enumerate(file_diff.lines):
        if number < 2 or line == NO_NEWLINE:  # the `---`/`+++` header: a content line can start with the same characters
            kind: LineKind = "meta"
        elif line.startswith("@@"):
            kind = "hunk"
        elif line.startswith("+"):
            kind = "add"
        elif line.startswith("-"):
            kind = "del"
        else:
            kind = "ctx"
        out.append(DiffLine(_show(line), kind))
    if file_diff.cut:
        out.append(DiffLine(f"[{file_diff.cut} more lines not shown]", "meta"))
    return out


def _show(text: str) -> str:
    shown = "\t".join(escape_display(part) for part in text[:LINE_CHARS].split("\t"))
    return shown if len(text) <= LINE_CHARS else f"{shown} [{len(text) - LINE_CHARS} more characters]"
