"""Project instructions: every AGENTS.md from the workspace up to the git root, fitted to a budget."""

from __future__ import annotations

import codecs
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.agent.compaction import CHARS_PER_TOKEN

AGENTS_MD = "AGENTS.md"
AGENTS_MD_LIMIT = 32 * 1024  # bytes read from one file at most
MIN_KEPT = 200  # a cut file keeps at least this many characters, else it is omitted
PRECEDENCE = "Instruction files are listed from the outermost folder to this one; where they conflict, the later one wins."

State = Literal["whole", "cut", "omitted"]


def _tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


@dataclass(frozen=True)
class InstructionFile:
    label: str  # path relative to the git root, POSIX separators
    tokens: int  # estimated tokens of this file's text in the prompt (0 when omitted)
    original: int  # estimated tokens of the whole file
    state: State
    capped: bool = False  # cut by the 32 KB read cap alone (it fits the budget), not by the budget


@dataclass(frozen=True)
class Instructions:
    text: str  # the instructions part of the system prompt; "" when there is none
    files: tuple[InstructionFile, ...]  # outermost first; only files that exist with content
    budget: int | None  # tokens; None = no budget (only the 32 KB cap)

    @property
    def tokens(self) -> int:
        return _tokens(len(self.text))

    @property
    def key(self) -> tuple[tuple[str, str], ...]:
        """What a notice reports: the files and their states, not their sizes."""
        return tuple((f.label, f.state) for f in self.files)

    def notice(self) -> tuple[Literal["info", "warn"], str] | None:
        if not self.files:
            return None
        entries = []
        for f in self.files:
            if f.state == "omitted":
                entries.append(f"{f.label} (omitted, ~{f.original})")
            elif f.capped:
                entries.append(f"{f.label} (~{f.tokens}, first 32 KB of ~{f.original})")
            elif f.state == "cut":
                entries.append(f"{f.label} (~{f.tokens}, cut from ~{f.original})")
            else:
                entries.append(f"{f.label} (~{f.tokens})")
        tail = f"~{self.tokens} tokens" if self.budget is None else f"~{self.tokens} of ~{self.budget} tokens"
        level = "warn" if any(f.state != "whole" for f in self.files) else "info"
        return level, f"Instructions: {', '.join(entries)} · {tail}"


def levels(workspace: Path) -> list[tuple[str, Path]]:
    """`(label, path to AGENTS.md)` from the git root down to the workspace; just the workspace without a root."""
    folders = [workspace, *workspace.parents]
    root: Path | None = None
    for folder in folders:
        try:
            os.lstat(folder / ".git")
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:  # a folder that cannot be examined: stop, never guess a root further up
            break
        root = folder
        break
    if root is None:
        return [(AGENTS_MD, workspace / AGENTS_MD)]
    chain = folders[: folders.index(root) + 1][::-1]
    return [((folder / AGENTS_MD).relative_to(root).as_posix(), folder / AGENTS_MD) for folder in chain]


def fingerprint(path: Path) -> tuple[int, int] | None:
    """`(size, mtime_ns)` of a regular file; None when it is missing or not regular. Never raises."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_size, info.st_mtime_ns) if stat.S_ISREG(info.st_mode) else None


def _read_head(path: Path, limit: int) -> tuple[bytes, int] | None:
    """Up to `limit` bytes of a regular file and its size; None for anything else (FIFO, device, directory)."""
    try:
        # O_NONBLOCK so opening a FIFO returns at once; the fstat then rejects it (no TOCTOU).
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except OSError:  # missing, unreadable
        return None
    try:
        file = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode):
            return None
        return file.read(limit), info.st_size


@dataclass(frozen=True)
class _Read:
    label: str
    text: str  # decoded, without any truncation note
    capped: bool  # longer than the read cap
    original: int  # estimated tokens of the whole file


def _read(label: str, path: Path) -> _Read | None:
    try:
        head = _read_head(path, AGENTS_MD_LIMIT + 1)
    except OSError:
        return None
    if head is None:
        return None
    raw, size = head
    capped = len(raw) > AGENTS_MD_LIMIT
    if capped:
        raw = raw[:AGENTS_MD_LIMIT]
    # Not final when cut: a character split by the limit is dropped, while invalid bytes before
    # it become U+FFFD instead of vanishing.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(raw, final=not capped)
    if not text.strip():
        return None
    # In characters throughout: a capped file is scaled by the characters-per-byte of what was read.
    chars = len(text) if not capped else round(size * len(text) / len(raw))
    return _Read(label, text, capped, _tokens(chars))


def _heading(label: str) -> str:
    return f"## {label}\n"


def _omitted_line(label: str) -> str:
    return f"[{label} omitted: no room in the instruction budget]"


def load_instructions(workspace: Path, budget: int | None) -> Instructions:
    """Read every level and fit the files into `budget` tokens; the workspace's file gets room first."""
    found = [r for label, path in levels(workspace) if (r := _read(label, path)) is not None]
    sections: dict[str, str] = {}  # label -> text for whole / cut / omitted-line entries
    states: dict[str, State] = {}
    capped_only: set[str] = set()  # cut by the read cap alone, not by the budget
    included: dict[str, str] = {}  # label -> text in the prompt, for whole and cut files

    remaining = None if budget is None else max(budget, 0) * CHARS_PER_TOKEN
    precedence = len(PRECEDENCE) + 2
    for index in range(len(found) - 1, -1, -1):
        r = found[index]
        # The precedence line is paid by the second section that gets included.
        extra = precedence if included else 0
        capped_note = f"\n[{r.label} truncated at 32 KB]"
        text = r.text + (capped_note if r.capped else "")
        heading = _heading(r.label)
        if remaining is None or extra + len(heading) + len(text) + 2 <= remaining:
            states[r.label] = "cut" if r.capped else "whole"
            if r.capped:
                capped_only.add(r.label)
            included[r.label] = text
            sections[r.label] = heading + text
            if remaining is not None:
                remaining -= extra + len(heading) + len(text) + 2
            continue
        note = f"\n[{r.label} cut to fit the instruction budget]"
        reserve = sum(len(_omitted_line(o.label)) + 2 for o in found[:index])
        base = remaining - extra - len(heading) - len(note) - 2
        keep = base - reserve
        if keep < MIN_KEPT:  # the workspace's content has priority over the notes
            keep = base
        if keep >= MIN_KEPT:
            text = r.text[:keep] + note
            states[r.label] = "cut"
            included[r.label] = text
            sections[r.label] = heading + text
            remaining -= extra + len(heading) + len(text) + 2
            continue
        states[r.label] = "omitted"
        line = _omitted_line(r.label)
        if len(line) + 2 <= remaining:
            sections[r.label] = line
            remaining -= len(line) + 2

    parts: list[str] = []
    if sum(1 for s in states.values() if s != "omitted") > 1:
        parts.append(PRECEDENCE)
    parts.extend(sections[r.label] for r in found if r.label in sections)
    text = "\n\n".join(parts)
    files = tuple(
        InstructionFile(
            r.label,
            _tokens(len(included[r.label])) if r.label in included else 0,
            r.original,
            states[r.label],
            r.label in capped_only,
        )
        for r in found
    )
    return Instructions(text, files, budget)
