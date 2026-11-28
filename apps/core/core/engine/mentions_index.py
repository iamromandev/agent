"""The files `@` completion offers: workspace-relative paths from `git ls-files` or a bounded walk, bucketed by folder."""

from __future__ import annotations

import os
import re
import signal
import stat
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from core.engine.checkpoints.git import find_git
from core.engine.guardrails.setup import Guardrails

INDEX_LIMIT = 50_000
GIT_TIMEOUT = 10.0
WALK_SECONDS = 3.0
WALK_FOLDERS = 20_000
GITIGNORE_LIMIT = 1 << 20
SKIP_DIRS = frozenset({".git", "node_modules", ".venv", "__pycache__"})


_TOKEN = re.compile(r'(?:(?<=\s)|^)@("[^"\n]*|[^\s"]*)$')


def current_token(before_cursor: str) -> str | None:
    """The `@token` being typed, without its `@` (a leading `"` is kept), from the text of the cursor's line up to the
    cursor; None when the cursor is not at the end of a mention."""
    found = _TOKEN.search(before_cursor)
    return found.group(1) if found else None


@dataclass(frozen=True)
class Suggestion:
    path: str  # workspace-relative, `/` separated; folders end with `/`
    folder: bool


def git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    return env


def _glob(pat: str) -> str:
    """A gitignore glob (no leading `!` or `/`, no trailing `/`) as regex source: `*` and `?` stay in a segment,
    `**` spans segments, `[...]` is a class, `\\x` is a literal."""
    out: list[str] = []
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat.startswith("**", i) and (i == 0 or pat[i - 1] == "/"):
                if pat.startswith("/", i + 2):
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                if i + 2 == len(pat):
                    out.append(".*")
                    break
            out.append("[^/]*")
            while i + 1 < len(pat) and pat[i + 1] == "*":
                i += 1
        elif c == "?":
            out.append("[^/]")
        elif c == "[" and (j := pat.find("]", i + 2)) != -1:
            body = pat[i + 1 : j]
            body = "^" + body[1:] if body[:1] in ("!", "^") else body
            out.append("[" + body.replace("\\", "\\\\") + "]")
            i = j
        elif c == "\\" and i + 1 < len(pat):
            i += 1
            out.append(re.escape(pat[i]))
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class _Rule:
    base: str  # folder of the .gitignore, "" or ending in `/`
    regex: re.Pattern[str]
    negate: bool
    dir_only: bool


def parse_gitignore(text: str, base: str) -> list[_Rule]:
    """The common gitignore syntax: `name`, `dir/`, `*.ext`, `/anchored`, `a/b` (anchored), `**`, `!negation`, `#`
    comments, `\\#` and `\\!`. Not supported: character-class edge cases, case folding, `core.excludesFile`,
    `.git/info/exclude`."""
    rules: list[_Rule] = []
    for line in text.splitlines():
        line = line.rstrip(" ")
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        line = line[1:] if negate else line
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        anchored = "/" in line
        line = line.lstrip("/")
        if not line:
            continue
        try:
            regex = re.compile(("^" if anchored else "^(?:.*/)?") + _glob(line) + "$")
        except re.error:
            continue
        rules.append(_Rule(base, regex, negate, dir_only))
    return rules


def _ignored(rules: list[_Rule], rel: str, is_dir: bool) -> bool:
    ignored = False
    for rule in rules:
        if rule.dir_only and not is_dir:
            continue
        if rel.startswith(rule.base) and rule.regex.match(rel[len(rule.base) :]):
            ignored = not rule.negate
    return ignored


class FileIndex:
    def __init__(self, workspace: Path, guard: Guardrails, limit: int = INDEX_LIMIT) -> None:
        self.workspace = workspace
        self.guard = guard
        self.limit = limit
        self._buckets: dict[str, list[tuple[str, Suggestion]]] = {}
        self._ready = False
        self._cancel = threading.Event()

    @property
    def ready(self) -> bool:
        return self._ready

    def cancel(self) -> None:
        """Ask a running (or later) `build` to stop: it returns without replacing the index. Final, for quitting."""
        self._cancel.set()

    def build(self) -> None:
        """Blocking: run it in a thread. Replaces the index at the end, so `suggest` never sees a half-built one."""
        files = self._git_files()
        if files is None and not self._cancel.is_set():
            files = self._walk()
        if files is None or self._cancel.is_set():
            return
        folders: dict[str, None] = {}
        kept: list[str] = []
        cache: dict[Path, bool] = {}
        root = self._real_root()
        for n, rel in enumerate(files):
            if n % 256 == 0 and self._cancel.is_set():
                return
            if not self._listable(rel, root):
                continue
            try:
                if self.guard.paths.is_protected(self.workspace / rel, cache):
                    continue
            except Exception:
                continue
            kept.append(rel)
            parts = rel.split("/")
            for i in range(1, len(parts)):
                folders["/".join(parts[:i]) + "/"] = None
            if len(kept) >= self.limit:
                break
        buckets: dict[str, list[tuple[str, Suggestion]]] = {}
        for rel in kept:
            buckets.setdefault(rel.rpartition("/")[0] + "/" if "/" in rel else "", []).append(
                (rel.rpartition("/")[2], Suggestion(rel, False))
            )
        for rel in folders:
            parent, _, name = rel[:-1].rpartition("/")
            buckets.setdefault(parent + "/" if parent else "", []).append((name, Suggestion(rel, True)))
        for entries in buckets.values():
            entries.sort(key=lambda e: (not e[1].folder, e[0].lower(), e[0]))
        self._buckets = buckets
        self._ready = True

    def _real_root(self) -> Path:
        try:
            return self.workspace.resolve()
        except (OSError, RuntimeError):
            return self.workspace

    def _listable(self, rel: str, root: Path) -> bool:
        """A regular file, or a symlink to one inside the workspace (git and the walk agree on this); no newline or
        quote in the name, which the prompt syntax cannot carry."""
        if not rel or rel.endswith("/") or "\n" in rel or '"' in rel:
            return False
        path = self.workspace / rel
        try:
            mode = os.lstat(path).st_mode
            if stat.S_ISREG(mode):
                return True
            if not stat.S_ISLNK(mode):
                return False  # a folder (nested repo, submodule), a socket...
            real = path.resolve(strict=True)
            return real.is_file() and real.is_relative_to(root)
        except (OSError, RuntimeError):
            return False

    def suggest(self, token: str, limit: int = 50) -> list[Suggestion]:
        token = token.removeprefix('"')
        directory, _, segment = token.rpartition("/")
        entries = self._buckets.get(directory + "/" if "/" in token else "", [])
        needle = segment.lower()
        starts = [s for name, s in entries if name.lower().startswith(needle)]
        contains = [s for name, s in entries if needle and needle in name.lower() and not name.lower().startswith(needle)]
        return (starts + contains)[:limit]

    def _git_files(self) -> list[str] | None:
        env = git_env()
        git = find_git(env, self.workspace)
        if git is None:
            return None
        try:
            proc = subprocess.Popen(
                [git, "-c", "core.fsmonitor=false", "ls-files", "-co", "--exclude-standard", "-z"],
                cwd=self.workspace,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        deadline = time.monotonic() + GIT_TIMEOUT
        while True:
            try:
                out, _ = proc.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if self._cancel.is_set() or time.monotonic() > deadline:
                    with suppress(ProcessLookupError, PermissionError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.communicate()
                    return None
        if proc.returncode != 0:
            return None
        names = out.decode("utf-8", "replace").split("\0")
        return sorted({n for n in names if n and not n.startswith("../")})

    def _walk(self) -> list[str]:
        """Files below the workspace, honouring `.gitignore` files; bounded by files, folders and time."""
        found: list[str] = []
        deadline = time.monotonic() + WALK_SECONDS
        rules_of: dict[str, list[_Rule]] = {}
        for folders, (base, dirs, names) in enumerate(os.walk(self.workspace), 1):
            if self._cancel.is_set() or folders > WALK_FOLDERS or time.monotonic() > deadline:
                break
            rel_base = os.path.relpath(base, self.workspace)
            prefix = "" if rel_base == "." else Path(rel_base).as_posix() + "/"
            rules = list(rules_of.pop(prefix, []))
            rules += self._read_gitignore(base, prefix)
            kept_dirs = sorted(
                d for d in dirs if d not in SKIP_DIRS and not _ignored(rules, prefix + d, True)
            )
            for d in kept_dirs:
                rules_of[prefix + d + "/"] = rules
            dirs[:] = kept_dirs
            for name in sorted(names):
                if not _ignored(rules, prefix + name, False):
                    found.append(prefix + name)
            if len(found) >= self.limit:
                break
        return found

    @staticmethod
    def _read_gitignore(base: str, prefix: str) -> list[_Rule]:
        path = os.path.join(base, ".gitignore")
        try:
            if not stat.S_ISREG(os.lstat(path).st_mode):
                return []
            with open(path, "rb") as handle:
                data = handle.read(GITIGNORE_LIMIT)
        except OSError:
            return []
        return parse_gitignore(data.decode("utf-8", "replace"), prefix)
