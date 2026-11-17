"""The shadow repository of a run: one bare git repo in the agent's data folder, snapshots of the workspace into it.

The user's own repository is never read except for its `.git/info/exclude`: git runs with its own `GIT_DIR`, its
own index file and no user config, hooks, attributes or filters (see `git.py`)."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.engine.checkpoints.git import GIT_TIMEOUT, GitError, GitRunner, GitTimeout, find_git
from src.engine.guardrails.setup import Guardrails
from src.engine.text import escape

FILE_CAP = 10 * 1024 * 1024
ATTRIBUTES = "* -filter -text -eol -ident -working-tree-encoding -diff -merge\n"
AGENT_FILES = frozenset({".agent/settings.json", ".agent/mcp.json"})
AGENT_FOLDERS = (".agent/commands/", ".agent/skills/")
OFF_SUFFIX = "/undo restores write/edit changes only"
MAX_NESTING = 8  # nested repositories inside nested repositories


def off_warning(reason: str) -> str:
    return f"checkpoints off: {reason}; {OFF_SUFFIX}"


@dataclass(frozen=True)
class Snapshot:
    commit: str
    skipped_large: tuple[str, ...]  # workspace-relative posix paths over FILE_CAP


def folder_name(workspace: Path) -> str:
    """`<sha256(workspace)[:16]>-<pid>`: private to this run."""
    return f"{hashlib.sha256(str(workspace).encode()).hexdigest()[:16]}-{os.getpid()}"


def _is_agent_file(rel: str) -> bool:
    lowered = rel.lower()  # the filesystem may be case-insensitive (macOS)
    return lowered in AGENT_FILES or lowered.startswith(AGENT_FOLDERS)


def make_private_dirs(path: Path) -> None:
    """`mkdir -p` where every folder this call creates is 0700 (`Path.mkdir(parents=True)` gives the parents the
    default mode). Folders that exist keep their mode."""
    missing: list[Path] = []
    current = path
    while not os.path.lexists(current) and current != current.parent:
        missing.append(current)
        current = current.parent
    for folder in reversed(missing):
        try:
            folder.mkdir(mode=0o700)
        except FileExistsError:
            continue
        folder.chmod(0o700)
    if not path.is_dir():
        raise NotADirectoryError(str(path))


class CheckpointStore:
    def __init__(
        self,
        workspace: Path,
        root: Path,
        guard: Guardrails,
        env: Mapping[str, str],
        *,
        timeout: float = GIT_TIMEOUT,
        git: str | None = None,
    ) -> None:
        """`git`: the absolute program, resolved once by `prepare_checkpoints`; looked up on `env`'s PATH (absolute
        entries only, never inside the workspace) at the first snapshot when not given."""
        self._git_program = git
        self._workspace = workspace
        self._root = root
        self._guard = guard
        self._env = dict(env)
        self._timeout = timeout
        self.folder = root / folder_name(workspace)
        self._git: GitRunner | None = None
        self._off: str | None = None
        self._warning: str | None = None
        self._closed = False
        self._tree: str | None = None
        self._commit: str | None = None
        self._verdicts: dict[str, tuple[int, ...]] = {}  # kept files, by stat: not asked to the guard again

    @property
    def enabled(self) -> bool:
        return self._off is None and not self._closed

    @property
    def off_reason(self) -> str | None:
        return self._off

    def take_warning(self) -> str | None:
        """The one `checkpoints off: ...` text, once."""
        warning, self._warning = self._warning, None
        return warning

    def _turn_off(self, reason: str) -> None:
        if self._off is None and not self._closed:
            self._off = reason
            self._warning = off_warning(reason)

    def _exclude_file(self) -> Path:
        """The workspace's own `.git/info/exclude` when it is a regular file that really lies inside the workspace;
        else `os.devnull`, so no other ignore file (`~/.config/git/ignore`, ...) is ever read."""
        try:
            real = (self._workspace / ".git" / "info" / "exclude").resolve(strict=True)
            if stat.S_ISREG(os.stat(real).st_mode) and self._workspace.resolve() in real.parents:
                return self._workspace / ".git" / "info" / "exclude"
        except (OSError, RuntimeError):
            pass
        return Path(os.devnull)

    async def _runner(self) -> GitRunner:
        if self._git is not None:
            return self._git
        program = self._git_program or find_git(self._env, self._workspace)
        if program is None:
            raise GitError("git not found")
        folder = self.folder
        make_private_dirs(self._root)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)  # left by an earlier run with this pid
        folder.mkdir(mode=0o700)
        hooks = folder / "hooks-empty"
        hooks.mkdir(mode=0o700)
        home = folder / "home"
        home.mkdir(mode=0o700)
        runner = GitRunner(program, folder, self._workspace, self._env, hooks, home, exclude=self._exclude_file())
        await runner.run("init", "--bare", "-q", timeout=self._timeout, tree=False)
        if self._closed:  # close() came while git ran: it may have made the folder again after close() removed it
            shutil.rmtree(folder, ignore_errors=True)
            raise GitError("checkpoints closed")
        folder.chmod(0o700)
        (folder / "info").mkdir(exist_ok=True)
        (folder / "info" / "attributes").write_text(ATTRIBUTES)
        self._git = runner
        return runner

    async def snapshot(self) -> Snapshot | None:
        """Record the workspace as a commit. None, with the store turned off, when git fails or runs too long."""
        if not self.enabled:
            return None
        try:
            async with asyncio.timeout(self._timeout):
                return await self._snapshot()
        except TimeoutError:
            self._turn_off(f"snapshot took longer than {self._timeout:g} s")
        except GitTimeout:
            self._turn_off(f"snapshot took longer than {self._timeout:g} s")
        except (GitError, OSError) as exc:
            self._turn_off(f"snapshot failed: {escape(str(exc))}")
        return None

    async def _snapshot(self) -> Snapshot:
        git = await self._runner()
        keep, drop, large = await self._classify_workspace(git)
        try:
            await self._update_index(git, keep, drop)
        except GitTimeout:
            raise
        except GitError:  # a file may have changed under us: look again, once
            keep, drop, large = await self._classify_workspace(git)
            await self._update_index(git, keep, drop)
        tree = (await git.run("write-tree", timeout=self._timeout)).decode().strip()
        if tree != self._tree or self._commit is None:
            args = ["commit-tree", tree, "-m", "checkpoint"]
            if self._commit is not None:
                args += ["-p", self._commit]
            self._commit = (await git.run(*args, timeout=self._timeout)).decode().strip()
            self._tree = tree
        return Snapshot(self._commit, tuple(sorted(large)))

    async def _update_index(self, git: GitRunner, keep: list[str], drop: list[str]) -> None:
        if drop:
            await git.run("update-index", "--force-remove", "-z", "--stdin", stdin=_nul(drop), timeout=self._timeout)
        if keep:  # --remove: a file that vanished since it was classified leaves the index instead of failing
            await git.run(
                "update-index", "--add", "--replace", "--remove", "-z", "--stdin",
                stdin=_nul(keep), timeout=self._timeout,
            )  # fmt: skip

    async def _classify_workspace(self, git: GitRunner) -> tuple[list[str], list[str], list[str]]:
        """Ask git for the candidate paths, then split them: paths to add, paths to remove from the index, files over
        the cap."""
        listing, ignored = await asyncio.gather(
            git.run("ls-files", "-z", "--cached", "--others", "--exclude-standard", timeout=self._timeout),
            git.run("ls-files", "-z", "--cached", "--ignored", "--exclude-standard", timeout=self._timeout),
        )
        rels: list[str] = []
        for raw in listing.split(b"\0"):
            if raw:
                rels.extend(await self._expand(git, os.fsdecode(raw), 0))
        gone = {os.fsdecode(raw) for raw in ignored.split(b"\0") if raw}  # tracked once, ignored now
        return await asyncio.to_thread(self._classify, rels, gone)

    async def _expand(self, git: GitRunner, rel: str, depth: int) -> list[str]:
        """The files of a nested repository git lists as a folder (`sub/`), found by git itself with that folder as
        work tree (so its own ignore files apply) and without its `.git`; else `[rel]`."""
        if not rel.endswith("/"):
            return [rel]
        if depth >= MAX_NESTING:
            return []
        base = self._workspace / rel
        options: list[str] = []
        exclude = self._nested_exclude(base)
        options += ["-c", f"core.excludesFile={exclude}"]
        listing = await git.run(
            "ls-files", "-z", "--others", "--exclude-standard",
            timeout=self._timeout, work=base, options=options,
        )  # fmt: skip
        found: list[str] = []
        for raw in listing.split(b"\0"):
            if raw:
                found.extend(await self._expand(git, rel + os.fsdecode(raw), depth + 1))
        return found

    @staticmethod
    def _nested_exclude(base: Path) -> Path:
        """`<base>/.git/info/exclude` when `.git` is a real folder and that is a regular file; else `os.devnull`."""
        try:
            if stat.S_ISDIR(os.lstat(base / ".git").st_mode) and stat.S_ISREG(os.lstat(base / ".git" / "info" / "exclude").st_mode):
                return base / ".git" / "info" / "exclude"
        except OSError:
            pass
        return Path(os.devnull)

    def _classify(self, rels: list[str], gone: set[str]) -> tuple[list[str], list[str], list[str]]:
        """Split the candidate paths into paths to add, paths to remove from the index (`gone`: ignored now), and
        files over the cap."""
        keep: list[str] = []
        drop: list[str] = []
        large: list[str] = []
        folders: dict[Path, bool] = {}
        parents: dict[str, bool] = {}
        seen: set[str] = set()
        for rel in rels:
            if rel in seen:
                continue
            seen.add(rel)
            if rel in gone:
                drop.append(rel)
                self._verdicts.pop(rel, None)
                continue
            self._sort(rel, keep, drop, large, folders, parents)
        return keep, drop, large

    def _parents_are_folders(self, rel: str, cache: dict[str, bool]) -> bool:
        """Is every folder above `rel` a real folder (not a symlink)? git refuses a path "beyond a symbolic link"."""
        parent = rel.rpartition("/")[0]
        if not parent:
            return True
        known = cache.get(parent)
        if known is None:
            try:
                is_dir = stat.S_ISDIR(os.lstat(self._workspace / parent).st_mode)
            except OSError:
                is_dir = False
            known = cache[parent] = is_dir and self._parents_are_folders(parent, cache)
        return known

    def _sort(
        self,
        rel: str,
        keep: list[str],
        drop: list[str],
        large: list[str],
        folders: dict[Path, bool],
        parents: dict[str, bool],
    ) -> None:
        if any(part.lower() == ".git" for part in rel.split("/")) or not self._parents_are_folders(rel, parents):
            drop.append(rel)
            self._verdicts.pop(rel, None)
            return
        path = self._workspace / rel
        try:
            info = os.lstat(path)
        except OSError:
            drop.append(rel)  # gone
            self._verdicts.pop(rel, None)
            return
        regular = stat.S_ISREG(info.st_mode)
        if not (regular or stat.S_ISLNK(info.st_mode)) or _is_agent_file(rel):
            drop.append(rel)
            self._verdicts.pop(rel, None)
            return
        signature = (info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev)
        if self._verdicts.get(rel) == signature:  # same file as the last time it was allowed
            keep.append(rel)
        elif self._guard.paths.is_protected(path, folders):
            drop.append(rel)
            self._verdicts.pop(rel, None)
        elif regular and info.st_size > FILE_CAP:
            drop.append(rel)
            large.append(rel)
            self._verdicts.pop(rel, None)
        else:
            keep.append(rel)
            self._verdicts[rel] = signature

    def close(self) -> None:
        """Synchronous: kill any git still running and delete the folder (errors ignored)."""
        self._closed = True
        if self._git is not None:
            self._git.kill_all()
        with contextlib.suppress(Exception):
            shutil.rmtree(self.folder, ignore_errors=True)


def _nul(paths: list[str]) -> bytes:
    return b"".join(os.fsencode(p) + b"\0" for p in paths)
