"""A minimal async git invoker for the shadow repository: its own env, its own options, its own process group."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from src.engine.process import kill_group

T = TypeVar("T")

GIT_TIMEOUT = 5.0
MIN_GIT = (2, 20)
STDERR_LIMIT = 500
IDENTITY = {
    "GIT_AUTHOR_NAME": "agent",
    "GIT_AUTHOR_EMAIL": "agent@localhost",
    "GIT_COMMITTER_NAME": "agent",
    "GIT_COMMITTER_EMAIL": "agent@localhost",
    "GIT_AUTHOR_DATE": "1970-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE": "1970-01-01T00:00:00Z",
}


class GitError(Exception):
    """git exited non-zero; the message is its trimmed stderr (callers escape it before showing it)."""


class GitTimeout(GitError):
    """git ran past its deadline and its process group was killed."""


def git_env(base: Mapping[str, str], git_dir: Path, index: Path, home: Path) -> dict[str, str]:
    """`base` without any `GIT_*` variable, pinned to the shadow repository, ignoring every user config file.

    `HOME` and `XDG_CONFIG_HOME` point at `home` (a private, empty folder): git older than 2.32 does not know
    `GIT_CONFIG_GLOBAL`, and would read `~/.gitconfig` and `~/.config/git/*` from the real ones."""
    env = {k: v for k, v in base.items() if not k.upper().startswith("GIT_")}
    env.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home),
        GIT_DIR=str(git_dir),
        GIT_INDEX_FILE=str(index),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_OPTIONAL_LOCKS="0",
        GIT_ATTR_NOSYSTEM="1",
        **IDENTITY,
    )
    return env


def find_git(env: Mapping[str, str], workspace: Path) -> str | None:
    """The absolute path of the `git` on `env`'s PATH, looking only at absolute entries (an empty or relative entry
    means the current folder, where a workspace could plant a `git`). None when there is none, or when it lies inside
    `workspace` (resolved)."""
    entries = [e for e in env.get("PATH", "").split(os.pathsep) if e and os.path.isabs(e)]
    found = shutil.which("git", path=os.pathsep.join(entries)) if entries else None
    if found is None:
        return None
    absolute = Path(found)
    try:
        real = absolute.resolve()
        root = workspace.resolve()
    except (OSError, RuntimeError):
        return None
    if real == root or root in real.parents or absolute == root or root in absolute.parents:
        return None
    return str(absolute)


def git_version(git: str, env: Mapping[str, str]) -> tuple[int, int] | None:
    """`(major, minor)` of the program `git`; None when it cannot run or the output is not understood."""
    try:
        done = subprocess.run([git, "--version"], env=dict(env), capture_output=True, timeout=2, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(rb"git version (\d+)\.(\d+)", done.stdout)
    return (int(match.group(1)), int(match.group(2))) if match else None


class GitRunner:
    def __init__(
        self,
        git: str,
        git_dir: Path,
        workspace: Path,
        env: Mapping[str, str],
        hooks_dir: Path,
        home: Path,
        *,
        exclude: Path,
    ) -> None:
        """`git` is the absolute program; `exclude` is the one ignore file git may read besides the workspace's
        `.gitignore` files (the workspace's own `.git/info/exclude`, or `os.devnull`)."""
        self._git = git
        self._workspace = workspace
        self._env = git_env(env, git_dir, git_dir / "index", home)
        self._options = [
            "-c", f"core.hooksPath={hooks_dir}",
            "-c", "core.fsmonitor=false",
            "-c", "core.untrackedCache=false",
            "-c", "core.autocrlf=false",
            "-c", "core.symlinks=true",
            "-c", "core.quotePath=false",
            "-c", "core.createObject=rename",  # `link` + unlink is 2.5 times slower for thousands of objects on macOS
            "-c", "commit.gpgSign=false",
            "-c", "gc.auto=0",
            "-c", f"core.attributesFile={os.devnull}",
            "-c", f"core.excludesFile={exclude}",
        ]  # fmt: skip
        self._procs: set[asyncio.subprocess.Process] = set()

    async def _spawn(
        self, args: Sequence[str], *, piped: bool, tree: bool, work: Path | None, options: Sequence[str]
    ) -> asyncio.subprocess.Process:
        folder = work or self._workspace
        env = {**self._env, "GIT_WORK_TREE": str(folder)} if tree else dict(self._env)
        return await asyncio.create_subprocess_exec(
            self._git,
            *self._options,
            *options,
            *args,
            stdin=asyncio.subprocess.PIPE if piped else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=folder,
            start_new_session=True,
        )

    async def run(
        self,
        *args: str,
        stdin: bytes | None = None,
        timeout: float = GIT_TIMEOUT,
        tree: bool = True,
        work: Path | None = None,
        options: Sequence[str] = (),
    ) -> bytes:
        """stdout of `git <args>`, with `work` (default: the workspace) as work tree and current folder. Raises
        GitError on a non-zero exit and GitTimeout (group killed) past `timeout`."""
        proc = await self._spawn(args, piped=stdin is not None, tree=tree, work=work, options=options)
        self._procs.add(proc)
        try:
            try:
                out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
            except TimeoutError:
                await kill_group(proc)
                raise GitTimeout(f"git {args[0] if args else ''} took longer than {timeout:g} s") from None
            except BaseException:
                await asyncio.shield(kill_group(proc))
                raise
        finally:
            self._procs.discard(proc)
        if proc.returncode != 0:
            text = err.decode("utf-8", "replace").strip()[:STDERR_LIMIT]
            raise GitError(text or f"git exited with {proc.returncode}")
        return out

    async def stream(
        self,
        *args: str,
        stdin: bytes,
        consume: Callable[[asyncio.StreamReader], Awaitable[T]],
        timeout: float = GIT_TIMEOUT,
        tree: bool = True,
    ) -> T:
        """Like `run`, but stdout goes through `consume` as it arrives instead of being collected, so the caller keeps
        only what it needs of a large output. Same errors; the group is killed when `consume` raises."""
        proc = await self._spawn(args, piped=True, tree=tree, work=None, options=())
        self._procs.add(proc)
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        stdin_pipe, stdout_pipe, stderr_pipe = proc.stdin, proc.stdout, proc.stderr

        async def feed() -> None:
            with contextlib.suppress(ConnectionError):  # git quit early: its exit code says why
                stdin_pipe.write(stdin)
                await stdin_pipe.drain()
                stdin_pipe.close()

        async def go() -> tuple[T, bytes]:
            feeding = asyncio.create_task(feed())
            reading = asyncio.create_task(stderr_pipe.read())
            try:
                result = await consume(stdout_pipe)
                await feeding
                err = await reading
                await proc.wait()
                return result, err
            finally:
                feeding.cancel()
                reading.cancel()

        try:
            try:
                result, err = await asyncio.wait_for(go(), timeout)
            except TimeoutError:
                await kill_group(proc)
                raise GitTimeout(f"git {args[0] if args else ''} took longer than {timeout:g} s") from None
            except BaseException:
                await asyncio.shield(kill_group(proc))
                raise
        finally:
            self._procs.discard(proc)
        if proc.returncode != 0:
            text = err.decode("utf-8", "replace").strip()[:STDERR_LIMIT]
            raise GitError(text or f"git exited with {proc.returncode}")
        return result

    def kill_all(self) -> None:
        """Synchronous: SIGKILL every git group still running (the quit path)."""
        for proc in list(self._procs):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
