"""Launch: decide whether checkpoints can work, build the store, remove folders that earlier runs left behind."""

from __future__ import annotations

import os
import re
import shutil
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from core.engine.checkpoints.git import MIN_GIT, find_git, git_version
from core.engine.checkpoints.store import CheckpointStore, make_private_dirs, off_warning
from core.engine.guardrails.paths import login_home
from core.engine.guardrails.setup import Guardrails
from core.engine.mcp.trust import trust_store_path

STALE_AFTER = 24 * 3600
CHECKPOINTS = "checkpoints"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


FOLDER_NAME = re.compile(r"[0-9a-f]{16}-[0-9]+", re.ASCII)


def prune_stale(root: Path, *, now: float | None = None) -> None:
    """Delete `<root>/<16 hex>-<pid>` folders (what `folder_name` makes: real folders, never symlinks) whose process is
    gone and that are over a day old. Anything else is left alone. Never raises."""
    try:
        now = time.time() if now is None else now
        entries = list(root.iterdir())
    except (OSError, ValueError):
        return
    for entry in entries:
        try:
            if not FOLDER_NAME.fullmatch(entry.name):
                continue
            info = entry.lstat()
            if not stat.S_ISDIR(info.st_mode) or now - info.st_mtime <= STALE_AFTER:
                continue
            if _alive(int(entry.name.rpartition("-")[2])):
                continue
        except (OSError, ValueError, OverflowError):
            continue
        shutil.rmtree(entry, ignore_errors=True)


def _inside(path: Path, other: Path) -> bool:
    return path == other or other in path.parents


def prepare_checkpoints(
    workspace: Path,
    guard: Guardrails,
    env: Mapping[str, str],
    *,
    enabled: bool,
    find: Callable[[Mapping[str, str], Path], str | None] = find_git,
    version: Callable[[str, Mapping[str, str]], tuple[int, int] | None] = git_version,
    data_env: Mapping[str, str] = os.environ,
) -> tuple[CheckpointStore | None, list[str]]:
    """`(store, [])`, or `(None, [warning])` when git is missing or old, the workspace is the home folder or holds or
    lies in the agent's data folder, or the checkpoint folder cannot be written. `(None, [])` when not `enabled`."""
    if not enabled:
        return None, []
    workspace = workspace.resolve()
    git = find(env, workspace)
    found = version(git, env) if git is not None else None
    if git is None or found is None:
        return None, [off_warning("git not found")]
    if found < MIN_GIT:
        return None, [off_warning(f"git {found[0]}.{found[1]} is older than {MIN_GIT[0]}.{MIN_GIT[1]}")]
    root = trust_store_path(data_env, CHECKPOINTS)
    agent_data = root.parent.resolve()
    homes = {Path.home().resolve(), *([login.resolve()] if (login := login_home()) else [])}
    if workspace in homes:
        return None, [off_warning("the workspace is the home folder")]
    if _inside(workspace, agent_data) or _inside(agent_data, workspace):
        return None, [off_warning("the workspace holds or lies inside the agent's data folder")]
    try:
        make_private_dirs(root)
        if not os.access(root, os.W_OK):
            raise OSError("not writable")
    except OSError as exc:
        return None, [off_warning(f"cannot write {root}: {exc.strerror or exc}")]
    prune_stale(root)
    return CheckpointStore(workspace, root, guard, env, git=git), []
