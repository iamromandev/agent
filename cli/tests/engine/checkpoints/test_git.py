"""GitRunner and its environment: isolation from the user's git, versions, deadlines."""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest
from src.engine.checkpoints.git import GitError, GitRunner, GitTimeout, git_env, git_version


def test_git_env_drops_inherited_git_variables_and_pins_the_shadow_repo(tmp_path: Path) -> None:
    base = {"PATH": "/usr/bin", "GIT_DIR": "/elsewhere", "GIT_INDEX_FILE": "/x", "GIT_SSH_COMMAND": "evil"}

    env = git_env(base, tmp_path / "g", tmp_path / "g" / "index", tmp_path / "home")

    assert env["GIT_DIR"] == str(tmp_path / "g")
    assert env["GIT_INDEX_FILE"] == str(tmp_path / "g" / "index")
    assert "GIT_SSH_COMMAND" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_git_env_passes_only_what_the_base_has(tmp_path: Path) -> None:
    env = git_env({"PATH": "/bin"}, tmp_path, tmp_path / "i", tmp_path / "home")

    assert "FOO_TOKEN" not in env


def test_git_env_points_home_and_xdg_config_at_the_private_folder(tmp_path: Path) -> None:
    base = {"PATH": "/bin", "HOME": "/users/me", "XDG_CONFIG_HOME": "/users/me/.config"}

    env = git_env(base, tmp_path / "g", tmp_path / "g" / "index", tmp_path / "home")

    assert env["HOME"] == str(tmp_path / "home")
    assert env["XDG_CONFIG_HOME"] == str(tmp_path / "home")


def test_git_version_is_none_for_a_missing_program(tmp_path: Path) -> None:
    assert git_version(str(tmp_path / "nope"), {"PATH": str(tmp_path)}) is None


def fake_git(tmp_path: Path, body: str) -> str:
    """The absolute path of a `git` that runs `body`."""
    script = tmp_path / "bin" / "git"
    script.parent.mkdir(exist_ok=True)
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def runner(tmp_path: Path, git: str, exclude: Path | None = None) -> GitRunner:
    hooks = tmp_path / "hooks"
    hooks.mkdir(exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return GitRunner(git, tmp_path / "g", tmp_path, {"PATH": "/bin:/usr/bin"}, hooks, home, exclude=exclude or Path(os.devnull))


async def test_a_failing_git_raises_with_trimmed_stderr(tmp_path: Path) -> None:
    git = fake_git(tmp_path, "echo boom >&2; exit 3")

    with pytest.raises(GitError, match="boom"):
        await runner(tmp_path, git).run("status")


async def test_a_slow_git_is_killed_with_its_group_on_timeout(tmp_path: Path) -> None:
    marker = tmp_path / "child.pid"
    git = fake_git(tmp_path, f"sleep 30 &\necho $! > {marker}\nwait")

    with pytest.raises(GitTimeout):
        await runner(tmp_path, git).run("status", timeout=0.5)

    pid = int(marker.read_text())
    await asyncio.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
