"""GitRunner and its environment: isolation from the user's git, versions, deadlines."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
from pathlib import Path

import pytest
from src.engine.checkpoints.git import GitError, GitRunner, GitTimeout, find_git, git_env, git_version


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("git version 2.39.3 (Apple Git-145)\n", (2, 39)),
        ("git version 2.20.0", (2, 20)),
        ("nonsense", None),
    ],
)
def test_git_version_parses_the_output(tmp_path: Path, text: str, expected: tuple[int, int] | None) -> None:
    git = fake_git(tmp_path, f"echo '{text}'")

    assert git_version(git, {"PATH": "/bin:/usr/bin"}) == expected


def test_git_version_is_none_for_a_missing_program(tmp_path: Path) -> None:
    assert git_version(str(tmp_path / "nope"), {"PATH": str(tmp_path)}) is None


def fake_git(tmp_path: Path, body: str) -> str:
    """The absolute path of a `git` that runs `body`."""
    script = tmp_path / "bin" / "git"
    script.parent.mkdir(exist_ok=True)
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


@pytest.mark.parametrize("template", [".:{}", ":{}", "{}:."])
def test_find_git_ignores_relative_and_empty_path_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, template: str
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    marker = tmp_path / "planted-ran"
    (workspace / "git").write_text(f"#!/bin/sh\ntouch {marker}\n")
    (workspace / "git").chmod(0o755)
    real = tmp_path / "real"
    real.mkdir()
    (real / "git").write_text("#!/bin/sh\necho git version 2.39.0\n")
    (real / "git").chmod(0o755)
    monkeypatch.chdir(workspace)

    found = find_git({"PATH": template.format(real)}, workspace)

    assert found == str(real / "git")
    assert not marker.exists()


def test_find_git_finds_nothing_when_only_the_current_folder_is_on_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "git").write_text("#!/bin/sh\n")
    (tmp_path / "git").chmod(0o755)
    monkeypatch.chdir(tmp_path)

    assert find_git({"PATH": "."}, tmp_path / "other") is None


def test_find_git_refuses_a_git_inside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "bin").mkdir(parents=True)
    (workspace / "bin" / "git").write_text("#!/bin/sh\n")
    (workspace / "bin" / "git").chmod(0o755)

    assert find_git({"PATH": str(workspace / "bin")}, workspace) is None


def test_find_git_is_none_without_a_path(tmp_path: Path) -> None:
    assert find_git({}, tmp_path) is None


def runner(tmp_path: Path, git: str, exclude: Path | None = None) -> GitRunner:
    hooks = tmp_path / "hooks"
    hooks.mkdir(exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return GitRunner(git, tmp_path / "g", tmp_path, {"PATH": "/bin:/usr/bin"}, hooks, home, exclude=exclude or Path(os.devnull))


async def test_run_returns_stdout(tmp_path: Path) -> None:
    git = fake_git(tmp_path, "printf hello")

    assert await runner(tmp_path, git).run("status") == b"hello"


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


async def test_stdin_reaches_git(tmp_path: Path) -> None:
    git = fake_git(tmp_path, "cat")

    assert await runner(tmp_path, git).run("x", stdin=b"abc") == b"abc"


async def test_options_hide_the_users_hooks_and_filters(tmp_path: Path) -> None:
    git = fake_git(tmp_path, 'for a in "$@"; do echo "$a"; done')

    out = (await runner(tmp_path, git).run("status")).decode()

    assert f"core.hooksPath={tmp_path / 'hooks'}" in out
    assert "core.fsmonitor=false" in out
    assert "commit.gpgSign=false" in out
    assert f"core.excludesFile={os.devnull}" in out


async def test_a_given_exclude_file_is_the_excludes_file(tmp_path: Path) -> None:
    git = fake_git(tmp_path, 'for a in "$@"; do echo "$a"; done')
    exclude = tmp_path / "exclude"

    out = (await runner(tmp_path, git, exclude).run("status")).decode()

    assert f"core.excludesFile={exclude}" in out
    assert out.count("core.excludesFile=") == 1


async def test_git_runs_with_the_given_program_and_a_private_home(tmp_path: Path) -> None:
    git = fake_git(tmp_path, 'echo "$HOME|$XDG_CONFIG_HOME|$0"')

    out = (await runner(tmp_path, git).run("status")).decode().strip()

    assert out == f"{tmp_path / 'home'}|{tmp_path / 'home'}|{git}"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_real_git_version_is_known() -> None:
    git = find_git(dict(os.environ), Path("/nonexistent-workspace"))
    assert git is not None
    assert git_version(git, dict(os.environ)) is not None
