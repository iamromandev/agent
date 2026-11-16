"""The macOS sandbox for real: every test runs a command under `sandbox-exec` (R11, R12, NFR2). Skipped elsewhere."""

from __future__ import annotations

import asyncio
import http.server
import os
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.client.mcp import SdkConnection, SdkConnector
from src.engine.guardrails.sandbox import SANDBOX_EXEC
from src.engine.guardrails.setup import Guardrails, prepare
from src.engine.mcp.config import ServerConfig
from src.engine.terminal.manager import TerminalManager
from src.engine.tools.base import Prepared, ToolContext
from src.engine.tools.shell import BashTool

from tests.helpers import wait_until

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not SANDBOX_EXEC.exists(), reason="needs macOS and /usr/bin/sandbox-exec"
)

SAMPLE = Path(__file__).resolve().parents[3] / "evals" / "fixtures" / "sample"


def online() -> bool:
    try:
        socket.create_connection(("example.com", 443), timeout=2).close()
    except OSError:
        return False
    return True


needs_network = pytest.mark.skipif(not online(), reason="needs network access")


ENV = {k: v for k, v in os.environ.items() if not k.startswith("XDG_")}  # the agent folders are under the tmp home


class Rig:
    def __init__(self, workspace: Path, home: Path, guard: Guardrails) -> None:
        self.workspace = workspace
        self.home = home
        self.guard = guard

    def run(self, command: str, *, cwd: Path | None = None, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if k != "HOME"} | {"HOME": str(self.home)}
        argv = self.guard.sandbox.wrap(["/bin/sh", "-c", command])
        return subprocess.run(argv, cwd=cwd or self.workspace, env=env, capture_output=True, text=True, timeout=timeout)


def make_rig(
    tmp_path: Path, global_data: dict | None = None, project_data: dict | None = None, *, workspace_is_home: bool = False
) -> Rig:
    home = (tmp_path / "home").resolve()
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "config").write_text("Host secret\n")
    workspace = home if workspace_is_home else (tmp_path / "ws").resolve()
    workspace.mkdir(exist_ok=True)
    guard = prepare(
        workspace, global_data, project_data, ENV, "Darwin", home=home, sessions=home / ".local" / "share" / "agent" / "sessions"
    )
    assert guard.sandbox.profile is not None, guard.warnings
    return Rig(workspace, home, guard)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return make_rig(tmp_path)


@pytest.fixture
def outside() -> Iterator[Path]:
    """A file name in the real home folder, which the sandbox must not let a command create. Removed if it did."""
    path = Path.home() / f".agent-sandbox-test-{uuid.uuid4().hex}"
    yield path
    path.unlink(missing_ok=True)


def test_a_write_outside_the_allowed_folders_is_blocked(rig: Rig, outside: Path) -> None:
    done = rig.run(f"touch {outside}")

    assert done.returncode != 0
    assert "Operation not permitted" in done.stderr
    assert not outside.exists()


def test_a_write_in_the_workspace_and_the_temp_folder_works(rig: Rig) -> None:
    done = rig.run('touch ./inside && touch "$(mktemp -d)/x" && echo ok')

    assert done.stdout.strip() == "ok", done.stderr
    assert (rig.workspace / "inside").exists()


def test_a_cache_folder_is_writable(rig: Rig) -> None:
    done = rig.run("mkdir -p ~/.cache/uv && touch ~/.cache/uv/probe && echo ok")

    assert done.stdout.strip() == "ok", done.stderr


def test_the_devices_a_script_may_write_stay_writable(rig: Rig) -> None:
    done = rig.run("echo a > /dev/null; echo b > /dev/stdout; echo c > /dev/stderr; echo d | tee /dev/stderr > /dev/null")

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "b"


def test_a_protected_folder_cannot_be_read(rig: Rig) -> None:
    done = rig.run("cat ~/.ssh/config")

    assert done.returncode != 0
    assert "Operation not permitted" in done.stderr
    assert "Host secret" not in done.stdout


def test_a_protected_folder_cannot_be_read_through_the_firmlink_spelling(rig: Rig) -> None:
    done = rig.run(f"cat /System/Volumes/Data{rig.home}/.ssh/config")

    assert done.returncode != 0
    assert "Host secret" not in done.stdout


def test_files_with_a_protected_name_cannot_be_read_but_templates_can(rig: Rig) -> None:
    (rig.workspace / ".env").write_text("SECRET=1\n")
    (rig.workspace / ".env.example").write_text("SECRET=\n")
    (rig.workspace / "server.PEM").write_text("pem\n")
    (rig.workspace / "notes.txt").write_text("fine\n")

    denied = rig.run("cat .env; cat server.PEM")
    allowed = rig.run("cat .env.example notes.txt")

    assert "SECRET=1" not in denied.stdout and "pem" not in denied.stdout
    assert denied.stderr.count("Operation not permitted") == 2
    assert allowed.stdout == "SECRET=\nfine\n"


def test_a_settings_pattern_is_enforced_for_commands_too(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {"guardrails": {"protected": ["secrets/**", "*.token"]}})
    (rig.workspace / "secrets" / "deep").mkdir(parents=True)
    (rig.workspace / "secrets" / "deep" / "a.txt").write_text("hidden\n")
    (rig.workspace / "x.token").write_text("hidden\n")

    done = rig.run("cat secrets/deep/a.txt; cat x.token")

    assert "hidden" not in done.stdout
    assert done.stderr.count("Operation not permitted") == 2


def test_system_certificates_stay_readable_despite_the_pem_rule(rig: Rig) -> None:
    done = rig.run("head -c 20 /etc/ssl/cert.pem")

    assert done.returncode == 0, done.stderr


def test_the_agent_settings_files_cannot_be_written_or_moved_away(rig: Rig) -> None:
    agent = rig.workspace / ".agent"
    agent.mkdir()
    (agent / "settings.json").write_text("{}")

    blocked = [
        rig.run("echo x > .agent/settings.json"),
        rig.run("echo x > .agent/mcp.json"),
        rig.run("mv .agent .agent-away"),
        rig.run("rm .agent/settings.json"),
        rig.run("mv ../ws ../ws-away"),
    ]
    allowed = rig.run("echo x > .agent/other && echo ok")

    assert all(done.returncode != 0 for done in blocked), [d.stderr for d in blocked]
    assert (agent / "settings.json").read_text() == "{}"
    assert not (agent / "mcp.json").exists()
    assert allowed.stdout.strip() == "ok", allowed.stderr


def test_the_agent_commands_and_skills_folders_cannot_be_written(rig: Rig) -> None:
    agent = rig.workspace / ".agent"
    (agent / "commands").mkdir(parents=True)
    (agent / "skills" / "s").mkdir(parents=True)
    (agent / "commands" / "c.md").write_text("old")

    blocked = [
        rig.run("echo x > .agent/commands/c.md"),
        rig.run("echo x > .agent/commands/new.md"),
        rig.run("echo x > .agent/skills/s/SKILL.md"),
        rig.run("mkdir .agent/skills/t"),
        rig.run("mv .agent/commands .agent/commands-away"),
        rig.run("rm .agent/commands/c.md"),
    ]

    assert all(done.returncode != 0 for done in blocked), [d.stderr for d in blocked]
    assert (agent / "commands" / "c.md").read_text() == "old"
    assert sorted(path.name for path in (agent / "skills").iterdir()) == ["s"]


def test_agents_md_in_the_workspace_stays_writable(rig: Rig) -> None:
    done = rig.run("echo notes > AGENTS.md && echo ok")

    assert done.stdout.strip() == "ok", done.stderr


def test_the_workspace_folder_itself_can_be_touched_extracted_into_and_copied_into(rig: Rig, tmp_path: Path) -> None:
    source = tmp_path / "src"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "f.txt").write_text("data\n")
    archive = tmp_path / "a.tar"
    subprocess.run(["tar", "-cf", str(archive), "-C", str(source), "."], check=True)

    steps = {
        "touch": rig.run("touch ."),
        "tar": rig.run(f"tar -xf {archive} -C . && tar -xf {archive}"),
        "cp": rig.run(f"cp -R {source}/. ."),
        "chmod": rig.run("chmod 755 ."),
    }
    if shutil.which("rsync"):
        steps["rsync"] = rig.run(f"rsync -a {source}/ ./")

    assert {name: done.returncode for name, done in steps.items()} == {name: 0 for name in steps}, {
        name: done.stderr for name, done in steps.items()
    }
    assert (rig.workspace / "sub" / "f.txt").read_text() == "data\n"


def test_the_workspace_folder_and_agent_folder_cannot_be_renamed_or_removed(rig: Rig) -> None:
    agent = rig.workspace / ".agent"
    agent.mkdir()
    (agent / "settings.json").write_text("{}")

    blocked = [
        rig.run("mv .agent x"),
        rig.run("rm -rf .agent"),
        rig.run("mv ../ws ../ws-away"),
        rig.run("rmdir ../ws"),
        rig.run("echo x > .agent/settings.json"),
    ]

    assert all(done.returncode != 0 for done in blocked), [d.stderr for d in blocked]
    assert (agent / "settings.json").read_text() == "{}"
    assert rig.workspace.is_dir()


# Rename and delete: a command must not move a protected file or folder somewhere its name no longer protects it.


def test_a_protected_file_cannot_be_renamed_or_removed(rig: Rig) -> None:
    (rig.workspace / ".env").write_text("SECRET=1\n")
    (rig.workspace / "k.pem").write_text("pem\n")

    blocked = [rig.run("mv .env x"), rig.run("rm .env"), rig.run("mv k.pem x.txt"), rig.run("mv k.pem ../x.txt")]
    cat = rig.run("cat x")

    assert all(done.returncode != 0 for done in blocked), [d.stderr for d in blocked]
    assert cat.stdout == ""
    assert (rig.workspace / ".env").read_text() == "SECRET=1\n"
    assert (rig.workspace / "k.pem").exists() and not (rig.workspace / "x").exists()


def test_a_protected_file_cannot_be_replaced_by_renaming_another_over_it(rig: Rig) -> None:
    (rig.workspace / ".env").write_text("SECRET=1\n")
    (rig.workspace / "other").write_text("other\n")

    assert rig.run("mv other .env").returncode != 0
    assert (rig.workspace / ".env").read_text() == "SECRET=1\n"


def test_a_folder_matched_by_a_settings_pattern_cannot_be_renamed_to_read_it(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {"guardrails": {"protected": ["secrets/**"]}})
    (rig.workspace / "secrets" / "deep").mkdir(parents=True)
    (rig.workspace / "secrets" / "k").write_text("hidden\n")
    (rig.workspace / "secrets" / "deep" / "j").write_text("hidden\n")

    blocked = [
        rig.run("mv secrets x"),
        rig.run("mv secrets/deep x"),
        rig.run("mv secrets/k x"),
        rig.run("mkdir elsewhere && mv secrets elsewhere/s"),
    ]
    rig.run("rm -rf secrets")
    cat = rig.run("cat x/k; cat x")

    assert all(done.returncode != 0 for done in blocked), [d.stderr for d in blocked]
    assert "hidden" not in cat.stdout
    assert (rig.workspace / "secrets" / "k").exists()
    assert (rig.workspace / "secrets" / "deep" / "j").exists()


def test_a_protected_folder_can_still_be_created_where_none_exists(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {"guardrails": {"protected": ["secrets/**"]}})

    done = rig.run("mkdir secrets && echo ok")

    assert done.stdout.strip() == "ok", done.stderr
    assert (rig.workspace / "secrets").is_dir()


def test_a_home_protected_folder_cannot_be_renamed_even_when_the_workspace_contains_it(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, workspace_is_home=True)

    blocked = [rig.run("mv .ssh x"), rig.run("mv .ssh/config x"), rig.run("rm .ssh/config")]
    rig.run("rm -rf .ssh")  # exits 0 whatever happens: it cannot even list a folder that reads are denied in

    assert all(done.returncode != 0 for done in blocked), [d.stderr for d in blocked]
    assert [path.name for path in rig.home.iterdir() if path.name.startswith(".ssh")] == [".ssh"]
    assert (rig.home / ".ssh" / "config").read_text() == "Host secret\n"


def test_a_non_protected_file_can_still_be_overwritten_by_a_rename(rig: Rig) -> None:
    (rig.workspace / "a").write_text("a\n")
    (rig.workspace / "b").write_text("b\n")

    done = rig.run("mv a b && cat b && rm b && echo ok")

    assert done.stdout == "a\nok\n", done.stderr


def test_the_env_templates_can_be_renamed_and_written(rig: Rig) -> None:
    (rig.workspace / ".env.example").write_text("SECRET=\n")

    done = rig.run("echo X= >> .env.example && mv .env.example .env.sample && echo ok")

    assert done.stdout.strip() == "ok", done.stderr


# A protected folder inside a writable root (the workspace is, or contains, the home folder).


def test_a_protected_folder_in_a_writable_root_cannot_be_written(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, workspace_is_home=True)

    blocked = [
        rig.run("echo x >> .ssh/config"),
        rig.run("echo x > .ssh/new"),
        rig.run("mkdir .ssh/d"),
        rig.run(f"echo x >> /System/Volumes/Data{rig.home}/.ssh/config"),
    ]
    allowed = rig.run("echo ok > fine.txt && cat fine.txt")

    assert all(done.returncode != 0 for done in blocked), [d.stderr for d in blocked]
    assert (rig.home / ".ssh" / "config").read_text() == "Host secret\n"
    assert not (rig.home / ".ssh" / "new").exists()
    assert allowed.stdout == "ok\n", allowed.stderr


def test_files_matched_by_a_pattern_in_a_writable_root_cannot_be_written(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {"guardrails": {"protected": ["secrets/**", "*.token"]}})
    (rig.workspace / "secrets").mkdir()
    (rig.workspace / "secrets" / "a.txt").write_text("hidden\n")
    (rig.workspace / "x.token").write_text("hidden\n")
    (rig.workspace / ".env").write_text("SECRET=1\n")

    blocked = [
        rig.run("echo x >> secrets/a.txt"),
        rig.run("echo x > secrets/new.txt"),
        rig.run("echo x >> x.token"),
        rig.run("echo x >> .env"),
    ]

    assert all(done.returncode != 0 for done in blocked), [d.stderr for d in blocked]
    assert (rig.workspace / "secrets" / "a.txt").read_text() == "hidden\n"
    assert (rig.workspace / ".env").read_text() == "SECRET=1\n"


# Devices and other launchers.


def test_a_pty_can_be_opened(rig: Rig) -> None:
    done = rig.run(f"{sys.executable} -c \"import pty; m, s = pty.openpty(); print('pty ok')\"")

    assert done.stdout.strip() == "pty ok", done.stderr


@pytest.mark.skipif(shutil.which("osascript") is None, reason="needs osascript")
def test_a_shell_script_run_through_osascript_stays_in_the_sandbox(rig: Rig, outside: Path) -> None:
    done = rig.run(f"osascript -e 'do shell script \"touch {outside}\"'")

    assert done.returncode != 0
    assert "Operation not permitted" in done.stderr  # the script ran, inside the sandbox
    assert not outside.exists()


def test_the_agents_own_config_data_and_sessions_cannot_be_written(rig: Rig) -> None:
    (rig.home / ".config" / "agent").mkdir(parents=True)
    done = rig.run("echo x > ~/.config/agent/settings.json")

    assert done.returncode != 0

    # Even where the folder lies inside a writable one: the sessions folder here is in the workspace.
    sessions = rig.workspace / "sessions"
    sessions.mkdir()
    guard = prepare(rig.workspace, None, None, ENV, "Darwin", home=rig.home, sessions=sessions)
    assert guard.sandbox.profile is not None
    other = Rig(rig.workspace, rig.home, guard)
    assert other.run("echo x > sessions/s.jsonl").returncode != 0
    assert other.run("echo x > other.txt").returncode == 0


def test_network_off_allows_only_local_connections(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, {"guardrails": {"sandbox": {"network": False}}})
    server = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    probe = (
        "import socket,sys\n"
        "s=socket.socket(); s.settimeout(3)\n"
        "try:\n"
        "    s.connect((sys.argv[1], int(sys.argv[2]))); print('connected')\n"
        "except PermissionError: print('blocked')\n"
        "except OSError as e: print('other', e)\n"
    )
    try:
        local = rig.run(f'{sys.executable} -c "{probe}" 127.0.0.1 {server.server_port}')
        remote = rig.run(f'{sys.executable} -c "{probe}" 192.0.2.1 80')
    finally:
        server.shutdown()

    assert local.stdout.strip() == "connected", local.stderr
    assert remote.stdout.strip() == "blocked", remote.stderr


# NFR2: normal work in a workspace still succeeds under the sandbox.


def test_git_status_and_commit_work(rig: Rig) -> None:
    done = rig.run(
        "git init -q && git status && echo hi > f && git add f && "
        "git -c user.name=a -c user.email=a@b.c -c commit.gpgsign=false commit -qm x && git log --oneline"
    )

    assert done.returncode == 0, done.stderr
    assert "x" in done.stdout


def test_make_works(rig: Rig) -> None:
    (rig.workspace / "Makefile").write_text("all:\n\techo built > out.txt\n")

    done = rig.run("make")

    assert done.returncode == 0, done.stderr
    assert (rig.workspace / "out.txt").read_text() == "built\n"


def test_zsh_and_bash_start_as_the_shell(rig: Rig) -> None:
    for shell in ("/bin/zsh", "/bin/bash"):
        done = rig.run(f"{shell} -c 'echo $0 ok'")
        assert done.returncode == 0 and "ok" in done.stdout, done.stderr


@needs_network
@pytest.mark.skipif(shutil.which("curl") is None, reason="needs curl")
def test_curl_https_works_with_the_network_on(rig: Rig) -> None:
    done = rig.run("curl -sS -m 15 -o /dev/null -w '%{http_code}' https://example.com")

    assert done.stdout == "200", done.stderr


@needs_network
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_uv_sync_and_pytest_work(rig: Rig) -> None:
    project = rig.workspace / "sample"
    shutil.copytree(SAMPLE, project)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "sample"\nversion = "0"\nrequires-python = ">=3.11"\n'
        '[dependency-groups]\ndev = ["pytest"]\n'
    )

    done = rig.run("uv sync && uv run pytest -q", cwd=project)

    assert done.returncode == 0, done.stdout + done.stderr


@needs_network
@pytest.mark.skipif(shutil.which("bun") is None, reason="needs bun")
def test_bun_install_works(rig: Rig) -> None:
    project = rig.workspace / "pkg"
    project.mkdir()
    (project / "package.json").write_text('{"name": "t", "dependencies": {"left-pad": "1.3.0"}}')

    done = rig.run("bun install", cwd=project)

    assert done.returncode == 0, done.stdout + done.stderr
    assert (project / "node_modules" / "left-pad").is_dir()


# The other two spawn sites, for real.


async def test_a_terminal_runs_in_the_sandbox_and_a_blocked_write_is_refused(
    rig: Rig, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")
    terminals = TerminalManager(rig.workspace, grace=0.3, guard=rig.guard)
    try:
        terminal = terminals.start("a", f"touch ok-file && echo wrote-inside; touch {outside} || echo blocked-outside")

        await wait_until(lambda: "blocked-outside" in "\n".join(terminal.buffer.tail(50)), 15)
    finally:
        await terminals.close_all()

    assert (rig.workspace / "ok-file").exists()
    assert not outside.exists()


async def test_the_interactive_shell_tab_can_write_to_its_terminal_device(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")
    terminals = TerminalManager(rig.workspace, grace=0.3, guard=rig.guard)
    try:
        terminal = terminals.start("shell", None)
        terminal.write(b'echo tty-$((40 + 2)) > "$(tty)"\n')

        await wait_until(lambda: "tty-42" in "\n".join(terminal.buffer.tail(50)), 15)
    finally:
        await terminals.close_all()


async def test_an_mcp_server_starts_and_answers_in_the_sandbox(rig: Rig) -> None:
    server = Path(__file__).resolve().parents[2] / "fixtures" / "fake_mcp_server.py"
    config = ServerConfig("fake", sys.executable, (str(server),), {"FAKE_MCP_MODE": "normal"})

    connection = await asyncio.wait_for(SdkConnector(guard=rig.guard).connect(config, rig.workspace), 30)
    try:
        assert isinstance(connection, SdkConnection)
        assert (await connection.call_tool("echo", {"text": "in the sandbox"}, 10)).parts == ["in the sandbox"]
    finally:
        await connection.aclose()


async def test_the_bash_tool_adds_the_sandbox_note_to_a_blocked_write(rig: Rig, outside: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")
    ctx = ToolContext(rig.workspace, 30.0, guard=rig.guard)
    prepared = await BashTool().prepare({"command": f"touch {outside}"}, ctx)

    assert isinstance(prepared, Prepared)
    result = await prepared.execute()

    assert "Operation not permitted" in result.output
    assert result.output.endswith("exit 1\n[sandbox: a write or read outside the allowed paths was blocked]")
    assert not outside.exists()
