"""AC1/AC5: no child process is started behind `sandbox-exec`: every spawn site starts its own argv."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

import pytest
from src.engine.guardrails.setup import Guardrails
from src.engine.hooks.runner import ProcessRunner
from src.engine.image_pull import download_size
from src.engine.imagegen import run_helper
from src.engine.tools.base import Prepared, ToolContext, ToolResult
from src.engine.tools.shell import BashTool

from tests.engine.test_image_pull_size import SAMPLE_A
from tests.image_fakes import fake_hf


@pytest.fixture(autouse=True)
def sh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    seen: list[tuple[str, ...]] = []
    real = asyncio.create_subprocess_exec

    async def spy(*argv: str, **kwargs: Any):
        seen.append(tuple(argv))
        return await real(*argv, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    return seen


async def _bash(command: str, workspace: Path) -> ToolResult:
    prepared = await BashTool().prepare({"command": command}, ToolContext(workspace=workspace, bash_timeout=5))
    assert isinstance(prepared, Prepared)
    return await prepared.execute()


async def test_bash_starts_the_shell_directly(tmp_path: Path, spawned: list[tuple[str, ...]]) -> None:
    await _bash("true", tmp_path)
    assert spawned == [("/bin/sh", "-c", "true")]


async def test_a_denial_message_gets_no_note(tmp_path: Path) -> None:
    result = await _bash("echo Operation not permitted; exit 1", tmp_path)
    assert result.output == "Operation not permitted\nexit 1"


async def test_a_hook_starts_the_shell_directly(tmp_path: Path, spawned: list[tuple[str, ...]]) -> None:
    await ProcessRunner(tmp_path, Guardrails.off(tmp_path)).run("true", b"", 10, {})
    assert spawned == [("/bin/sh", "-c", "true")]


async def test_run_helper_starts_its_argv_directly(tmp_path: Path, spawned: list[tuple[str, ...]]) -> None:
    await run_helper(["/usr/bin/true"], tmp_path, Guardrails.off(tmp_path), 20)
    assert spawned == [("/usr/bin/true",)]


async def test_the_hf_dry_run_starts_hf_directly(tmp_path: Path, spawned: list[tuple[str, ...]]) -> None:
    hf = fake_hf(tmp_path, SAMPLE_A)
    await download_size(hf, "a/b", Guardrails.off(tmp_path))
    assert spawned == [(str(hf), "download", "a/b", "--dry-run")]


async def _terminal_argv(monkeypatch: pytest.MonkeyPatch, workspace: Path, command: str | None) -> list[str]:
    from src.engine.terminal import manager as terminal_manager
    from src.engine.terminal.manager import TerminalError, TerminalManager

    seen: list[list[str]] = []

    def spy(argv: list[str], *args: Any, **kwargs: Any) -> Any:
        seen.append(list(argv))
        raise OSError("spy")

    monkeypatch.setattr(terminal_manager.PtyProcess, "spawn", spy)
    with pytest.raises(TerminalError):
        TerminalManager(workspace).start("t", command)
    return seen[0]


async def test_term_start_spawns_the_shell_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _terminal_argv(monkeypatch, tmp_path, "true") == ["/bin/sh", "-c", "true"]


async def test_the_shell_tab_spawns_the_shell_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _terminal_argv(monkeypatch, tmp_path, None) == ["/bin/sh", "-i"]


async def test_the_mcp_connector_uses_the_configured_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.client import mcp as client_mcp
    from src.engine.mcp.config import ServerConfig

    seen: list[Any] = []

    class Spy:
        def __init__(self, params: Any, pids: set[int]) -> None:
            seen.append(params)

        async def start(self) -> None:
            return None

    monkeypatch.setattr(client_mcp, "SdkConnection", Spy)
    await client_mcp.SdkConnector().connect(ServerConfig("s", "/usr/bin/server", ("--a", "b")), tmp_path)
    assert (seen[0].command, seen[0].args) == ("/usr/bin/server", ["--a", "b"])


@pytest.mark.parametrize("kind", ["generate", "edit", "upscale"])
def test_mflux_argv_starts_with_the_command(tmp_path: Path, kind: Literal["generate", "edit", "upscale"]) -> None:
    from src.engine.imagegen import ImageRequest, MfluxGenerator

    command = tmp_path / "mflux-x"
    generator = MfluxGenerator(command, "x/y", 5, Guardrails.off(tmp_path), kind=kind)
    inputs = (tmp_path / "in-0.png",) if kind != "generate" else ()
    request = ImageRequest("p", 256, 256, 1, 0, inputs=inputs, scale=2 if kind == "upscale" else None)
    argv = generator.argv(request, tmp_path)
    assert argv[0] == str(command)
    assert not any("sandbox-exec" in item for item in argv)
