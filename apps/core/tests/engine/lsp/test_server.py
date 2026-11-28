"""The language-server client against the fake server in each mode."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from core.engine.lsp import server as server_module
from core.engine.lsp.config import ServerSpec
from core.engine.lsp.server import Diagnostic, LspServer
from core.testing.helpers import assert_all_gone, wait_until
from core.testing.lsp_fakes import fake_lsp


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    return root.resolve()


@pytest.fixture(autouse=True)
def _quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "QUIET", 0.1)


class Maker:
    def __init__(self, tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path, self.workspace, self.monkeypatch = tmp_path, workspace, monkeypatch
        self.made: list[LspServer] = []
        self.dump = tmp_path / "dump.jsonl"
        monkeypatch.setenv("FAKE_LSP_DUMP", str(self.dump))

    def __call__(self, mode: str, **env: str) -> LspServer:
        self.monkeypatch.setenv("FAKE_LSP_MODE", mode)
        for key, value in env.items():
            self.monkeypatch.setenv(key, value)
        exe = fake_lsp(self.tmp_path, "ty")
        spec = ServerSpec("ty", str(exe), (), frozenset({".py"}), {}, "global")
        found = LspServer(spec, self.workspace, dict(os.environ))
        self.made.append(found)
        return found


@pytest.fixture
def make(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Maker]:
    maker = Maker(tmp_path, workspace, monkeypatch)
    yield maker
    for found in maker.made:
        found.kill()


def lines(dump: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in dump.read_text().splitlines()]


@pytest.mark.parametrize("mode", ["pull", "push"])
async def test_diagnostics_by_pull_or_by_push_with_the_version(make: Maker, workspace: Path, mode: str) -> None:
    server = make(mode)
    got = await server.diagnostics(workspace / "a.py", "x = 1\n", wait=3, start_wait=5)
    assert got == [Diagnostic(1, 5, 1, "bad thing", "E1"), Diagnostic(2, 1, 2, "meh", None)]
    assert server.state == "running"
    await server.stop()


async def test_a_second_call_reuses_the_process_and_sends_a_change(make: Maker, workspace: Path) -> None:
    server = make("pull")
    await server.diagnostics(workspace / "a.py", "x = 1\n", wait=3, start_wait=5)
    await server.diagnostics(workspace / "a.py", "x = 2\n", wait=3, start_wait=5)
    seen = lines(make.dump)
    assert len({x["pid"] for x in seen}) == 1
    assert [x["method"] for x in seen if str(x["method"]).startswith("textDocument/did")] == ["textDocument/didOpen", "textDocument/didChange"]
    assert seen[0]["cwd"] == str(workspace) and seen[0]["pgid"] == seen[0]["pid"]
    await server.stop()


async def test_the_hostile_server_cannot_show_stale_or_foreign_diagnostics(make: Maker, workspace: Path) -> None:
    server = make("hostile", FAKE_LSP_REPORT=json.dumps([{"line": 0, "character": 0, "severity": 1, "message": "real"}]))
    got = await server.diagnostics(workspace / "a.py", "x\n", wait=3, start_wait=5)
    assert got == [Diagnostic(1, 1, 1, "real", None)]
    await server.stop()


async def test_a_silent_server_times_out_with_a_reason_and_no_diagnostics(make: Maker, workspace: Path) -> None:
    server = make("silent")
    got = await server.diagnostics(workspace / "a.py", "x\n", wait=0.4, start_wait=5)
    assert got is None and server.state == "timed out" and server.reason == "did not answer within 0.4 s"
    await server.stop()


@pytest.mark.parametrize("mode", ["crash", "garbage", "flood"])
async def test_a_bad_server_is_exited_and_never_restarted(make: Maker, workspace: Path, mode: str) -> None:
    server = make(mode)
    assert await server.diagnostics(workspace / "a.py", "x\n", wait=1, start_wait=5) is None
    assert server.state == "exited" and server.reason is not None and server.reason.startswith("exited (code ")
    before = len(lines(make.dump))
    assert await server.diagnostics(workspace / "a.py", "y\n", wait=1, start_wait=5) is None
    assert len(lines(make.dump)) == before  # no new process, nothing sent


async def test_a_missing_program_is_could_not_start(workspace: Path) -> None:
    spec = ServerSpec("ty", str(workspace / "nope"), (), frozenset({".py"}), {}, "global")
    server = LspServer(spec, workspace, dict(os.environ))
    assert await server.diagnostics(workspace / "a.py", "x\n", wait=1, start_wait=2) is None
    assert server.state == "exited" and server.reason is not None and server.reason.startswith("could not start: ")


async def test_stop_and_kill_end_the_process_group(make: Maker, workspace: Path) -> None:
    server = make("pull")
    await server.diagnostics(workspace / "a.py", "x\n", wait=3, start_wait=5)
    pid = int(lines(make.dump)[0]["pid"])
    await server.stop()
    await assert_all_gone(pid)
    again = make("pull")
    await again.diagnostics(workspace / "a.py", "x\n", wait=3, start_wait=5)
    pid2 = max(int(x["pid"]) for x in lines(make.dump))
    again.kill()
    await assert_all_gone(pid2)


async def test_esc_while_the_server_starts_kills_it_and_marks_it_exited(make: Maker, workspace: Path) -> None:
    server = make("slowinit")
    task = asyncio.ensure_future(server.diagnostics(workspace / "a.py", "x\n", wait=3, start_wait=30))
    await wait_until(lambda: make.dump.exists() and "initialize" in make.dump.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert server.state == "exited" and server.reason == "start interrupted"
    await assert_all_gone(int(lines(make.dump)[0]["pid"]))
    assert await server.diagnostics(workspace / "a.py", "x\n", wait=1, start_wait=2) is None


async def test_an_initialize_error_kills_the_server_and_says_why(make: Maker, workspace: Path) -> None:
    server = make("initerror")
    assert await server.diagnostics(workspace / "a.py", "x\n", wait=1, start_wait=5) is None
    assert server.state == "exited" and server.reason is not None and server.reason.startswith("could not start: ") and "boom" in server.reason
    await assert_all_gone(int(lines(make.dump)[0]["pid"]))


async def test_a_flood_of_publications_for_other_files_is_not_kept(make: Maker, workspace: Path) -> None:
    server = make("spray")
    got = await server.diagnostics(workspace / "a.py", "x\n", wait=10, start_wait=5)
    assert got is not None and len(got) == 2
    assert len(server._published) <= 1
    await server.stop()


async def test_a_message_for_a_known_file_keeps_only_the_first_items(
    make: Maker, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = {"line": 0, "character": 0, "severity": 1, "message": "m"}
    server = make("pull", FAKE_LSP_REPORT=json.dumps({"repeat": 5000, "item": item}))
    got = await server.diagnostics(workspace / "a.py", "x\n", wait=5, start_wait=5)
    assert got is not None and len(got) == server_module.MAX_ITEMS == 1000
    await server.stop()


async def test_after_the_process_ended_kill_signals_no_group(
    make: Maker, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = make("crash")
    assert await server.diagnostics(workspace / "a.py", "x\n", wait=1, start_wait=5) is None
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(server_module.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    server.kill()
    await server.stop()
    assert sent == [] and server.pgid is None


async def test_a_server_that_closes_stdout_but_keeps_running_is_killed_and_exited(make: Maker, workspace: Path) -> None:
    server = make("eof")
    started = time.monotonic()
    assert await server.diagnostics(workspace / "a.py", "x\n", wait=5, start_wait=5) is None
    assert time.monotonic() - started < 3 and server.state == "exited"
    await assert_all_gone(int(lines(make.dump)[0]["pid"]))


async def test_a_push_without_a_version_is_accepted_after_the_quiet_period(make: Maker, workspace: Path) -> None:
    server = make("nover")
    got = await server.diagnostics(workspace / "a.py", "x\n", wait=3, start_wait=5)
    assert got == [Diagnostic(1, 5, 1, "bad thing", "E1"), Diagnostic(2, 1, 2, "meh", None)]
    await server.stop()


async def test_shutdown_and_exit_go_without_a_params_key(make: Maker, workspace: Path) -> None:
    server = make("pull")
    await server.diagnostics(workspace / "a.py", "x\n", wait=3, start_wait=5)
    await server.stop()
    seen = {x["method"]: x["params"] for x in lines(make.dump)}
    assert seen["shutdown"] is False and seen["exit"] is False and seen["initialize"] is True


async def test_the_first_call_has_one_deadline_for_the_start_and_the_wait(make: Maker, workspace: Path) -> None:
    server = make("slowinit", FAKE_LSP_DELAY="0.5")
    started = time.monotonic()
    got = await server.diagnostics(workspace / "a.py", "x\n", wait=10, start_wait=1.5)
    assert got is None and server.state == "timed out"
    assert time.monotonic() - started < 3  # not start + wait (about 10.5 s)
    server.kill()


async def test_a_bare_command_is_never_resolved_by_the_spawn(workspace: Path) -> None:
    spec = ServerSpec("ty", "ty", (), frozenset({".py"}), {}, "global")
    server = LspServer(spec, workspace, dict(os.environ))
    assert await server.diagnostics(workspace / "a.py", "x\n", wait=1, start_wait=2) is None
    assert server.state == "exited" and server.reason is not None and server.reason.startswith("could not start: ")
