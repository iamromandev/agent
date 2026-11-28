"""FormatHub.format: the note counts, failures and restore, the timeout, and the files it never touches."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from core.engine.format.config import FormatSetup, FormatterSpec
from core.engine.format.runner import FormatHub
from core.engine.guardrails.setup import Guardrails
from core.engine.lsp.config import Source, take_fingerprint
from core.testing.format_fakes import fake_formatter

EXTS = frozenset({".txt", ".pem"})


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root.resolve()


def hub_for(
    tmp_path: Path, workspace: Path, mode: str, *, timeout: float = 5, args: tuple[str, ...] = ("{file}",), env: dict[str, str] | None = None,
    source: Source = "global", fingerprint: bool = False, project_active: Callable[[], bool] = lambda: True
) -> FormatHub:
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    program = str(fake_formatter(folder, "fmt", mode))
    spec = FormatterSpec("fmt", (program, *args), EXTS, env or {}, source, take_fingerprint(program) if fingerprint else None)
    return FormatHub(FormatSetup(True, (spec,), (), ()), workspace, Guardrails.off(workspace), timeout=timeout, project_active=project_active)


def put(workspace: Path, name: str = "a.txt", text: str = "hello\nworld\n") -> tuple[Path, bytes]:
    path = workspace / name
    path.write_text(text)
    return path, text.encode()


async def test_a_change_is_counted_in_changed_lines(tmp_path: Path, workspace: Path) -> None:
    path, data = put(workspace)
    [outcome] = await hub_for(tmp_path, workspace, "upper").format(path, data)
    assert (outcome.name, outcome.changed_lines, outcome.error) == ("fmt", 4, None)
    assert path.read_text() == "HELLO\nWORLD\n"


async def test_no_change_gives_zero_lines(tmp_path: Path, workspace: Path) -> None:
    path, data = put(workspace)
    [outcome] = await hub_for(tmp_path, workspace, "same").format(path, data)
    assert outcome.changed_lines == 0 and outcome.error is None


@pytest.mark.parametrize("mode", ["fail", "empty"])
async def test_a_failing_formatter_leaves_the_written_bytes(tmp_path: Path, workspace: Path, mode: str) -> None:
    path, data = put(workspace)
    [outcome] = await hub_for(tmp_path, workspace, mode).format(path, data)
    assert path.read_bytes() == data and outcome.changed_lines == 0 and outcome.error
    assert ("boom" in outcome.error) == (mode == "fail")


async def test_a_hanging_formatter_is_killed_within_its_timeout_plus_a_second(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_FORMAT_DUMP", str(tmp_path / "dump.jsonl"))
    path, data = put(workspace)
    started = time.monotonic()
    [outcome] = await hub_for(tmp_path, workspace, "sleep", timeout=1).format(path, data)
    assert time.monotonic() - started < 2 and "timed out" in (outcome.error or "") and path.read_bytes() == data
    pgid = json.loads((tmp_path / "dump.jsonl").read_text().splitlines()[0])["pgid"]
    for _ in range(50):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


async def test_esc_restores_the_written_file_and_propagates(tmp_path: Path, workspace: Path) -> None:
    path, data = put(workspace)
    task = asyncio.create_task(hub_for(tmp_path, workspace, "sleep", timeout=30).format(path, data))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert path.read_bytes() == data


async def test_the_file_is_one_argv_item_and_the_run_is_in_the_workspace_with_the_extra_env(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_FORMAT_DUMP", str(tmp_path / "dump.jsonl"))
    monkeypatch.setenv("GH_TOKEN", "secretsecretsecret")
    path, data = put(workspace, "a b;$(x).txt")
    hub = hub_for(tmp_path, workspace, "same", args=("--in-place", "x={file}"), env={"MY_FMT": "1"})
    await hub.format(path, data)
    seen = json.loads((tmp_path / "dump.jsonl").read_text())
    assert seen["argv"] == ["--in-place", f"x={path}"] and Path(seen["cwd"]).resolve() == workspace
    assert "MY_FMT" in seen["env"] and "GH_TOKEN" not in seen["env"]


async def test_without_a_placeholder_the_path_is_appended(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_FORMAT_DUMP", str(tmp_path / "dump.jsonl"))
    path, data = put(workspace)
    await hub_for(tmp_path, workspace, "same", args=("-q",)).format(path, data)
    assert json.loads((tmp_path / "dump.jsonl").read_text())["argv"] == ["-q", str(path)]


async def test_files_it_must_not_touch_are_left_alone(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_FORMAT_DUMP", str(tmp_path / "dump.jsonl"))
    hub = hub_for(tmp_path, workspace, "upper")
    protected, data = put(workspace, "key.pem")
    outside = tmp_path / "out.txt"
    outside.write_text("hello\n")
    target, _ = put(workspace, "t.txt")
    link = workspace / "link.txt"
    link.symlink_to(target)
    linked, ldata = put(workspace, "l.txt")
    os.link(linked, workspace / "l2.txt")
    changed, _ = put(workspace, "c.txt")
    cases = [(protected, data), (outside, b"hello\n"), (link, b"hello\nworld\n"), (linked, ldata), (changed, b"something else\n")]
    for path, written in cases:
        assert await hub.format(path, written) == []
    assert not (tmp_path / "dump.jsonl").exists()


async def test_a_program_changed_since_launch_is_refused(tmp_path: Path, workspace: Path) -> None:
    path, data = put(workspace)
    hub = hub_for(tmp_path, workspace, "upper", fingerprint=True)
    program = tmp_path / "bin" / "fmt"
    program.write_text(program.read_text() + "\n# changed\n")
    [outcome] = await hub.format(path, data)
    assert outcome.error and "changed since launch" in outcome.error and path.read_bytes() == data


async def test_an_untrusted_or_withdrawn_formatter_does_not_run(tmp_path: Path, workspace: Path) -> None:
    path, data = put(workspace)
    hub = hub_for(tmp_path, workspace, "upper", source="project", project_active=lambda: False)
    assert await hub.format(path, data) == [] and [s.name for s in hub.untrusted()] == ["fmt"]
    trusted = hub_for(tmp_path, workspace, "upper", source="project", project_active=lambda: True)
    trusted.withdraw(["fmt"], "config changed")
    assert await trusted.format(path, data) == [] and path.read_bytes() == data
    assert trusted.listing() == [f"fmt (project) off: config changed  {' '.join(sorted(EXTS))}"]


async def test_a_restore_after_a_failure_keeps_the_mode_and_leaves_no_temp_file(tmp_path: Path, workspace: Path) -> None:
    path, data = put(workspace)
    path.chmod(0o640)
    inode = path.stat().st_ino
    [outcome] = await hub_for(tmp_path, workspace, "empty").format(path, data)
    assert outcome.error and path.read_bytes() == data
    assert (path.stat().st_mode & 0o777) == 0o640
    assert path.stat().st_ino != inode  # replaced by a rename, not rewritten in place
    assert [p.name for p in workspace.iterdir()] == ["a.txt"]


async def test_the_eligibility_checks_run_off_the_event_loop(tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    path, data = put(workspace)
    hub = hub_for(tmp_path, workspace, "same")
    threads: list[int] = []
    original = hub._eligible

    def spy(p: Path, w: bytes) -> bool:
        threads.append(threading.get_ident())
        return original(p, w)

    monkeypatch.setattr(hub, "_eligible", spy)
    await hub.format(path, data)
    assert threads and threads[0] != threading.get_ident()
