"""run_helper: the compositing subprocess's environment, exit code, stderr, timeout and cancel (AC11, AC12)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import ImageGenError, run_helper


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _script(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "h.py"
    path.write_text(body)
    return [sys.executable, "-I", str(path)]


async def test_the_exit_code_and_stderr_come_back(tmp_path: Path) -> None:
    argv = _script(tmp_path, "import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n")
    code, stderr = await run_helper(argv, tmp_path, Guardrails.off(tmp_path), 20)
    assert (code, stderr.strip()) == (3, b"boom")


async def test_stderr_is_capped(tmp_path: Path) -> None:
    argv = _script(tmp_path, "import sys\nsys.stderr.write('x' * 500000)\n")
    code, stderr = await run_helper(argv, tmp_path, Guardrails.off(tmp_path), 20)
    assert code == 0
    assert len(stderr) <= 64 * 1024


async def test_it_runs_with_a_filtered_environment_in_its_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard = Guardrails.off(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecretsecret")
    dump = tmp_path / "dump.json"
    argv = _script(
        tmp_path, f"import json, os\njson.dump({{'env': dict(os.environ), 'cwd': os.getcwd()}}, open({str(dump)!r}, 'w'))\n"
    )
    await run_helper(argv, tmp_path, guard, 20)
    seen = json.loads(dump.read_text())
    assert "HF_TOKEN" not in seen["env"]
    assert Path(seen["cwd"]).resolve() == tmp_path.resolve()


async def test_a_timeout_kills_the_group(tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    argv = _script(
        tmp_path,
        f"import os, subprocess, time\nc = subprocess.Popen(['sleep', '60'])\nopen({str(pids)!r}, 'w').write('%d %d' % (os.getpid(), c.pid))\ntime.sleep(60)\n",
    )
    started = time.monotonic()
    with pytest.raises(ImageGenError, match="timed out after 1 s"):
        await run_helper(argv, tmp_path, Guardrails.off(tmp_path), 1)
    assert time.monotonic() - started < 10
    for pid in map(int, pids.read_text().split()):
        for _ in range(50):
            if _gone(pid):
                break
            await asyncio.sleep(0.1)
        assert _gone(pid)


async def test_a_cancel_kills_the_group_and_propagates(tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    argv = _script(tmp_path, f"import os, time\nopen({str(pids)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    task = asyncio.create_task(run_helper(argv, tmp_path, Guardrails.off(tmp_path), None))
    for _ in range(100):
        if pids.exists() and pids.read_text():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(pids.read_text())
    for _ in range(50):
        if _gone(pid):
            break
        await asyncio.sleep(0.1)
    assert _gone(pid)


async def test_a_missing_program_is_an_image_error(tmp_path: Path) -> None:
    with pytest.raises(ImageGenError, match="Could not start"):
        await run_helper([str(tmp_path / "nope")], tmp_path, Guardrails.off(tmp_path), 5)
