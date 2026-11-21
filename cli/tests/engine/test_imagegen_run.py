"""MfluxGenerator against a fake mflux: argv, sandbox, environment, failures, timeout and cancel (AC10, AC12, AC14)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import ImageGenError, ImageRequest, MfluxGenerator, png_size

from tests.fakes import recording_sandbox
from tests.image_fakes import fake_mflux

REQUEST = ImageRequest("a red --flag; $(touch x) 'q'", 512, 256, 7, 1234)


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "out"
    folder.mkdir()
    return folder


def _generator(tmp_path: Path, guard: Guardrails | None = None, timeout: float = 20, model: str = "x/y") -> MfluxGenerator:
    return MfluxGenerator(fake_mflux(tmp_path), model, timeout, guard or Guardrails.off(tmp_path))


async def test_a_valid_png_is_returned(tmp_path: Path, out_dir: Path) -> None:
    path = await _generator(tmp_path).generate(REQUEST, out_dir)
    assert path.parent == out_dir
    assert png_size(path.read_bytes()) == (512, 256)


async def test_every_value_is_its_own_argv_item(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    await _generator(tmp_path, model="org/name").generate(REQUEST, out_dir)
    argv = json.loads(dump.read_text())["argv"]
    assert argv == [
        "--model", "org/name",
        f"--prompt={REQUEST.prompt}",
        "--width", "512",
        "--height", "256",
        "--steps", "7",
        "--seed", "1234",
        "--output", str(out_dir / "image.png"),
        "--no-exif",
    ]  # fmt: skip
    assert Path(json.loads(dump.read_text())["cwd"]).resolve() == out_dir.resolve()


async def test_the_command_runs_behind_the_sandbox_when_it_is_on(
    tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard, log = recording_sandbox(tmp_path, monkeypatch)
    await _generator(tmp_path, guard).generate(REQUEST, out_dir)
    assert log.read_text().strip() == "(version 1) (fake profile)"


async def test_secret_looking_variables_are_not_passed(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    monkeypatch.setenv("GITHUB_TOKEN", "sekrit-value-1234")
    await _generator(tmp_path).generate(REQUEST, out_dir)
    env = json.loads(dump.read_text())["env"]
    assert "GITHUB_TOKEN" not in env
    assert "HF_HUB_OFFLINE" not in env


async def test_a_nonzero_exit_reports_the_stderr_tail(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "fail")
    with pytest.raises(ImageGenError) as caught:
        await _generator(tmp_path).generate(REQUEST, out_dir)
    text = str(caught.value)
    assert text.startswith("mflux failed (exit 1): ")
    assert "error line 2" in text
    assert "\x1b" not in text  # escaped


async def test_the_stderr_tail_is_capped_in_lines_and_characters(
    tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "fail")
    monkeypatch.setenv("FAKE_MFLUX_LINES", "50")
    with pytest.raises(ImageGenError) as caught:
        await _generator(tmp_path).generate(REQUEST, out_dir)
    tail = str(caught.value).split(": ", 1)[1]
    assert len(tail.splitlines()) <= 20
    assert "error line 49" in tail and "error line 29" not in tail
    assert len(tail) <= 2000 * 4  # escaping can widen, never more than the cap of raw characters times 4


async def test_progress_bars_are_dropped_from_the_tail(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "tqdm")
    with pytest.raises(ImageGenError) as caught:
        await _generator(tmp_path).generate(REQUEST, out_dir)
    text = str(caught.value)
    assert "real reason" in text
    assert "it/s" not in text and "s/it" not in text


async def test_a_success_without_a_file_is_an_error(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "nothing")
    with pytest.raises(ImageGenError, match="mflux wrote no image"):
        await _generator(tmp_path).generate(REQUEST, out_dir)


async def test_a_missing_command_is_an_error(tmp_path: Path, out_dir: Path) -> None:
    generator = MfluxGenerator(tmp_path / "nope", "x/y", 5, Guardrails.off(tmp_path))
    with pytest.raises(ImageGenError, match="Could not start mflux"):
        await generator.generate(REQUEST, out_dir)


async def _pids(path: Path) -> list[int]:
    deadline = time.monotonic() + 5
    while not path.exists() or len(path.read_text().split()) < 2:
        assert time.monotonic() < deadline, "the fake never started"
        await asyncio.sleep(0.02)
    return [int(p) for p in path.read_text().split()]


async def _all_gone(pids: list[int]) -> None:
    deadline = time.monotonic() + 2
    while not all(_gone(p) for p in pids):
        assert time.monotonic() < deadline, f"still alive: {pids}"
        await asyncio.sleep(0.02)


async def test_a_timeout_kills_the_group_and_leaves_no_image(
    tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids_file = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids_file))
    with pytest.raises(ImageGenError, match=r"timed out after 1 s"):
        await _generator(tmp_path, timeout=1).generate(REQUEST, out_dir)
    await _all_gone(await _pids(pids_file))


async def test_a_cancel_kills_the_group_and_propagates(
    tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids_file = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids_file))
    task = asyncio.create_task(_generator(tmp_path).generate(REQUEST, out_dir))
    pids = await _pids(pids_file)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 1
    await _all_gone(pids)


async def test_a_prompt_starting_with_a_dash_is_one_argv_item(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    request = ImageRequest("-sunset", 256, 256, 1, 1)
    await _generator(tmp_path).generate(request, out_dir)
    assert "--prompt=-sunset" in json.loads(dump.read_text())["argv"]


async def test_a_relative_model_folder_is_made_absolute_once(
    tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "m").mkdir(parents=True)
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    monkeypatch.chdir(tmp_path)
    generator = _generator(tmp_path, model="models/m")
    monkeypatch.chdir(out_dir)
    await generator.generate(REQUEST, out_dir)
    argv = json.loads(dump.read_text())["argv"]
    assert argv[argv.index("--model") + 1] == str((tmp_path / "models" / "m").resolve())
    assert generator.model == str((tmp_path / "models" / "m").resolve())


def test_the_tail_stays_within_the_cap_after_escaping() -> None:
    from src.engine.imagegen import _tail

    assert len(_tail(b"\x01" * 5000)) <= 2000
