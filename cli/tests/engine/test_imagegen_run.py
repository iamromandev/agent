"""MfluxGenerator against a fake mflux: argv, environment, failures, timeout and cancel (AC10, AC12, AC14)."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
from src.engine.guardrails.setup import Guardrails
from src.engine.imagegen import ImageGenError, ImageProgress, ImageRequest, MfluxGenerator, png_size

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


async def test_inputs_and_extra_arguments_go_into_the_argv(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    a, b = out_dir / "in-0.png", out_dir / "in-1.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    request = ImageRequest("p", 512, 256, 4, 9, inputs=(a, b))
    generator = MfluxGenerator(fake_mflux(tmp_path), "m", 20, Guardrails.off(tmp_path), extra=("--quantize", "4"))
    await generator.generate(request, out_dir)
    assert json.loads(dump.read_text())["argv"] == [
        "--model", "m",
        "--prompt=p",
        "--image-paths", str(a), str(b),
        "--width", "512",
        "--height", "256",
        "--steps", "4",
        "--seed", "9",
        "--quantize", "4",
        "--output", str(out_dir / "image.png"),
        "--no-exif",
    ]  # fmt: skip


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


async def _progress(tmp_path: Path, out_dir: Path, mode: str, monkeypatch: pytest.MonkeyPatch, request: ImageRequest | None = None):
    monkeypatch.setenv("FAKE_MFLUX_MODE", mode)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    seen: list[ImageProgress] = []
    await _generator(tmp_path).generate(request or ImageRequest("p", 256, 256, 9, 1), out_dir, on_progress=seen.append)
    return seen


async def test_progress_values_arrive_in_order(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = await _progress(tmp_path, out_dir, "progress", monkeypatch)
    assert seen[0].phase == "downloading"  # the fake model is not cached
    steps = [p.step for p in seen if p.phase == "steps"]
    assert steps == list(range(1, 10))
    assert seen[-1] == ImageProgress("saving")


async def test_cached_model_starts_with_loading(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snap = tmp_path / "hub" / "models--x--y" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "f").write_text("x")
    seen = await _progress(tmp_path, out_dir, "progress", monkeypatch)
    assert seen[0] == ImageProgress("loading")


async def test_download_output_gives_a_percentage(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = await _progress(tmp_path, out_dir, "download", monkeypatch)
    first_step = next(i for i, p in enumerate(seen) if p.phase == "steps")
    assert [p.percent for p in seen[:first_step] if p.phase == "downloading"][1:] == [10, 42, 100]


async def test_bad_lines_give_no_wrong_step(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = await _progress(tmp_path, out_dir, "badlines", monkeypatch)
    steps = [p.step for p in seen if p.phase == "steps"]
    assert steps == sorted(steps) and set(steps) <= set(range(1, 10))
    assert seen[-1] == ImageProgress("saving")


async def test_failure_text_is_unchanged_by_progress(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progressfail")
    with pytest.raises(ImageGenError) as info:
        await _generator(tmp_path).generate(ImageRequest("p", 256, 256, 9, 1), out_dir, on_progress=lambda p: None)
    text = str(info.value)
    assert text.startswith("mflux failed (exit 1): ")
    assert "ValueError: real reason" in text and "it/s" not in text and "%|" not in text


async def test_a_raising_callback_changes_nothing(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "progress")

    def boom(p: ImageProgress) -> None:
        raise RuntimeError("no")

    path = await _generator(tmp_path).generate(ImageRequest("p", 256, 256, 9, 1), out_dir, on_progress=boom)
    assert png_size(path.read_bytes()) == (256, 256)


async def test_a_huge_digit_line_does_not_break_the_call(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = await _progress(tmp_path, out_dir, "hugedigits", monkeypatch)
    assert [p.step for p in seen if p.phase == "steps"] == list(range(1, 10))


async def test_phase_is_loading_between_the_download_and_the_first_step(
    tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = await _progress(tmp_path, out_dir, "download", monkeypatch)
    first_step = next(i for i, p in enumerate(seen) if p.phase == "steps")
    assert ImageProgress("loading") in seen[1:first_step]


async def test_no_timeout_runs_to_the_end(tmp_path: Path, out_dir: Path) -> None:
    generator = MfluxGenerator(fake_mflux(tmp_path), "x/y", None, Guardrails.off(tmp_path))
    path = await generator.generate(REQUEST, out_dir)
    assert png_size(path.read_bytes()) == (512, 256)


async def test_on_start_gets_the_pid_of_a_group_leader(tmp_path: Path, out_dir: Path) -> None:
    pids: list[int] = []
    await _generator(tmp_path).generate(REQUEST, out_dir, on_start=pids.append)
    assert len(pids) == 1
    assert pids[0] > 0


async def test_run_with_ticker_sends_progress_text(tmp_path: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.engine.imagegen import run_with_ticker

    monkeypatch.setenv("FAKE_MFLUX_MODE", "progress")
    texts: list[str] = []
    path = await run_with_ticker(_generator(tmp_path), ImageRequest("p", 256, 256, 2, 1), out_dir, texts.append)
    assert path.is_file()
    assert texts
    assert texts[0].startswith("starting")
