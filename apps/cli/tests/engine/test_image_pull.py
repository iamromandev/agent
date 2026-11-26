"""ImagePuller against a fake mflux and a fake hf (AC3, AC4, AC8, AC9, AC11, AC12, AC14)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from src.engine.guardrails.setup import Guardrails
from src.engine.image_pull import ImagePuller, Role, free_space
from src.engine.imagegen import EDIT_EXTRA, MfluxGenerator

from tests.image_fakes import dry_run, fake_hf, fake_mflux

HUB = "hub"


class Notes:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self.progress: list[str] = []

    async def emit(self, level: str, text: str) -> None:
        self.items.append((level, text))

    def texts(self) -> list[str]:
        return [text for _, text in self.items]


def _cache(tmp_path: Path, model: str) -> None:
    folder = tmp_path / HUB / f"models--{model.replace('/', '--')}"
    (folder / "blobs").mkdir(parents=True)
    (folder / "snapshots" / "r").mkdir(parents=True)
    (folder / "blobs" / "b").write_bytes(b"x" * 1000)
    (folder / "snapshots" / "r" / "f").write_text("l")


def _puller(
    tmp_path: Path,
    guard: Guardrails | None = None,
    hf: Path | None = None,
    disk: Callable[[Path], tuple[int, Path]] | None = None,
    gen: bool = True,
    edit: bool = True,
) -> ImagePuller:
    guard = guard or Guardrails.off(tmp_path)
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    generators: dict[Role, MfluxGenerator | None] = {
        "generate": MfluxGenerator(fake_mflux(folder, "gen"), "org/gen", 5, guard) if gen else None,
        "edit": MfluxGenerator(fake_mflux(folder, "edit"), "org/edit", 5, guard, extra=EDIT_EXTRA) if edit else None,
    }
    return ImagePuller(generators, hf, guard, env={"HF_HUB_CACHE": str(tmp_path / HUB)}, disk=disk or free_space)


def _ignore(text: str) -> None:
    pass


async def _pull(puller: ImagePuller, roles: tuple[Role, ...], notes: Notes) -> None:
    await puller.pull(roles, notes.emit, _ignore)


async def test_generate_runs_the_warmup_with_the_models_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    _cache(tmp_path, "org/gen")
    notes = Notes()
    await _pull(_puller(tmp_path), ("generate",), notes)
    data = json.loads(dump.read_text())
    argv = data["argv"]
    assert argv[argv.index("--model") + 1] == "org/gen"
    assert argv[argv.index("--width") + 1] == "256"
    assert argv[argv.index("--height") + 1] == "256"
    assert argv[argv.index("--steps") + 1] == "1"
    assert not Path(data["cwd"]).exists()


async def test_edit_gets_one_staged_grey_png_and_the_extra_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    _cache(tmp_path, "org/edit")
    await _pull(_puller(tmp_path), ("edit",), Notes())
    data = json.loads(dump.read_text())
    assert len(data["inputs"]) == 1
    assert data["inputs"][0].endswith("in-0.png")
    assert data["argv"][data["argv"].index("--quantize") + 1] == "4"
    assert data["shas"][data["inputs"][0]] is not None


async def test_both_roles_run_in_order(tmp_path: Path) -> None:
    _cache(tmp_path, "org/gen")
    _cache(tmp_path, "org/edit")
    notes = Notes()
    await _pull(_puller(tmp_path), ("generate", "edit"), notes)
    ready = [t for t in notes.texts() if "ready" in t]
    assert [t.split()[0] for t in ready] == ["generate", "edit"]


async def test_success_gives_one_ready_notice(tmp_path: Path) -> None:
    _cache(tmp_path, "org/gen")
    notes = Notes()
    await _pull(_puller(tmp_path), ("generate",), notes)
    ready = [t for t in notes.texts() if "model ready" in t]
    assert len(ready) == 1
    assert ready[0].startswith("generate model ready (org/gen, ")
    assert ready[0].endswith("s)")


async def test_a_failure_is_one_pull_failed_notice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "fail")
    monkeypatch.setenv("FAKE_MFLUX_LINES", "1")
    _cache(tmp_path, "org/edit")
    notes = Notes()
    await _pull(_puller(tmp_path), ("edit",), notes)
    failed = [t for t in notes.texts() if "failed" in t]
    assert failed == ["edit model pull failed: mflux failed (exit 1): error line 0 \\x1b[2J"]
    assert not [t for t in notes.texts() if "ready" in t]


async def test_missing_command_is_a_failure_notice(tmp_path: Path) -> None:
    notes = Notes()
    await _pull(_puller(tmp_path, edit=False), ("edit",), notes)
    assert notes.texts() == ["edit model pull failed: mflux command not found"]


async def test_cached_model_still_checks_the_remaining_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "hfdump.json"
    monkeypatch.setenv("FAKE_HF_DUMP", str(dump))
    _cache(tmp_path, "org/gen")
    hf = fake_hf(tmp_path, dry_run([("transformer/a", "15.0G")]))
    notes = Notes()
    await _pull(_puller(tmp_path, hf=hf, disk=lambda f: (10_000_000_000, f)), ("generate",), notes)
    assert dump.exists()
    assert any("not enough disk space for org/gen" in t for t in notes.texts())
    assert not any("generate model ready" in t for t in notes.texts())


async def test_nothing_left_to_download_never_refuses(tmp_path: Path) -> None:
    _cache(tmp_path, "org/gen")
    hf = fake_hf(tmp_path, dry_run([("transformer/a", "-")]))
    notes = Notes()
    await _pull(_puller(tmp_path, hf=hf, disk=lambda f: (1, f)), ("generate",), notes)
    assert "pulling generate model org/gen" in notes.texts()
    assert any("generate model ready" in t for t in notes.texts())


async def test_an_unexpected_size_error_pulls_anyway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*args: object) -> int:
        raise ValueError("odd")

    monkeypatch.setattr("src.engine.image_pull.download_size", boom)
    notes = Notes()
    await _pull(_puller(tmp_path, hf=fake_hf(tmp_path, "")), ("generate",), notes)
    assert any(t.startswith("could not read the download size of org/gen (") and t.endswith("pulling anyway") for t in notes.texts())
    assert any("generate model ready" in t for t in notes.texts())


async def test_the_hf_dry_run_happens_before_the_warmup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hf = fake_hf(tmp_path, dry_run([("transformer/a", "1.0G")]))
    monkeypatch.setenv("FAKE_HF_DUMP", str(tmp_path / "h.json"))
    puller = _puller(tmp_path, hf=hf, disk=lambda f: (50_000_000_000, f))
    notes = Notes()
    await _pull(puller, ("generate",), notes)
    assert (tmp_path / "h.json").exists()
    assert any("generate model ready" in t for t in notes.texts())


async def test_cancel_during_the_dry_run_leaves_no_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "h.json"
    monkeypatch.setenv("FAKE_HF_DUMP", str(dump))
    notes = Notes()
    puller = _puller(tmp_path, hf=fake_hf(tmp_path, "", sleep=30))
    task = asyncio.create_task(_pull(puller, ("generate",), notes))
    for _ in range(100):
        if dump.exists() and dump.read_text():
            break
        await asyncio.sleep(0.05)
    pid = json.loads(dump.read_text())["pid"]
    assert await puller.cancel() is True
    with pytest.raises(asyncio.CancelledError):
        await task
    await _all_gone([pid])
    assert notes.texts()[-1] == "image pull cancelled"


async def test_kill_all_during_the_dry_run_kills_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "h.json"
    monkeypatch.setenv("FAKE_HF_DUMP", str(dump))
    puller = _puller(tmp_path, hf=fake_hf(tmp_path, "", sleep=30))
    task = asyncio.create_task(_pull(puller, ("generate",), Notes()))
    for _ in range(100):
        if dump.exists() and dump.read_text():
            break
        await asyncio.sleep(0.05)
    pid = json.loads(dump.read_text())["pid"]
    puller.kill_all()
    await _all_gone([pid])
    task.cancel()
    await asyncio.wait({task})


async def test_reserve_makes_busy_at_once_and_pull_takes_it(tmp_path: Path) -> None:
    _cache(tmp_path, "org/gen")
    puller = _puller(tmp_path)
    puller.reserve()
    assert puller.busy
    with pytest.raises(RuntimeError, match="already running"):
        puller.reserve()
    await _pull(puller, ("generate",), Notes())
    assert not puller.busy


def test_release_frees_an_unused_reservation(tmp_path: Path) -> None:
    puller = _puller(tmp_path)
    puller.reserve()
    puller.release()
    assert not puller.busy


async def test_a_local_folder_never_runs_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dump = tmp_path / "hfdump.json"
    monkeypatch.setenv("FAKE_HF_DUMP", str(dump))
    local = tmp_path / "weights"
    local.mkdir()
    guard = Guardrails.off(tmp_path)
    generators: dict[Role, MfluxGenerator | None] = {"generate": MfluxGenerator(fake_mflux(tmp_path), str(local), None, guard), "edit": None}
    puller = ImagePuller(generators, fake_hf(tmp_path, ""), guard, env={"HF_HUB_CACHE": str(tmp_path / HUB)})
    await _pull(puller, ("generate",), Notes())
    assert not dump.exists()


async def test_not_enough_disk_refuses_that_model_only(tmp_path: Path) -> None:
    hf = fake_hf(tmp_path, dry_run([("transformer/a", "15.0G")]))
    notes = Notes()
    frees = iter([10_000_000_000, 50_000_000_000])
    puller = _puller(tmp_path, hf=hf, disk=lambda folder: (next(frees), folder))
    await _pull(puller, ("generate", "edit"), notes)
    folder = tmp_path / HUB
    assert f"not enough disk space for org/gen: needs 15.0 GB + 2 GB, 10.0 GB free in {folder}" in notes.texts()
    assert any("edit model ready" in t for t in notes.texts())
    assert not any("generate model ready" in t for t in notes.texts())


async def test_enough_disk_shows_the_size_and_runs(tmp_path: Path) -> None:
    hf = fake_hf(tmp_path, dry_run([("transformer/a", "1.0G")]))
    notes = Notes()
    await _pull(_puller(tmp_path, hf=hf, disk=lambda folder: (50_000_000_000, folder)), ("generate",), notes)
    assert "pulling generate model org/gen (download size 1.0 GB, 50.0 GB free)" in notes.texts()


async def test_a_failing_hf_warns_and_pulls_anyway(tmp_path: Path) -> None:
    notes = Notes()
    await _pull(_puller(tmp_path, hf=fake_hf(tmp_path, "", code=1)), ("generate",), notes)
    warn = [t for t in notes.texts() if t.startswith("could not read the download size of org/gen")]
    assert warn == ["could not read the download size of org/gen (hf exited 1); pulling anyway"]
    assert any("generate model ready" in t for t in notes.texts())


async def test_no_hf_warns_and_pulls_anyway(tmp_path: Path) -> None:
    notes = Notes()
    await _pull(_puller(tmp_path, hf=None), ("generate",), notes)
    assert any("(hf not found); pulling anyway" in t for t in notes.texts())
    assert any("generate model ready" in t for t in notes.texts())


async def test_tokens_stay_out_of_both_children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(tmp_path / "m.json"))
    monkeypatch.setenv("FAKE_HF_DUMP", str(tmp_path / "h.json"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.txt").write_text("a")
    guard = Guardrails.off(tmp_path)
    puller = _puller(tmp_path, guard, hf=fake_hf(tmp_path, dry_run([("transformer/a", "1.0")])),
                     disk=lambda folder: (50_000_000_000, folder))
    await _pull(puller, ("generate",), Notes())
    assert "HF_TOKEN" not in json.loads((tmp_path / "m.json").read_text())["env"]
    assert "HF_TOKEN" not in json.loads((tmp_path / "h.json").read_text())["env"]
    assert sorted(p.name for p in workspace.iterdir()) == ["a.txt"]


async def _pids(path: Path) -> list[int]:
    for _ in range(100):
        if path.exists() and path.read_text().count(" "):
            return [int(p) for p in path.read_text().split()]
        await asyncio.sleep(0.05)
    raise AssertionError("fake mflux never started")


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def _all_gone(pids: list[int]) -> None:
    for _ in range(100):
        if all(_gone(p) for p in pids):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("process still alive")


async def test_cancel_kills_the_group_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pids_file = tmp_path / "pids"
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids_file))
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    _cache(tmp_path, "org/gen")
    notes = Notes()
    puller = _puller(tmp_path)
    task = asyncio.create_task(_pull(puller, ("generate",), notes))
    pids = await _pids(pids_file)
    assert puller.busy
    assert puller.pulling is not None
    assert await puller.cancel() is True
    with pytest.raises(asyncio.CancelledError):
        await task
    await _all_gone(pids)
    assert notes.texts()[-1] == "image pull cancelled"
    assert not puller.busy
    assert not Path(json.loads(dump.read_text())["cwd"]).exists()
    assert await puller.cancel() is False


async def test_a_second_pull_while_one_runs_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pids_file = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids_file))
    _cache(tmp_path, "org/gen")
    puller = _puller(tmp_path)
    task = asyncio.create_task(_pull(puller, ("generate",), Notes()))
    await _pids(pids_file)
    with pytest.raises(RuntimeError, match="already running"):
        await _pull(puller, ("generate",), Notes())
    await puller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_kill_all_stops_the_process_synchronously(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pids_file = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids_file))
    _cache(tmp_path, "org/gen")
    puller = _puller(tmp_path)
    task = asyncio.create_task(_pull(puller, ("generate",), Notes()))
    pids = await _pids(pids_file)
    puller.kill_all()
    await _all_gone(pids)
    await asyncio.wait({task})
    assert not puller.busy


def test_kill_all_with_nothing_running_is_a_no_op(tmp_path: Path) -> None:
    _puller(tmp_path).kill_all()
