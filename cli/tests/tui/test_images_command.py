"""/images in the app: status, pull, cancel, and what stays out of the way (AC5, AC6, AC8, AC10, AC13)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from src.engine.commands.builtins import BUILTINS
from src.engine.guardrails.setup import Guardrails
from src.engine.image_pull import NOT_INSTALLED, ImagePuller, Role
from src.engine.imagegen import EDIT_EXTRA, MfluxGenerator
from src.engine.provider.base import Done, TextDelta
from src.tui.app import HELP
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry

from tests.image_fakes import fake_mflux
from tests.tui.conftest import AppFactory, type_and_send, until


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _cache(tmp_path: Path, model: str, incomplete: bool = False) -> None:
    folder = tmp_path / "hub" / f"models--{model.replace('/', '--')}"
    (folder / "blobs").mkdir(parents=True)
    (folder / "snapshots" / "r").mkdir(parents=True)
    (folder / "blobs" / "b").write_bytes(b"x" * 1000)
    (folder / "snapshots" / "r" / "f").write_text("l")
    if incomplete:
        (folder / "blobs" / "z.incomplete").write_bytes(b"y")


def _puller(tmp_path: Path, edit: bool = True, gen: bool = True) -> ImagePuller:
    folder = tmp_path / "bin"
    folder.mkdir(exist_ok=True)
    guard = Guardrails.off(tmp_path)
    generators: dict[Role, MfluxGenerator | None] = {
        "generate": MfluxGenerator(fake_mflux(folder, "g"), "org/gen", None, guard) if gen else None,
        "edit": MfluxGenerator(fake_mflux(folder, "e"), "org/edit", None, guard, extra=EDIT_EXTRA) if edit else None,
    }
    return ImagePuller(generators, None, guard, env={"HF_HUB_CACHE": str(tmp_path / "hub")})


@pytest.fixture(autouse=True)
def _hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))


def test_images_is_a_builtin_with_a_help_entry_and_is_reserved() -> None:
    from evals.tasks import RESERVED

    assert "images" in BUILTINS
    assert "images" in RESERVED
    assert "/images" in HELP


async def test_status_lists_cached_and_missing(make_app: AppFactory, tmp_path: Path) -> None:
    _cache(tmp_path, "org/gen")
    app = make_app(images=_puller(tmp_path))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images")
        await until(pilot, lambda: any("org/gen" in n for n in _notices(app)))
        text = next(n for n in _notices(app) if "org/gen" in n)
    assert "generate: org/gen - cached, 0.0 GB" in text or "generate: org/gen - cached" in text
    assert "edit: org/edit - missing" in text


async def test_status_shows_partial_and_a_local_folder(make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _cache(tmp_path, "org/edit", incomplete=True)
    local = tmp_path / "weights"
    local.mkdir()
    guard = Guardrails.off(tmp_path)
    folder = tmp_path / "bin"
    folder.mkdir()
    puller = ImagePuller(
        {"generate": MfluxGenerator(fake_mflux(folder, "g"), str(local), None, guard), "edit": MfluxGenerator(fake_mflux(folder, "e"), "org/edit", None, guard)},
        None, guard, env={"HF_HUB_CACHE": str(tmp_path / "hub")},
    )
    app = make_app(images=puller)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images")
        await until(pilot, lambda: any("local folder" in n for n in _notices(app)))
        text = next(n for n in _notices(app) if "local folder" in n)
    assert "edit: org/edit - partial" in text


async def test_not_installed(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app(images=_puller(tmp_path, edit=False, gen=False))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images")
        await until(pilot, lambda: any(NOT_INSTALLED in n for n in _notices(app)))
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images pull")
        await until(pilot, lambda: any(NOT_INSTALLED in n for n in _notices(app)))


async def test_a_bad_argument_warns_and_starts_nothing(make_app: AppFactory, tmp_path: Path) -> None:
    puller = _puller(tmp_path)
    app = make_app(images=puller)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images pull foo")
        await until(pilot, lambda: any("usage: /images" in n for n in _notices(app)))
        assert not puller.busy
        assert any("usage: /images [pull [gen|edit] | cancel]" in n for n in _notices(app))


async def test_pull_runs_generate_then_edit(make_app: AppFactory, tmp_path: Path) -> None:
    _cache(tmp_path, "org/gen")
    _cache(tmp_path, "org/edit")
    app = make_app(images=_puller(tmp_path))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images pull")
        await until(pilot, lambda: any("edit model ready" in n for n in _notices(app)))
        ready = [n for n in _notices(app) if "model ready" in n]
        assert [n.split()[1] for n in ready] == ["generate", "edit"]
        assert app.query_one(StatusBar).background == ""


async def test_busy_is_true_as_soon_as_the_command_returns(make_app: AppFactory, tmp_path: Path) -> None:
    _cache(tmp_path, "org/gen")
    puller = _puller(tmp_path)
    app = make_app(images=puller)
    async with app.run_test() as pilot:
        await app._images_command("pull gen")
        assert puller.busy
        await until(pilot, lambda: any("generate model ready" in n for n in _notices(app)))


async def test_an_unexpected_pull_error_is_a_notice(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    puller = _puller(tmp_path)

    async def boom(*args: object) -> None:
        raise ValueError("kaput\x1b[2J")

    monkeypatch.setattr(puller, "pull", boom)
    app = make_app(images=puller)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images pull gen")
        await until(pilot, lambda: any("image pull failed: kaput" in n for n in _notices(app)))
        assert not any("\x1b" in n for n in _notices(app))


async def test_pull_gen_runs_only_generate(make_app: AppFactory, tmp_path: Path) -> None:
    _cache(tmp_path, "org/gen")
    app = make_app(images=_puller(tmp_path))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images pull gen")
        await until(pilot, lambda: any("generate model ready" in n for n in _notices(app)))
        await pilot.pause(0.2)
        assert not any("edit model" in n for n in _notices(app))


async def test_a_pull_runs_in_the_background_while_a_turn_works(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids = tmp_path / "pids"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids))
    _cache(tmp_path, "org/gen")
    puller = _puller(tmp_path)
    app = make_app([[TextDelta("hello"), Done(1, 1)]], images=puller)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images pull gen")
        await until(pilot, lambda: pids.exists() and len(pids.read_text().split()) == 2)
        status = app.query_one(StatusBar)
        await until(pilot, lambda: "◐ images: " in status.line)
        await type_and_send(pilot, "hi")
        await until(pilot, lambda: any(m.role == "assistant" and m.content == "hello" for m in app.agent.history))
        assert puller.busy
        await type_and_send(pilot, "/images pull")
        await until(pilot, lambda: any("an image pull is already running" in n for n in _notices(app)))
        await type_and_send(pilot, "/images")
        await until(pilot, lambda: any("pulling generate" in n for n in _notices(app)))
        await type_and_send(pilot, "/images cancel")
        await until(pilot, lambda: any("image pull cancelled" in n for n in _notices(app)))
        assert not puller.busy
        await until(pilot, lambda: app.query_one(StatusBar).background == "")


async def test_escape_leaves_the_pull_and_cancel_stops_it(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids = tmp_path / "pids"
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids))
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    _cache(tmp_path, "org/gen")
    puller = _puller(tmp_path)
    app = make_app(images=puller)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images pull gen")
        await until(pilot, lambda: pids.exists() and len(pids.read_text().split()) == 2)
        pid_list = [int(p) for p in pids.read_text().split()]
        await pilot.press("escape")
        await pilot.pause(0.3)
        assert puller.busy
        await type_and_send(pilot, "/images cancel")
        await until(pilot, lambda: not puller.busy)
        deadline = time.monotonic() + 2
        while not all(_gone(p) for p in pid_list):
            assert time.monotonic() < deadline
            await pilot.pause(0.02)
        assert not Path(json.loads(dump.read_text())["cwd"]).exists()
        await type_and_send(pilot, "/images cancel")
        await until(pilot, lambda: any("no image pull is running" in n for n in _notices(app)))
        assert app.query_one(ChatInput).text == ""


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def test_a_pull_failure_is_shown_and_nothing_is_saved_to_the_session(
    make_app: AppFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "fail")
    monkeypatch.setenv("FAKE_MFLUX_LINES", "1")
    monkeypatch.setenv("AGENT_SESSIONS_DIR", str(tmp_path / "sess"))
    _cache(tmp_path, "org/edit")
    app = make_app(images=_puller(tmp_path))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/images pull edit")
        await until(pilot, lambda: any("edit model pull failed: mflux failed (exit 1)" in n for n in _notices(app)))
        assert app.agent.history == []
    assert not list((tmp_path / "sess").rglob("*.jsonl"))
