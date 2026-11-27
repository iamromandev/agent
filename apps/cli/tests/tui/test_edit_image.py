"""edit_image in the app: pasted images become labels, the transcript is unchanged, /undo reverses the edit (AC4, AC8, AC11)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from core.client import clipboard
from core.engine.guardrails.setup import Guardrails
from core.engine.imagegen import EDIT_COMMAND, EDIT_EXTRA, MfluxGenerator, png_size
from core.engine.loop import PASTED_NOTE
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.testing.fakes import FakeCatalog
from core.testing.image_fakes import fake_mflux, png
from src.tui.widgets.attachment_panel import AttachmentPanel
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.transcript import Transcript, UserEntry

from tests.tui.conftest import AppFactory, choose, dialog, scripted, settled, type_and_send, until

PASTED = png(40, 30) + b"clip"
CALL = {"prompt": "pencil sketch", "image": "clipboard:1", "path": "sketch.png", "width": 256, "height": 256, "seed": 3}


@pytest.fixture
def clip(monkeypatch: pytest.MonkeyPatch) -> None:
    async def read(dest_dir: Path, timeout: float = 2.0) -> Path:
        path = dest_dir / "1.png"
        path.write_bytes(PASTED)
        return path

    monkeypatch.setattr(clipboard, "read_clipboard_image", read)


def _editor(tmp_path: Path, workspace: Path) -> MfluxGenerator:
    folder = tmp_path / "bin"
    folder.mkdir()
    return MfluxGenerator(fake_mflux(folder, EDIT_COMMAND), "x/y", 60, Guardrails.off(workspace), extra=EDIT_EXTRA)


def _script(**overrides: object) -> list:
    call = ToolCallEvent(ToolCall("c1", "edit_image", {**CALL, **overrides}))
    return [[call, Done(1, 1)], [TextDelta("ok"), Done(1, 1)]]


async def _paste(pilot, app) -> None:
    await pilot.press("ctrl+v")
    await until(pilot, lambda: "[image 1" in app.query_one(AttachmentPanel).plain)


async def test_a_pasted_image_is_labelled_for_the_model_and_the_transcript_shows_only_the_text(
    make_app: AppFactory, clip: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    app = make_app(
        _script(), workspace=workspace, editor=_editor(tmp_path, workspace), context_window=20_000, catalog=FakeCatalog(["fake"])
    )
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        await type_and_send(pilot, "sketch this")
        await dialog(pilot)
        await choose(pilot, "once")
        await settled(app)

        assert [e.text for e in app.query(UserEntry)] == ["sketch this"]
        sent = next(m for m in scripted(app).requests[0][0] if m.role == "user")
        assert sent.content == f"sketch this\n\n{PASTED_NOTE.format(labels='clipboard:1 (PNG, 40\u00d730)')}"
        assert sent.images == ()  # no vision: the labels exist all the same
        seen = json.loads(dump.read_text())
        assert seen["shas"][seen["inputs"][0]] == hashlib.sha256(PASTED).hexdigest()
        assert png_size((workspace / "sketch.png").read_bytes()) == (256, 256)
        entry = app.query_one(Transcript).tool("c1")
        assert entry is not None and "edit_image sketch.png" in str(entry.title)


async def test_undo_deletes_a_new_edit_and_restores_an_overwritten_one(
    make_app: AppFactory, clip: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "old.png").write_bytes(png(8, 8))
    script = [*_script(), *_script(path="old.png")]
    app = make_app(script, workspace=workspace, editor=_editor(tmp_path, workspace), context_window=20_000)
    async with app.run_test() as pilot:
        for text in ("one", "two"):
            await _paste(pilot, app)
            await type_and_send(pilot, text)
            await dialog(pilot)
            await choose(pilot, "once")
            await settled(app)
        assert png_size((workspace / "old.png").read_bytes()) == (256, 256)

        await type_and_send(pilot, "/undo")
        await until(pilot, lambda: app.query_one(ChatInput).text == "two")
        app.query_one(ChatInput).clear()
        assert (workspace / "old.png").read_bytes() == png(8, 8)
        assert (workspace / "sketch.png").exists()

        await type_and_send(pilot, "/undo")
        await until(pilot, lambda: app.query_one(ChatInput).text == "one")
        assert not (workspace / "sketch.png").exists()


REPLY = [TextDelta("ok"), Done(1, 1)]
NOTE = PASTED_NOTE.format(labels="clipboard:1 (PNG, 40\u00d730)")


def _user_contents(app) -> list[str]:
    return [m.content for m in app.agent.history if m.role == "user"]


async def test_a_merged_queue_turn_labels_the_images_that_turn_took(
    make_app: AppFactory, clip: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.testing.fakes import Gate

    from tests.tui.conftest import reached

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    gate = Gate()
    app = make_app([[gate], REPLY], workspace=workspace, editor=_editor(tmp_path, workspace), context_window=20_000)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await reached(gate)
        await _paste(pilot, app)
        await type_and_send(pilot, "second")
        gate.release.set()
        await settled(app)
        await pilot.pause()
        assert _user_contents(app) == ["first", f"second\n\n{NOTE}"]


async def test_a_blocked_prompt_gives_the_chips_back_and_no_label_leaks_into_the_resend(
    make_app: AppFactory, clip: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.engine.hooks.setup import HookDecision

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    app = make_app([REPLY], workspace=workspace, editor=_editor(tmp_path, workspace), context_window=20_000)
    real = app.agent.prompt_hooks

    async def block(text, emit):
        return HookDecision(True, "no", [], [])

    async with app.run_test() as pilot:
        await _paste(pilot, app)
        monkeypatch.setattr(app.agent, "prompt_hooks", block)
        await type_and_send(pilot, "hi")
        await settled(app)
        assert "[image 1" in app.query_one(AttachmentPanel).plain
        assert _user_contents(app) == []

        monkeypatch.setattr(app.agent, "prompt_hooks", real)
        await type_and_send(pilot, "")
        await settled(app)
        assert _user_contents(app) == [f"hi\n\n{NOTE}"]
