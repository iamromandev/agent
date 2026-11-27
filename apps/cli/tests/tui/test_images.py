"""Images attached with `@path` in the app: vision at send time, notices, the session, /undo."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from core.engine.provider.base import Done, TextDelta
from core.engine.session import FileRecorder
from core.testing.fakes import FakeCatalog
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.transcript import NoticeEntry, UserEntry

from tests.tui.conftest import AppFactory, scripted, settled, type_and_send, until

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 2048
REPLY = [TextDelta("ok"), Done(1, 1)]
VISION = FakeCatalog(["fake", "plain"], capabilities={"fake": ("completion", "vision")})


def _notices(app) -> list[str]:
    return [e.text[2:] for e in app.query(NoticeEntry)]  # without the level mark


def _user_messages(app):
    return [m for m in scripted(app).requests[-1][0] if m.role == "user"]


async def test_an_image_mention_reaches_the_model_and_is_announced(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(PNG)
    app = make_app([REPLY], context_window=20_000, catalog=FakeCatalog(["fake"], capabilities={"fake": ("vision",)}))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "see @a.png")
        await settled(app)

        (user,) = _user_messages(app)
        assert user.content == "see @a.png"
        assert [base64.b64encode(i.data) for i in user.images] == [base64.b64encode(PNG)]
        assert "attached a.png (PNG, 2 KB)" in _notices(app)
        assert [e.text for e in app.query(UserEntry)] == ["see @a.png"]


async def test_without_vision_the_text_goes_and_a_warning_replaces_the_image(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.png").write_bytes(PNG)
    app = make_app([REPLY], context_window=20_000, catalog=FakeCatalog(["fake"]))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "see @a.png")
        await settled(app)

        (user,) = _user_messages(app)
        assert user.content == "see @a.png"
        assert user.images == ()
        notices = _notices(app)
        assert "not attached a.png (model 'fake' has no vision)" in notices
        assert not any(n.startswith("attached a.png") for n in notices)


async def test_a_failing_capability_lookup_warns_once_and_the_turn_runs(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(PNG)
    app = make_app([REPLY, REPLY], context_window=20_000, catalog=FakeCatalog(["fake"], fail=True))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "see @a.png")
        await settled(app)
        await type_and_send(pilot, "again @a.png")
        await settled(app)

        assert len(scripted(app).requests) == 2
        assert all(m.images == () for r in scripted(app).requests for m in r[0])
        warnings = [n for n in _notices(app) if n.startswith("could not read the capabilities of 'fake'")]
        assert len(warnings) == 1


async def test_the_vision_of_the_model_at_send_time_decides(make_app: AppFactory, tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(PNG)
    app = make_app([REPLY, REPLY], context_window=20_000, catalog=VISION)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "one @a.png")
        await settled(app)
        assert len(_user_messages(app)[-1].images) == 1

        await type_and_send(pilot, "/model plain")
        await settled(app)
        await type_and_send(pilot, "two @a.png")
        await settled(app)

        assert _user_messages(app)[-1].images == ()
        assert "not attached a.png (model 'plain' has no vision)" in _notices(app)


async def test_the_session_file_holds_the_image_and_undo_puts_back_the_typed_text(
    make_app: AppFactory, tmp_path: Path
) -> None:
    (tmp_path / "a.png").write_bytes(PNG)
    app = make_app([REPLY], context_window=20_000, catalog=FakeCatalog(["fake"], capabilities={"fake": ("vision",)}))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "see @a.png")
        await settled(app)
        recorder = app.agent._recorder
        assert isinstance(recorder, FileRecorder) and recorder.path is not None
        assert base64.b64encode(PNG).decode() in recorder.path.read_text()

        await type_and_send(pilot, "/undo")
        await until(pilot, lambda: app.query_one(ChatInput).text == "see @a.png")
        assert app.agent.history == []


# pasted images (ctrl+v)


import stat  # noqa: E402

import pytest  # noqa: E402
from core.client import clipboard  # noqa: E402
from core.client.clipboard import ClipboardError, ClipboardUnsupported  # noqa: E402
from src.tui.widgets.attachment_panel import AttachmentPanel  # noqa: E402

VISION_CAT = FakeCatalog(["fake"], capabilities={"fake": ("vision",)})


@pytest.fixture
def clip(monkeypatch: pytest.MonkeyPatch):
    """The clipboard: `state["next"]` is an image (bytes), None (text only), or an exception to raise."""
    state: dict = {"next": PNG, "calls": 0}

    async def read(dest_dir: Path, timeout: float = 2.0):
        state["calls"] += 1
        number = state["calls"]
        await asyncio.sleep(state.get("delay", 0))
        item = state["next"]
        if isinstance(item, Exception):
            raise item
        if item is None:
            return None
        path = dest_dir / f"{number}.png"
        path.write_bytes(item)
        return path

    monkeypatch.setattr(clipboard, "read_clipboard_image", read)
    return state


def _chips(app) -> str:
    return app.query_one(AttachmentPanel).plain


async def _paste(pilot, app, count: int = 1) -> None:
    for i in range(count):
        await pilot.press("ctrl+v")
        await until(pilot, lambda want=i + 1: _chips(app).count("[image") >= want)


async def test_ctrl_v_adds_a_chip_and_sending_carries_the_image(make_app: AppFactory, clip, tmp_path: Path) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        assert _chips(app) == "[image 1: clipboard (PNG, 2 KB)]"
        folder = app.paste.path()
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        assert tmp_path.resolve() not in folder.resolve().parents

        await type_and_send(pilot, "hi")
        await settled(app)

        (user,) = _user_messages(app)
        assert user.content == "hi"
        assert [i.data for i in user.images] == [PNG]
        assert _chips(app) == ""
        assert "image clipboard:1 (PNG, 2 KB)" in _notices(app)
    assert not folder.exists()


async def test_pasted_images_follow_the_vision_of_the_model_at_send_time(
    make_app: AppFactory, clip, tmp_path: Path
) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION)
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        await type_and_send(pilot, "/model plain")
        await settled(app)
        await type_and_send(pilot, "hi")
        await settled(app)

        assert _user_messages(app)[-1].images == ()
        assert "not attached clipboard (model 'plain' has no vision)" in _notices(app)


async def test_mentions_and_pasted_images_share_the_cap_of_ten(make_app: AppFactory, clip, tmp_path: Path) -> None:
    for i in range(9):
        (tmp_path / f"{i}.png").write_bytes(PNG)
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app, 2)
        text = " ".join(f"@{i}.png" for i in range(9))
        app.query_one(ChatInput).load_text(text)
        await pilot.press("enter")
        await settled(app)

        assert len(_user_messages(app)[-1].images) == 10
        assert "not attached clipboard (image limit (10 per message))" in _notices(app)


async def test_esc_on_an_empty_input_clears_the_pending_images(make_app: AppFactory, clip) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app, 2)
        await pilot.press("escape")
        await until(pilot, lambda: _chips(app) == "")

        await type_and_send(pilot, "hi")
        await settled(app)
        assert _user_messages(app)[-1].images == ()


async def test_esc_keeps_the_images_while_something_is_typed(make_app: AppFactory, clip) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        await pilot.press("h")
        await pilot.press("escape")
        await settled(app)
        assert "[image 1" in _chips(app)


@pytest.mark.parametrize(
    ("error", "notice"),
    [
        (ClipboardUnsupported(), "clipboard images need macOS"),
        (ClipboardError("boom \x1b[2J"), "could not read the clipboard: boom \\x1b[2J"),
    ],
)
async def test_a_clipboard_failure_is_announced_and_the_text_paste_runs(
    make_app: AppFactory, clip, error: Exception, notice: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip["next"] = error
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        pasted: list[int] = []
        monkeypatch.setattr(ChatInput, "action_paste", lambda self: pasted.append(1))
        await pilot.press("ctrl+v")
        await until(pilot, lambda: notice in _notices(app) and bool(pasted))
        assert _chips(app) == ""


async def test_a_text_only_clipboard_pastes_text_and_adds_no_chip(
    make_app: AppFactory, clip, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip["next"] = None
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        pasted: list[int] = []
        monkeypatch.setattr(ChatInput, "action_paste", lambda self: pasted.append(1))
        await pilot.press("ctrl+v")
        await until(pilot, lambda: bool(pasted))
        assert _chips(app) == ""


async def test_a_pasted_image_over_the_limit_is_refused(make_app: AppFactory, clip) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app, 10)
        await pilot.press("ctrl+v")
        await until(pilot, lambda: "not attached clipboard (image limit (10 per message))" in _notices(app))
        assert _chips(app).count("[image") == 10


async def test_a_pasted_image_that_vanished_is_warned_about(make_app: AppFactory, clip) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        for f in app.paste.path().iterdir():
            f.unlink()
        await type_and_send(pilot, "hi")
        await settled(app)

        assert _user_messages(app)[-1].images == ()
        assert "not attached clipboard (file is gone)" in _notices(app)


async def test_undo_gives_back_the_text_and_no_chip(make_app: AppFactory, clip, tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(PNG)
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        await type_and_send(pilot, "see @a.png")
        await settled(app)
        assert len(_user_messages(app)[-1].images) == 2

        await type_and_send(pilot, "/undo")
        await until(pilot, lambda: app.query_one(ChatInput).text == "see @a.png")
        assert _chips(app) == ""


async def test_a_message_that_never_sends_keeps_the_chips(
    make_app: AppFactory, clip, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        monkeypatch.setattr(app.agent, "prompt_hooks", _block)
        await type_and_send(pilot, "hi")
        await settled(app)
        assert "[image 1" in _chips(app)
        assert app.query_one(ChatInput).text == "hi"


async def _block(text, emit):
    from core.engine.hooks.setup import HookDecision

    return HookDecision(True, "no", [], [])


async def test_an_os_error_while_pasting_is_announced_and_the_text_paste_runs(
    make_app: AppFactory, clip, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        pasted: list[int] = []
        monkeypatch.setattr(ChatInput, "action_paste", lambda self: pasted.append(1))

        def broken() -> Path:
            raise OSError("disk \x1b[2J full")

        monkeypatch.setattr(app.paste, "path", broken)
        await pilot.press("ctrl+v")
        await until(pilot, lambda: bool(pasted))
        assert any(n.startswith("could not read the clipboard: disk") and "\\x1b" in n for n in _notices(app))
        assert _chips(app) == ""


async def test_two_quick_pastes_at_nine_keep_the_cap_and_drop_the_extra_file(make_app: AppFactory, clip) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app, 9)
        clip["delay"] = 0.3
        await pilot.press("ctrl+v")
        await pilot.press("ctrl+v")
        await until(pilot, lambda: "not attached clipboard (image limit (10 per message))" in _notices(app))
        await settled(app)
        assert _chips(app).count("[image") == 10
        assert len(list(app.paste.path().iterdir())) == 10


async def test_images_come_back_when_the_turn_is_cancelled_before_the_agent_started(
    make_app: AppFactory, clip, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tui.widgets.transcript import Transcript

    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app)

        async def cancelled(self, *args, **kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr(Transcript, "add_user", cancelled)
        await type_and_send(pilot, "hi")
        await until(pilot, lambda: "[image 1" in _chips(app))
        await settled(app)
        assert "[image 1" in _chips(app)


async def test_a_pasted_file_that_was_replaced_is_not_attached(make_app: AppFactory, clip) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        (file,) = app.paste.path().iterdir()
        file.unlink()
        file.write_bytes(PNG + b"swapped")
        await type_and_send(pilot, "hi")
        await settled(app)

        assert _user_messages(app)[-1].images == ()
        assert "not attached clipboard (file changed)" in _notices(app)


@pytest.mark.parametrize("command", ["/new", "/clear"])
async def test_new_session_commands_clear_the_pending_images(make_app: AppFactory, clip, command: str) -> None:
    app = make_app([REPLY], context_window=20_000, catalog=VISION_CAT)
    async with app.run_test() as pilot:
        await _paste(pilot, app)
        await type_and_send(pilot, command)
        await until(pilot, lambda: _chips(app) == "")
