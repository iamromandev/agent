"""Images attached with `@path` in the app: vision at send time, notices, the session, /undo."""

from __future__ import annotations

import base64
from pathlib import Path

from src.engine.provider.base import Done, TextDelta
from src.engine.session import FileRecorder
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.transcript import NoticeEntry, UserEntry

from tests.fakes import FakeCatalog
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


async def test_a_hostile_image_name_is_escaped_in_every_notice(make_app: AppFactory, tmp_path: Path) -> None:
    name = "a\x1b[2J.png"
    (tmp_path / name).write_bytes(PNG)
    app = make_app([REPLY], context_window=20_000, catalog=FakeCatalog(["fake"]))
    async with app.run_test() as pilot:
        app.query_one(ChatInput).load_text(f'see @"{name}"')
        await pilot.press("enter")
        await settled(app)

        assert not any("\x1b" in n for n in _notices(app))
        assert any("a\\x1b[2J.png" in n for n in _notices(app))


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
