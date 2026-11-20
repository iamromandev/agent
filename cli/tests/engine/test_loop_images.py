"""Images on a user turn: `run_turn(images=...)` puts them on the user message, the provider gets them, the session keeps them."""

from __future__ import annotations

from src.engine.loop import Agent
from src.engine.provider.base import Done, Image, TextDelta
from src.engine.session import NullRecorder
from src.engine.tools.base import ToolContext

from tests.engine.test_loop import WINDOW, Events
from tests.fakes import RecordingApprover, ScriptedProvider

PNG = Image("png", b"\x89PNG\r\n\x1a\nxyz", "a.png")


class _Recorder(NullRecorder):
    def __init__(self) -> None:
        self.saved: list = []

    def message(self, message, meta) -> None:
        self.saved.append(message)


def _agent(tmp_path, provider, recorder=None) -> Agent:
    return Agent(provider, [], RecordingApprover(), ToolContext(tmp_path, 5.0), 5, recorder=recorder or NullRecorder())


async def test_run_turn_puts_the_images_on_the_user_message(tmp_path) -> None:
    provider = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=WINDOW)
    agent = _agent(tmp_path, provider)
    await agent.run_turn("see @a.png", Events(), images=(PNG,))
    sent = provider.requests[0][0]
    user = [m for m in sent if m.role == "user"][-1]
    assert user.content == "see @a.png"
    assert user.images == (PNG,)
    assert agent.history[0].images == (PNG,)


async def test_a_turn_without_images_has_none_on_any_message(tmp_path) -> None:
    provider = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=WINDOW)
    agent = _agent(tmp_path, provider)
    await agent.run_turn("hi", Events())
    assert all(m.images == () for m in provider.requests[0][0])


async def test_the_recorder_gets_the_images(tmp_path) -> None:
    provider = ScriptedProvider([[TextDelta("ok"), Done(1, 1)]], context_window=WINDOW)
    recorder = _Recorder()
    await _agent(tmp_path, provider, recorder).run_turn("see", Events(), images=(PNG,))
    assert recorder.saved[0].images == (PNG,)
