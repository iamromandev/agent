"""Who gets generate_image: the app's main agent only, and only when mflux is found (AC1, AC2, AC3)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.config.settings import get_settings
from src.engine.imagegen import COMMAND, EDIT_COMMAND, UPSCALE_COMMAND, MfluxGenerator
from src.engine.tools.image import EditImageTool, GenerateImageTool, UpscaleImageTool
from src.runtime import Runtime, build_runtime

from tests.fakes import FakeCatalog, ScriptedProvider
from tests.image_fakes import fake_mflux


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT_IMAGE_MODEL", "a/b")
    monkeypatch.setenv("AGENT_IMAGE_EDIT_MODEL", "e/m")
    monkeypatch.setenv("AGENT_IMAGE_UPSCALE_MODEL", "u/m")
    monkeypatch.setenv("AGENT_IMAGE_TIMEOUT", "42")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    folder = tmp_path / "ws"
    folder.mkdir()
    return folder


class _NoApprover:
    async def ask(self, request):  # pragma: no cover
        raise AssertionError


def _build(ws: Path, **kwargs: object) -> Runtime:
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog([]), **kwargs)  # ty: ignore[invalid-argument-type]
    asyncio.run(rt.close())
    return rt


def _image_tools(rt: Runtime) -> list[GenerateImageTool]:
    agent = rt.agent_factory(_NoApprover())
    return [t for t in agent.tools if isinstance(t, GenerateImageTool)]


def _edit_tools(rt: Runtime) -> list[EditImageTool]:
    agent = rt.agent_factory(_NoApprover())
    return [t for t in agent.tools if isinstance(t, EditImageTool)]


def _upscale_tools(rt: Runtime) -> list[UpscaleImageTool]:
    agent = rt.agent_factory(_NoApprover())
    return [t for t in agent.tools if isinstance(t, UpscaleImageTool)]


def _finder(generate: Path | None, edit: Path | str | None = "same", upscale: Path | None = None):
    """A `find_mflux` stand-in: one answer per command name."""
    answers = {COMMAND: generate, EDIT_COMMAND: generate if edit == "same" else edit, UPSCALE_COMMAND: upscale}
    return lambda command=COMMAND: answers.get(command)


def test_the_main_agent_gets_the_tool_when_mflux_is_found(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(command))
    (tool,) = _image_tools(_build(ws, images=True))
    generator: MfluxGenerator = tool.generator
    assert (generator.command, generator.model, generator.timeout) == (command, "a/b", 42.0)


def test_the_tool_is_absent_when_mflux_is_not_found(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(None))
    assert _image_tools(_build(ws, images=True)) == []


def test_the_tool_is_absent_unless_the_caller_asks_for_it(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(command))
    assert _image_tools(_build(ws)) == []


def test_the_task_child_never_gets_the_tool(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(command))
    agent = _build(ws, images=True).agent_factory(_NoApprover())
    task = next(t for t in agent.tools if t.name == "task")
    assert not any(isinstance(t, GenerateImageTool) for t in task._tools)  # ty: ignore[unresolved-attribute]


def test_the_main_agent_gets_edit_image_with_its_own_model_and_arguments(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path, EDIT_COMMAND)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(None, command))
    rt = _build(ws, images=True)
    assert _image_tools(rt) == []
    (tool,) = _edit_tools(rt)
    generator: MfluxGenerator = tool.generator
    assert (generator.command, generator.model, generator.timeout) == (command, "e/m", 42.0)
    assert generator.extra == ("--quantize", "4")


def test_an_install_with_only_the_z_image_command_still_gets_generate_image(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(command, None))
    rt = _build(ws, images=True)
    assert len(_image_tools(rt)) == 1 and _edit_tools(rt) == []


def test_both_tools_are_absent_without_images_or_without_the_extra(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(fake_mflux(tmp_path)))
    assert _edit_tools(_build(ws)) == []
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(None))
    rt = _build(ws, images=True)
    assert _edit_tools(rt) == [] and _image_tools(rt) == []


def test_the_task_child_never_gets_edit_image(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(fake_mflux(tmp_path)))
    agent = _build(ws, images=True).agent_factory(_NoApprover())
    task = next(t for t in agent.tools if t.name == "task")
    assert not any(isinstance(t, EditImageTool) for t in task._tools)  # ty: ignore[unresolved-attribute]


def test_the_runtime_has_one_puller_shared_with_both_tools(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(fake_mflux(tmp_path)))
    rt = _build(ws, images=True)
    assert rt.images is not None and rt.images.installed
    assert rt.images.hf is None  # the finder knows no `hf`
    (generate,) = _image_tools(rt)
    (edit,) = _edit_tools(rt)
    assert generate.pull is rt.images and edit.pull is rt.images


def test_the_runtime_has_no_puller_without_images(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(fake_mflux(tmp_path)))
    assert _build(ws).images is None


def test_close_cancels_a_running_pull_and_kill_all_reaches_the_puller(
    ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(fake_mflux(tmp_path)))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("FAKE_MFLUX_MODE", "sleep")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(tmp_path / "pids"))
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog([]), images=True)
    assert rt.images is not None
    puller = rt.images
    notes: list[tuple[str, str]] = []

    async def emit(level: str, text: str) -> None:
        notes.append((level, text))

    async def scenario() -> None:
        task = asyncio.create_task(puller.pull(("generate",), emit, lambda text: None))
        for _ in range(100):
            if (tmp_path / "pids").exists() and (tmp_path / "pids").read_text().count(" "):
                break
            await asyncio.sleep(0.05)
        assert puller.busy
        await rt.close()
        assert not puller.busy
        assert task.cancelled() or task.done()

    asyncio.run(scenario())
    assert notes[-1] == ("warn", "image pull cancelled")
    called: list[bool] = []
    monkeypatch.setattr(puller, "kill_all", lambda: called.append(True))
    rt.kill_all()
    assert called == [True]


def test_the_main_agent_gets_upscale_image_with_its_own_model_and_arguments(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path, UPSCALE_COMMAND)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(None, None, command))
    rt = _build(ws, images=True)
    assert _image_tools(rt) == [] and _edit_tools(rt) == []
    (tool,) = _upscale_tools(rt)
    generator: MfluxGenerator = tool.generator
    assert (generator.command, generator.model, generator.timeout) == (command, "u/m", 42.0)
    assert generator.extra == ("--low-ram", "--quantize", "8") and generator.kind == "upscale"
    assert tool.pull is rt.images and rt.images is not None
    assert [row.state for row in rt.images.status()][2] != "unavailable"


def test_upscale_image_is_absent_without_its_command_without_images_and_for_the_task_child(
    ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = fake_mflux(tmp_path)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(command))
    rt = _build(ws, images=True)
    assert _upscale_tools(rt) == []
    assert rt.images is not None and rt.images.status()[2].state == "unavailable"
    monkeypatch.setattr("src.engine.imagegen.find_mflux", _finder(command, command, command))
    assert _upscale_tools(_build(ws)) == []
    agent = _build(ws, images=True).agent_factory(_NoApprover())
    task = next(t for t in agent.tools if t.name == "task")
    assert not any(isinstance(t, UpscaleImageTool) for t in task._tools)  # ty: ignore[unresolved-attribute]


def _speak_tools(rt: Runtime) -> list:
    from src.engine.tools.media import SpeakTool

    agent = rt.agent_factory(_NoApprover())
    return [t for t in agent.tools if isinstance(t, SpeakTool)]


def test_the_main_agent_gets_speak_with_the_settings_when_the_engine_is_found(
    ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_SPEECH_MODEL", "s/m")
    monkeypatch.setenv("AGENT_SPEECH_TIMEOUT", "17")
    get_settings.cache_clear()
    monkeypatch.setattr("src.engine.media.find_speech", lambda: tmp_path / "helper.py")
    (tool,) = _speak_tools(_build(ws, images=True))
    assert (tool.model, tool.timeout) == ("s/m", 17.0)


def test_speak_is_absent_without_the_engine_without_images_and_for_the_task_child(
    ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.engine.media.find_speech", lambda: None)
    assert _speak_tools(_build(ws, images=True)) == []
    monkeypatch.setattr("src.engine.media.find_speech", lambda: tmp_path / "helper.py")
    assert _speak_tools(_build(ws)) == []
    agent = _build(ws, images=True).agent_factory(_NoApprover())
    task = next(t for t in agent.tools if t.name == "task")
    assert not any(t.name == "speak" for t in task._tools)  # ty: ignore[unresolved-attribute]
