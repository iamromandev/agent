"""Who gets generate_image: the app's main agent only, and only when mflux is found (AC1, AC2, AC3)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.config.settings import get_settings
from src.engine.imagegen import COMMAND, EDIT_COMMAND, MfluxGenerator
from src.engine.tools.image import EditImageTool, GenerateImageTool
from src.runtime import Runtime, build_runtime

from tests.fakes import FakeCatalog, ScriptedProvider
from tests.image_fakes import fake_mflux


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT_IMAGE_MODEL", "a/b")
    monkeypatch.setenv("AGENT_IMAGE_EDIT_MODEL", "e/m")
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


def _finder(generate: Path | None, edit: Path | str | None = "same"):
    """A `find_mflux` stand-in: one answer per command name."""
    answers = {COMMAND: generate, EDIT_COMMAND: generate if edit == "same" else edit}
    return lambda command=COMMAND: answers[command]


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
