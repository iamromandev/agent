"""Who gets generate_image: the app's main agent only, and only when mflux is found (AC1, AC2, AC3)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.config.settings import get_settings
from src.engine.imagegen import MfluxGenerator
from src.engine.tools.image import GenerateImageTool
from src.runtime import Runtime, build_runtime

from tests.fakes import FakeCatalog, ScriptedProvider
from tests.image_fakes import fake_mflux


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("AGENT_IMAGE_MODEL", "a/b")
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


def test_the_main_agent_gets_the_tool_when_mflux_is_found(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", lambda: command)
    (tool,) = _image_tools(_build(ws, images=True))
    generator: MfluxGenerator = tool.generator
    assert (generator.command, generator.model, generator.timeout) == (command, "a/b", 42.0)


def test_the_tool_is_absent_when_mflux_is_not_found(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.engine.imagegen.find_mflux", lambda: None)
    assert _image_tools(_build(ws, images=True)) == []


def test_the_tool_is_absent_unless_the_caller_asks_for_it(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", lambda: command)
    assert _image_tools(_build(ws)) == []


def test_the_task_child_never_gets_the_tool(ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = fake_mflux(tmp_path)
    monkeypatch.setattr("src.engine.imagegen.find_mflux", lambda: command)
    agent = _build(ws, images=True).agent_factory(_NoApprover())
    task = next(t for t in agent.tools if t.name == "task")
    assert not any(isinstance(t, GenerateImageTool) for t in task._tools)  # ty: ignore[unresolved-attribute]
