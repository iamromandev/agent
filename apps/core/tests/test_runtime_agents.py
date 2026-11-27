"""Launch-time resolution of the `model:` keys of custom agents (AC15, NFR3)."""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path

import pytest
from core.engine.provider.models import ModelInfo
from core.runtime import Runtime, build_runtime
from core.testing.fakes import FakeCatalog, ScriptedProvider


class CountingCatalog(FakeCatalog):
    def __init__(self, *args, list_delay: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.list_calls = 0
        self.list_delay = list_delay

    async def list(self) -> builtins.list[ModelInfo]:
        self.list_calls += 1
        if self.list_delay:
            await asyncio.sleep(self.list_delay)
        return await super().list()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    folder = tmp_path / "ws"
    (folder / ".agent" / "agents").mkdir(parents=True)
    return folder


def _agent(ws: Path, name: str, model: str | None) -> None:
    line = f"model: {model}\n" if model is not None else ""
    (ws / ".agent" / "agents" / f"{name}.md").write_text(f"---\ndescription: d\n{line}---\nbody\n")


def _build(ws: Path, catalog: CountingCatalog, provider: ScriptedProvider | None = None) -> Runtime:
    rt = build_runtime(ws, {"agents"}, terminals=False, provider=provider, catalog=catalog)
    asyncio.run(rt.close())
    return rt


def test_ac15_the_models_are_listed_once_and_each_key_is_resolved(ws: Path) -> None:
    _agent(ws, "a", "tagged")
    _agent(ws, "b", "missing")
    _agent(ws, "c", None)
    catalog = CountingCatalog(["tagged:latest"])
    rt = _build(ws, catalog)
    assert catalog.list_calls == 1
    assert rt.agents.by_name["a"].resolved_model == "tagged:latest"
    assert rt.agents.by_name["b"].resolved_model is None
    assert sum("'missing' is not installed" in w for w in rt.warnings) == 1
    assert rt.agents.by_name["b"].missing


def test_nfr3_no_list_call_when_no_agent_names_a_model(ws: Path) -> None:
    _agent(ws, "c", None)
    catalog = CountingCatalog(["x"])
    _build(ws, catalog)
    assert catalog.list_calls == 0


def test_an_injected_provider_means_no_catalog_call_and_names_as_written(ws: Path) -> None:
    _agent(ws, "a", "whatever")
    catalog = CountingCatalog(["x"])
    rt = _build(ws, catalog, ScriptedProvider([]))
    assert catalog.list_calls == 0
    assert rt.agents.by_name["a"].resolved_model == "whatever"


def test_an_unreachable_catalog_takes_the_names_as_written(ws: Path) -> None:
    _agent(ws, "a", "whatever")
    catalog = CountingCatalog(["x"], fail=True)
    rt = _build(ws, catalog)
    assert catalog.list_calls == 1
    assert rt.agents.by_name["a"].resolved_model == "whatever"
    assert not any("not installed" in w for w in rt.warnings)


def test_a_slow_catalog_is_given_up_on(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import core.engine.provider.models as models

    monkeypatch.setattr(models, "SHOW_TIMEOUT", 0.05)
    _agent(ws, "a", "whatever")
    catalog = CountingCatalog(["x"], list_delay=5.0)
    rt = _build(ws, catalog)
    assert rt.agents.by_name["a"].resolved_model == "whatever"


def test_one_list_call_when_the_project_model_key_decides_and_an_agent_names_a_model(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ws)  # no developer .env: an explicit AGENT_MODEL would beat the project key
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    (ws / ".agent" / "settings.json").write_text('{"model": "proj:1"}')
    _agent(ws, "a", "other")
    catalog = CountingCatalog(["proj:1", "other:latest"])
    settings_before = (ws / ".agent" / "settings.json").read_text()
    rt = _build(ws, catalog)
    assert catalog.list_calls == 1
    assert rt.provider.model == "proj:1"
    assert rt.agents.by_name["a"].resolved_model == "other:latest"
    assert (ws / ".agent" / "settings.json").read_text() == settings_before


def test_the_runtime_builds_one_vision_check_over_its_catalog_and_gives_it_to_the_agent(ws: Path) -> None:
    catalog = CountingCatalog(["v"], capabilities={"v": ("completion", "vision")})
    rt = _build(ws, catalog, ScriptedProvider([]))
    assert asyncio.run(rt.vision.has("v")) is True
    assert asyncio.run(rt.vision.has("other")) is False
    agent = rt.agent_factory(_NoApprover())
    assert agent.vision is rt.vision


class _NoApprover:
    async def ask(self, request):  # pragma: no cover
        raise AssertionError
