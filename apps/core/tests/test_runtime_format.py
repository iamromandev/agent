"""The runtime builds the format hub from the settings and gives it to the main agent only."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from core.config.settings import get_settings
from core.engine.approval import Choice, Decision
from core.runtime import Runtime, build_runtime
from core.testing.fakes import FakeCatalog, RecordingApprover, ScriptedProvider
from core.testing.format_fakes import fake_formatter
from pydantic import ValidationError


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_formatter(bin_dir, "gofmt", "upper")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return root.resolve()


def settings(value: object) -> None:
    folder = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(json.dumps({"format": value}))


def build(ws: Path) -> Runtime:
    return build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())


def test_without_the_key_there_is_no_hub(ws: Path) -> None:
    rt = build(ws)
    asyncio.run(rt.close())
    assert rt.format is None


def test_with_the_key_the_main_agent_gets_the_hub_and_bad_entries_warn(ws: Path) -> None:
    settings({"servers": {"mydsl": {"args": []}}})
    rt = build(ws)
    asyncio.run(rt.close())
    assert rt.format is not None and any(row.startswith("gofmt (global) found") for row in rt.format.listing())
    assert any("mydsl" in w for w in rt.warnings)
    agent = rt.agent_factory(RecordingApprover([Decision(Choice.ONCE)]))
    assert agent.format is rt.format


def test_the_timeout_setting_defaults_to_ten_seconds_and_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_settings().format_timeout == 10.0
    get_settings.cache_clear()
    monkeypatch.setenv("AGENT_FORMAT_TIMEOUT", "2.5")
    assert get_settings().format_timeout == 2.5
    get_settings.cache_clear()
    monkeypatch.setenv("AGENT_FORMAT_TIMEOUT", "0")
    with pytest.raises(ValidationError):
        get_settings()


def test_a_workspace_prettier_without_a_settings_file_is_untrusted_until_trusted(ws: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    settings(True)
    (ws / "node_modules" / ".bin").mkdir(parents=True)
    local = fake_formatter(ws / "node_modules" / ".bin", "prettier", "upper")
    (ws / ".prettierrc").write_text("{}")
    rt = build(ws)
    asyncio.run(rt.close())
    assert rt.format is not None and [s.name for s in rt.format.untrusted()] == ["prettier"]
    assert rt.permissions.rules.pending_trust
    rt.permissions.rules.trust_project()
    again = build(ws)
    asyncio.run(again.close())
    assert again.format is not None and again.format.untrusted() == []
    local.write_text(local.read_text() + "\n# changed\n")
    changed = build(ws)
    asyncio.run(changed.close())
    assert changed.format is not None and [s.name for s in changed.format.untrusted()] == ["prettier"]
