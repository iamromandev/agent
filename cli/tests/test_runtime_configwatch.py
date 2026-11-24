"""The runtime builds one config watch and guard and gives them to the agent, the hook runner and the rules."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from src.engine.approval import Choice, Decision
from src.engine.configwatch import Cause
from src.engine.provider.base import Done, ProviderEvent, TextDelta, ToolCall, ToolCallEvent
from src.runtime import build_runtime

from tests.fakes import FakeCatalog, Gate, RecordingApprover, ScriptedProvider


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / ".agent").mkdir(parents=True)
    return ws.resolve()


def test_the_test_environment_never_points_the_watch_at_the_real_home(tmp_path: Path) -> None:
    """The root conftest gives every test its own config and data folders; the watch reads exactly those."""
    ws = _workspace(tmp_path)
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    asyncio.run(rt.close())

    watched = {path for path, _, _ in rt.config_watch._watched.roots}
    real = Path.home().resolve()
    assert os.environ["XDG_CONFIG_HOME"] and os.environ["XDG_DATA_HOME"]
    assert all(Path(os.environ["XDG_CONFIG_HOME"]) in p.parents or Path(os.environ["XDG_DATA_HOME"]) in p.parents or ws in p.parents for p in watched)
    assert not any(real / ".config" in p.parents or real / ".local" in p.parents for p in watched)


def test_a_command_that_rewrites_the_project_settings_revokes_trust_through_the_runtime_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")
    ws = _workspace(tmp_path)
    (ws / ".agent" / "settings.json").write_text(json.dumps({"permissions": {"allow": ["bash(echo ok*)"]}}))
    call: list[ProviderEvent | Gate] = [ToolCallEvent(ToolCall("b1", "bash", {"command": "echo '{}' > .agent/settings.json"})), Done(1, 1)]
    provider = ScriptedProvider([call, [TextDelta("done"), Done(1, 1)]])
    rt = build_runtime(ws, set(), terminals=False, provider=provider, catalog=FakeCatalog(), trust_project_mcp=True)
    rt.permissions.rules.trust_for_run()
    agent = rt.agent_factory(RecordingApprover([Decision(Choice.ONCE)]))
    seen: list[str] = []
    rt.config_guard.listeners.append(seen.append)

    async def go() -> None:
        await agent.run_turn("go", lambda event: None)
        await rt.config_guard.settle()
        await rt.close()

    asyncio.run(go())

    assert len(seen) == 1 and ".agent/settings.json" in seen[0] and "project trust revoked" in seen[0]
    assert rt.permissions.rules.project_trusted is False


def test_closing_the_runtime_checks_the_watch_one_last_time(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    seen: list[str] = []
    rt.config_guard.listeners.append(seen.append)

    (ws / ".agent" / "mcp.json").write_text("{}")  # a process that outlived the last turn
    asyncio.run(rt.close())

    assert len(seen) == 1 and "a background process or terminal" in seen[0]


def test_a_change_made_while_the_runtime_is_built_is_reported_by_the_first_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline is taken before any loader reads the config, so a write during the build is not absorbed."""
    from src.engine.skills import loader

    ws = _workspace(tmp_path)
    real = loader.load_skills

    def load_and_meddle(*args: object, **kwargs: object) -> object:
        (ws / ".agent" / "mcp.json").write_text("{}")
        return real(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(loader, "load_skills", load_and_meddle)
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    assert rt.permissions.rules.revoked  # the check at the end of the build already reacted

    seen: list[str] = []
    rt.config_guard.listen(seen.append)  # the first listener hears what was said during the build
    rt.config_watch.check(Cause.background())
    asyncio.run(rt.close())

    assert len(seen) == 1 and ".agent/mcp.json" in seen[0]


def test_closing_checks_again_after_the_child_processes_are_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _workspace(tmp_path)
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    seen: list[str] = []
    rt.config_guard.listeners.append(seen.append)
    close_all = rt.mcp.close_all

    async def close_and_write() -> None:
        await close_all()
        (ws / ".agent" / "mcp.json").write_text("{}")  # a server that wrote on its way out

    monkeypatch.setattr(rt.mcp, "close_all", close_and_write)
    asyncio.run(rt.close())

    assert len(seen) == 1 and ".agent/mcp.json" in seen[0]


def test_kill_all_checks_after_killing(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    seen: list[str] = []
    rt.config_guard.listeners.append(seen.append)
    asyncio.run(rt.close())

    (ws / ".agent" / "settings.json").write_text("{}")
    rt.kill_all()

    assert len(seen) == 1 and ".agent/settings.json" in seen[0]


def _global_config(allow: list[str], servers: dict[str, object], hooks: dict[str, object] | None = None) -> Path:
    config = Path(os.environ["XDG_CONFIG_HOME"]) / "agent"
    config.mkdir(parents=True, exist_ok=True)
    body: dict[str, object] = {"permissions": {"allow": allow}}
    if hooks is not None:
        body["hooks"] = hooks
    (config / "settings.json").write_text(json.dumps(body))
    (config / "mcp.json").write_text(json.dumps({"servers": servers}))
    return config


def _set_marker() -> None:
    from src.engine.configguard import GlobalMarker, marker_path

    GlobalMarker(marker_path()).write(["~/.config/agent/settings.json"])


@pytest.mark.parametrize("trust_project", [False, True])
def test_a_marker_at_launch_holds_the_global_rules_hooks_and_servers(tmp_path: Path, trust_project: bool) -> None:
    _global_config(["bash(ls)"], {"g": {"command": "srv"}}, {"Stop": [{"command": "true"}]})
    _set_marker()
    ws = _workspace(tmp_path)

    rt = build_runtime(
        ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog(), trust_project_mcp=trust_project
    )
    asyncio.run(rt.close())

    assert rt.global_held is True
    assert rt.permissions.rules.global_active is False
    assert rt.permissions.rules.allow("bash", {"command": "ls"}, None) is None
    assert rt.hooks.has("Stop") is False
    assert [s.name for s in rt.mcp.untrusted("global")] == ["g"]  # --trust-project trusts only the project's


def _global_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """A global keepEnv, a deny rule and one global command, skill, agent and workflow."""
    from src.engine.workflows.loader import builtin_dir

    config = _global_config(["bash(ls)"], {})
    body = {"permissions": {"allow": ["bash(ls)"], "deny": ["bash(rm*)"]}, "guardrails": {"keepEnv": ["MY_TOKEN"]}}
    (config / "settings.json").write_text(json.dumps(body))
    (config / "commands").mkdir()
    (config / "commands" / "gc.md").write_text("do it")
    (config / "skills" / "gs").mkdir(parents=True)
    (config / "skills" / "gs" / "SKILL.md").write_text("---\ndescription: a skill\n---\nbody")
    (config / "agents").mkdir()
    (config / "agents" / "ga.md").write_text("---\ndescription: an agent\n---\nYou are it.")
    (config / "workflows").mkdir()
    (config / "workflows" / "gw.json").write_bytes((builtin_dir() / "reel.json").read_bytes())
    monkeypatch.setenv("MY_TOKEN", "s3cr3t-value-123")


def test_a_marker_at_launch_also_holds_keep_env_and_the_global_commands_skills_agents_and_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_extras(monkeypatch)
    _set_marker()
    ws = _workspace(tmp_path)

    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    asyncio.run(rt.close())

    assert rt.commands.get("gc") is None and rt.skills.get("gs") is None
    assert "ga" not in rt.agents.by_name and "gw" not in rt.workflows.items
    assert "MY_TOKEN" not in rt.guard.env.filter(os.environ)  # keepEnv is treated as empty
    assert rt.permissions.rules.deny("bash", {"command": "rm x"}, None) is not None  # deny only tightens
    held = rt.held
    assert (held.keep_env, held.commands, held.skills, held.agents, held.workflows) == (
        ("MY_TOKEN",), ("gc",), ("gs",), ("ga",), ("gw",)
    )
    notices = [n for n in rt.held.notices() if "held until trusted" in n]
    assert len(notices) == 5  # keepEnv and one per kind, each with its count
    assert any("1 global command not loaded" in n for n in notices)
    assert any("1 global workflow not loaded" in n for n in notices)
    assert all("next launch" in n for n in notices)


def test_without_a_marker_keep_env_and_the_global_files_load_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_extras(monkeypatch)
    ws = _workspace(tmp_path)

    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    asyncio.run(rt.close())

    assert rt.commands.get("gc") is not None and rt.skills.get("gs") is not None
    assert "ga" in rt.agents.by_name and "gw" in rt.workflows.items
    assert rt.guard.env.filter(os.environ).get("MY_TOKEN") == "s3cr3t-value-123"
    assert rt.held.notices() == [] and not rt.held.any


def test_without_a_marker_the_global_config_is_used_as_before(tmp_path: Path) -> None:
    _global_config(["bash(ls)"], {"g": {"command": "srv"}})
    ws = _workspace(tmp_path)

    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    asyncio.run(rt.close())

    assert rt.global_held is False and rt.permissions.rules.global_active
    assert rt.mcp.untrusted() == []


def test_the_runtimes_own_trust_decisions_are_not_changes(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / ".agent" / "settings.json").write_text(json.dumps({"permissions": {"allow": ["bash(ls)"]}}))
    rt = build_runtime(ws, set(), terminals=False, provider=ScriptedProvider([]), catalog=FakeCatalog())
    seen: list[str] = []
    rt.config_guard.listeners.append(seen.append)

    rt.permissions.rules.trust_project()  # writes permission-trust.json
    rt.config_watch.check(Cause.background())
    asyncio.run(rt.close())

    assert seen == [] and rt.permissions.rules.project_trusted
