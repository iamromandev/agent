"""The config watch in headless mode: the warning on stderr (or a `notice` event), a revoke that `--trust-project`
does not undo, and the global marker of an earlier run (one warning, no prompt)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from core.config.settings import get_settings
from core.engine.configguard import GlobalMarker, marker_path
from core.engine.provider.base import Done, TextDelta, ToolCall, ToolCallEvent
from core.testing.fakes import ScriptedProvider
from src.headless import EXIT_OK, HeadlessOptions
from src.headless.run import run_headless

MARKER_WARNING = (
    "global config was changed by a command in an earlier run; until you trust it in the app, global allow rules, "
    "hooks, MCP servers and language servers are off; global keepEnv is ignored (1 name); not loaded: 2 global commands, 1 global skill"
)
REWRITE = "echo '{}' > .agent/settings.json"


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    config = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("SHELL", "/bin/sh")
    monkeypatch.delenv("AGENT_SEARCH_URL", raising=False)
    get_settings.cache_clear()
    yield config
    get_settings.cache_clear()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    path = tmp_path / "ws"
    (path / ".agent").mkdir(parents=True)
    return path


def say(text: str) -> list:
    return [TextDelta(text), Done(10, 5)]


def call(tool: str, call_id: str = "c1", **args: object) -> list:
    return [ToolCallEvent(ToolCall(call_id, tool, dict(args))), Done(10, 5)]


def _results(provider: ScriptedProvider) -> list[str]:
    return [m.content for m in provider.requests[-1][0] if m.role == "tool"]


def test_a_command_that_rewrites_the_settings_warns_on_stderr_also_with_quiet(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ScriptedProvider([call("bash", command=REWRITE), say("ok")])

    code = run_headless(ws, HeadlessOptions("hi", allow=(f"bash({REWRITE})",), quiet=True), provider=provider)

    err = capsys.readouterr().err
    assert code == EXIT_OK
    lines = [line for line in err.splitlines() if "config changed" in line]
    assert len(lines) == 1
    assert lines[0].startswith("warning: config changed by bash (")
    assert ".agent/settings.json" in lines[0] and "project trust revoked" in lines[0]


def test_stream_json_reports_the_change_as_a_warn_notice(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([call("bash", command=REWRITE), say("ok")])

    run_headless(ws, HeadlessOptions("hi", allow=(f"bash({REWRITE})",), output="stream-json"), provider=provider)

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    notices = [e for e in events if e["type"] == "notice" and "config changed" in e["text"]]
    assert len(notices) == 1 and notices[0]["level"] == "warn"
    assert events[-1]["type"] == "result"


def test_json_output_keeps_its_one_object_and_the_change_is_a_stderr_warning(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ScriptedProvider([call("bash", command=REWRITE), say("ok")])

    code = run_headless(ws, HeadlessOptions("hi", allow=(f"bash({REWRITE})",), output="json"), provider=provider)

    captured = capsys.readouterr()
    [line] = captured.out.splitlines()  # one object, nothing else on stdout
    result = json.loads(line)
    assert code == EXIT_OK and result["type"] == "result" and result["result"] == "ok"
    assert result["exit_code"] == 0 and result["stop_reason"] == "done" and result["error"] is None
    assert "config changed" not in captured.out
    warnings = [x for x in captured.err.splitlines() if "config changed" in x]
    assert len(warnings) == 1 and warnings[0].startswith("warning: config changed by bash (")


def test_trust_project_does_not_bring_back_the_project_rules_after_a_revoke(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rewrite = "echo '{\"permissions\": {\"allow\": [\"bash(touch made.txt)\"]}}' > .agent/settings.json"
    (ws / ".agent" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": [f"bash({rewrite})", "bash(touch made.txt)"]}})
    )
    provider = ScriptedProvider(
        [call("bash", command=rewrite), call("bash", "c2", command="touch made.txt"), say("ok")]
    )

    code = run_headless(ws, HeadlessOptions("hi", trust_project=True), provider=provider)

    assert code == EXIT_OK
    assert _results(provider)[1].startswith("User denied")  # the project rule is off after the revoke
    assert not (ws / "made.txt").exists()
    assert "project trust revoked" in capsys.readouterr().err


def test_a_marker_from_an_earlier_run_gives_one_warning_and_holds_the_global_rules(
    ws: Path, env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (env / "agent").mkdir(parents=True)
    settings = {"permissions": {"allow": ["bash(touch made.txt)"]}, "guardrails": {"keepEnv": ["MY_TOKEN"]}}
    (env / "agent" / "settings.json").write_text(json.dumps(settings))
    (env / "agent" / "commands").mkdir()
    for name in ("a", "b"):
        (env / "agent" / "commands" / f"{name}.md").write_text("x")
    (env / "agent" / "skills" / "s").mkdir(parents=True)
    (env / "agent" / "skills" / "s" / "SKILL.md").write_text("---\ndescription: d\n---\nbody")
    GlobalMarker(marker_path()).write(["~/.config/agent/settings.json"])
    provider = ScriptedProvider([call("bash", command="touch made.txt"), say("ok")])

    code = run_headless(ws, HeadlessOptions("hi", quiet=True), provider=provider)

    err = capsys.readouterr().err
    assert code == EXIT_OK
    assert err.count(MARKER_WARNING) == 1
    assert "held until trusted" not in err  # the app's per-kind notices: headless has the one warning
    assert _results(provider)[0].startswith("User denied") and not (ws / "made.txt").exists()
    assert GlobalMarker(marker_path()).exists()  # only the app's dialog removes it


def test_without_a_marker_or_a_change_nothing_is_said(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run_headless(ws, HeadlessOptions("hi"), provider=ScriptedProvider([say("ok")]))

    err = capsys.readouterr().err
    assert code == EXIT_OK and "config" not in err and "global" not in err
