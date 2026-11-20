"""/agents, its launch warnings, and its place among the built-in commands."""

from __future__ import annotations

from pathlib import Path

from src.engine.agents.loader import AgentDef, Agents
from src.engine.commands.loader import load_commands
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, type_and_send, until


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _agent(
    name: str,
    description: str,
    source: str = "global",
    tools: tuple[str, ...] | None = None,
    model: str | None = None,
    resolved: str | None = None,
) -> AgentDef:
    return AgentDef(name, description, "body", tools, model, resolved, source, Path("/x") / f"{name}.md")  # ty: ignore[invalid-argument-type]


def _agents(*items: AgentDef, warnings: tuple[str, ...] = ()) -> Agents:
    return Agents({a.name: a for a in items}, warnings)


async def test_launch_warnings_about_agents_are_shown(make_app: AppFactory) -> None:
    app = make_app(permission_warnings=("a.md: no front matter",))
    async with app.run_test() as pilot:
        await until(pilot, lambda: any("a.md: no front matter" in n for n in _notices(app)))


async def test_agents_lists_source_model_description_tools_and_warnings(make_app: AppFactory) -> None:
    app = make_app(
        agents=_agents(
            _agent("reviewer", "Проверка ‮ кода", tools=("read", "grep")),
            _agent("tests", "Writes\ntests", "project", ("read", "write"), model="m:1", resolved="m:1"),
            _agent("quiet", "No tools", tools=()),
            _agent("plain", "Default tools"),
            warnings=("bad.md: no front matter",),
        )
    )
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/agents")
        await pilot.pause()
        (text,) = [n for n in _notices(app) if "(global)" in n]

    assert text[2:].splitlines() == [
        "plain (global): Default tools",
        "  tools: read, glob, grep, web_fetch, web_search, skill",
        "quiet (global): No tools",
        "  tools: none",
        "reviewer (global): Проверка \\u202e кода",
        "  tools: read, grep",
        "tests (project) model m:1: Writes tests",
        "  tools: read, write",
        "bad.md: no front matter",
    ]


async def test_agents_without_any_says_so(make_app: AppFactory) -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/agents")
        await until(pilot, lambda: "No agents." in "\n".join(_notices(app)))


def test_a_custom_command_named_agents_is_skipped_as_a_built_in(tmp_path: Path) -> None:
    from src.engine.commands.builtins import BUILTINS

    folder = tmp_path / ".agent" / "commands"
    folder.mkdir(parents=True)
    (folder / "agents.md").write_text("hi")
    commands = load_commands(tmp_path, BUILTINS, {"XDG_CONFIG_HOME": str(tmp_path / "none")})
    assert commands.get("agents") is None
    assert any("/agents is a built-in" in w for w in commands.warnings)
