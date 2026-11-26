"""The "Available agents" prompt section."""

from __future__ import annotations

from pathlib import Path

from src.engine.agents.loader import SECTION_CHARS, AgentDef, Agents


def agent(name: str, description: str) -> AgentDef:
    return AgentDef(name, description, "body", None, None, None, "global", Path("/x") / f"{name}.md")


def agents(*items: AgentDef) -> Agents:
    return Agents({a.name: a for a in items}, ())


def test_no_agents_gives_no_section() -> None:
    assert agents().section() is None
    assert Agents.empty().section() is None


def test_lists_agents_in_name_order_under_the_header() -> None:
    text = agents(agent("tests", "Writes tests."), agent("review", "Reviews.")).section()
    assert text is not None
    lines = text.splitlines()
    assert lines[0].startswith("Available agents (pass one as the task tool's agent argument")
    assert lines[1:] == ["- review: Reviews.", "- tests: Writes tests."]


def test_descriptions_are_one_line_escaped_and_cut() -> None:
    text = agents(agent("a", "x\n‮y " + "z" * 300)).section()
    assert text is not None
    line = text.splitlines()[1]
    assert "\\u202e" in line and line.endswith("…") and "\n" not in line
    assert len(line) <= len("- a: ") + 200


def test_ac11_many_long_agents_stay_in_budget_and_say_how_many_are_left() -> None:
    items = [agent(f"a{i:03}", "d" * 200) for i in range(500)]
    text = agents(*items).section()
    assert text is not None
    assert len(text) <= SECTION_CHARS
    last = text.splitlines()[-1]
    shown = len(text.splitlines()) - 2
    assert last == f"and {500 - shown} more agents (see /agents)"
