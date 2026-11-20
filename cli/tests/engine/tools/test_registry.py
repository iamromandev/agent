from __future__ import annotations

from src.engine.tools import default_tools, read_only_tools

from tests.fakes import FakeFetcher


def test_read_only_tools_include_web_search_with_a_searcher() -> None:
    from tests.fakes import FakeSearcher

    assert [t.name for t in read_only_tools()] == ["read", "glob", "grep"]
    names = [t.name for t in read_only_tools(searcher=FakeSearcher())]
    assert names == ["read", "glob", "grep", "web_search"]
    assert [t.name for t in read_only_tools(FakeFetcher(), FakeSearcher())][-2:] == ["web_fetch", "web_search"]


def test_skill_tool_is_offered_only_with_skills() -> None:
    from pathlib import Path

    from src.engine.skills.loader import Skill, Skills

    some = Skills({"a": Skill("a", "d", "b", "global", Path("/x"))}, ())
    empty = Skills({}, ())
    assert "skill" not in [t.name for t in default_tools()]
    assert "skill" not in [t.name for t in default_tools(skills=empty)]
    assert "skill" not in [t.name for t in read_only_tools(skills=empty)]
    assert "skill" in [t.name for t in default_tools(skills=some)]
    assert [t.name for t in read_only_tools(skills=some)] == ["read", "glob", "grep", "skill"]
