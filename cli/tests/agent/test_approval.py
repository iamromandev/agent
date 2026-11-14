from __future__ import annotations

from src.agent.approval import Choice, Decision, Grants
from src.agent.mode import Mode


def test_mode_values() -> None:
    assert Mode.BUILD == "build"
    assert Mode.PLAN == "plan"


def test_choice_values() -> None:
    assert [c.value for c in Choice] == ["once", "always", "deny"]
    assert Decision(Choice.DENY).note == ""


def test_nothing_is_covered_before_a_grant() -> None:
    grants = Grants()
    assert not grants.covers("edit", {"path": "a.py"}, Mode.BUILD)
    assert not grants.covers("bash", {"command": "ls"}, Mode.BUILD)


def test_a_tool_grant_covers_every_call_of_that_tool() -> None:
    grants = Grants()
    grants.add("edit", {"path": "a.py"})
    assert grants.covers("edit", {"path": "a.py"}, Mode.BUILD)
    assert grants.covers("edit", {"path": "other.py"}, Mode.BUILD)
    assert not grants.covers("write", {"path": "a.py"}, Mode.BUILD)


def test_a_bash_grant_covers_only_that_exact_command() -> None:
    grants = Grants()
    grants.add("bash", {"command": "ls"})
    assert grants.covers("bash", {"command": "ls"}, Mode.BUILD)
    assert not grants.covers("bash", {"command": "ls -la"}, Mode.BUILD)
    assert not grants.covers("bash", {"command": "rm -rf x"}, Mode.BUILD)


def test_a_bash_grant_does_not_cover_other_tools() -> None:
    grants = Grants()
    grants.add("bash", {"command": "ls"})
    assert not grants.covers("edit", {"command": "ls"}, Mode.BUILD)


def test_a_bash_call_without_a_command_is_not_covered() -> None:
    grants = Grants()
    grants.add("bash", {"command": "ls"})
    assert not grants.covers("bash", {}, Mode.BUILD)


def test_plan_mode_is_never_covered() -> None:
    grants = Grants()
    grants.add("edit", {"path": "a.py"})
    grants.add("bash", {"command": "ls"})
    assert not grants.covers("edit", {"path": "a.py"}, Mode.PLAN)
    assert not grants.covers("bash", {"command": "ls"}, Mode.PLAN)
    assert grants.covers("bash", {"command": "ls"}, Mode.BUILD)


def test_a_grant_keyed_by_an_argument_covers_only_that_exact_value() -> None:
    grants = Grants()
    grants.add("web_fetch", {"url": "https://a.test/x", "max_chars": 10}, key="url")
    assert grants.covers("web_fetch", {"url": "https://a.test/x"}, Mode.BUILD, key="url")
    assert not grants.covers("web_fetch", {"url": "https://b.test/x"}, Mode.BUILD, key="url")
    assert not grants.covers("web_fetch", {}, Mode.BUILD, key="url")
    assert "web_fetch" not in grants.tools


def test_a_keyed_grant_does_not_cover_another_tool_with_the_same_value() -> None:
    grants = Grants()
    grants.add("web_fetch", {"url": "u"}, key="url")
    assert not grants.covers("other", {"url": "u"}, Mode.BUILD, key="url")


def test_a_keyed_grant_is_not_covered_in_plan_mode() -> None:
    grants = Grants()
    grants.add("web_fetch", {"url": "u"}, key="url")
    assert not grants.covers("web_fetch", {"url": "u"}, Mode.PLAN, key="url")
