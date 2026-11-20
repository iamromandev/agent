"""Rule parsing and matching: the exact, prefix and server-wide forms, and what a prefix never matches."""

from __future__ import annotations

import re

import pytest
from src.engine.permissions.rules import RuleError, exact_rule_text, name_matcher, parse_rule

OPS = [";", "&&", "||", "|", "`", "$(", ">", "<", "&", "\n"]  # the shell operators of the spec (R4)


def _match(rule: str, tool: str, args: dict[str, object], key: str | None = None) -> bool:
    return parse_rule(rule).matches(tool, args, key)


def _bash(rule: str, command: str) -> bool:
    return _match(rule, "bash", {"command": command})


def _denies(rule: str, command: str, tool: str = "bash") -> bool:
    return parse_rule(rule).denies(tool, {"command": command}, "command")


def test_a_server_rule_matches_every_tool_of_that_server_only() -> None:
    assert _match("mcp__gh__*", "mcp__gh__issues", {})
    assert _match("mcp__gh__*", "mcp__gh__a_b", {"x": 1})
    assert not _match("mcp__gh__*", "mcp__ghx__a", {})
    assert not _match("mcp__gh__*", "mcp__gh", {})


def test_a_web_fetch_prefix_matches_urls_under_it() -> None:
    rule = "web_fetch(https://docs.python.org/*)"
    assert _match(rule, "web_fetch", {"url": "https://docs.python.org/3/library/os.html"}, "url")
    assert not _match(rule, "web_fetch", {"url": "https://example.com/"}, "url")


def test_a_valued_rule_for_a_tool_without_a_key_argument_never_matches() -> None:
    assert not _match("edit(a.py)", "edit", {"path": "a.py"})  # no key: only the bare name form can match
    assert not _match("edit(a*)", "edit", {"path": "a.py"})


# deny rules are matched the other way round: anything that could hide the denied command still matches


def test_a_deny_prefix_still_does_not_match_an_unrelated_command() -> None:
    assert not _denies("bash(rm*)", "echo hi; ls -la")
    assert not _denies("bash(rm*)", "farm x")


def test_an_exact_deny_rule_matches_the_command_in_any_segment_of_a_chain() -> None:
    assert _denies("bash(rm -rf build)", "rm -rf build")
    assert _denies("bash(rm -rf build)", "  rm -rf build  ")
    assert _denies("bash(rm -rf build)", "echo hi; rm -rf build")
    assert not _denies("bash(rm -rf build)", "rm -rf build2")


def test_a_deny_for_another_tool_is_a_plain_prefix() -> None:
    assert _denies("web_fetch(https://evil.org/*)", "https://evil.org/x", tool="web_fetch")
    assert not _denies("web_fetch(https://evil.org/*)", "https://ok.org/ https://evil.org/x", tool="web_fetch")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("bash()", "empty value"),
        ("bash(ls", "parenthes"),
        ("bash)", "parenthes"),
    ],
)
def test_an_invalid_rule_is_refused_with_a_reason(text: str, reason: str) -> None:
    with pytest.raises(RuleError, match=re.escape(reason)):
        parse_rule(text)


@pytest.mark.parametrize("value", ["ls *.py"])
def test_a_value_a_rule_cannot_express_exactly_has_no_rule(value: str) -> None:
    """`bash(rm x*)` would be a prefix rule, broader than the one command the user approved."""
    with pytest.raises(RuleError):
        exact_rule_text("bash", value)


@pytest.mark.parametrize(
    ("matcher", "tool", "expected"),
    [
        ("*", "anything", True),
    ],
)
def test_name_matcher_tests_the_tool_name_only(matcher: str, tool: str, expected: bool) -> None:
    assert name_matcher(matcher)(tool) is expected


@pytest.mark.parametrize("text", ["", "a*b", "bad name", "(x)"])
def test_name_matcher_rejects_text_that_names_more_than_a_tool(text: str) -> None:
    with pytest.raises(RuleError):
        name_matcher(text)
