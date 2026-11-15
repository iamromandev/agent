"""Rule parsing and matching: the exact, prefix and server-wide forms, and what a prefix never matches."""

from __future__ import annotations

import re
import time

import pytest
from src.engine.permissions.rules import Rule, RuleError, exact_rule_text, parse_rule

OPS = [";", "&&", "||", "|", "`", "$(", ">", "<", "&", "\n"]  # the shell operators of the spec (R4)


def _match(rule: str, tool: str, args: dict[str, object], key: str | None = None) -> bool:
    return parse_rule(rule).matches(tool, args, key)


def _bash(rule: str, command: str) -> bool:
    return _match(rule, "bash", {"command": command})


def _denies(rule: str, command: str, tool: str = "bash") -> bool:
    return parse_rule(rule).denies(tool, {"command": command}, "command")


def test_an_exact_rule_matches_only_that_command() -> None:
    assert _bash("bash(make test)", "make test")
    assert not _bash("bash(make test)", "make test2")
    assert not _bash("bash(make test)", "make")


def test_a_prefix_rule_matches_what_starts_with_the_prefix() -> None:
    assert _bash("bash(git status*)", "git status --short")
    assert _bash("bash(git status*)", "git status")
    assert not _bash("bash(git status*)", "git stat")


def test_a_bare_tool_rule_matches_any_call_of_that_tool() -> None:
    assert _match("write", "write", {"path": "a.py"}, "path")
    assert _match("write", "write", {})
    assert not _match("write", "edit", {"path": "a.py"}, "path")
    assert _bash("bash", "anything at all; rm x")  # a bare rule is the user's choice: no chain check


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


def test_the_key_argument_decides_what_is_compared() -> None:
    assert _match("edit(a.py)", "edit", {"path": "a.py", "command": "x"}, "path")
    assert not _match("edit(x)", "edit", {"path": "a.py", "command": "x"}, "path")


def test_a_missing_or_non_string_value_never_matches() -> None:
    assert not _match("bash(ls*)", "bash", {})
    assert not _match("bash(ls*)", "bash", {"command": 5})
    assert not _match("bash(5)", "bash", {"command": 5})


@pytest.mark.parametrize("op", OPS)
def test_a_chained_or_redirected_command_never_matches_a_prefix_rule(op: str) -> None:
    assert not _bash("bash(git status*)", f"git status{op}rm x")


@pytest.mark.parametrize("op", OPS)
def test_an_exact_rule_equal_to_a_chained_command_matches(op: str) -> None:
    command = f"git status{op}rm x"
    assert _bash(f"bash({command})", command)


def test_a_chain_inside_the_prefix_does_not_make_it_match_a_chained_command() -> None:
    assert not _bash("bash(git status; *)", "git status; rm x")  # still chained: only an exact rule may allow it
    assert _bash("bash(git status; rm x)", "git status; rm x")


@pytest.mark.parametrize(
    "command",
    [
        "git status =(touch pwned)",  # zsh process substitution into a file
        "git status *(e:'touch pwned':)",  # zsh glob qualifier that runs code
        "git status (touch pwned)",  # fish/zsh subshell
        "git status {touch,pwned}",  # brace expansion
        "git status ~/x",
        "git status 'x'",
        'git status "x"',
        "git status \\x",
        "git status #x",
        "git status !x",
        "git status *",
        "git status ?",
        "git status [a]",
        "git status\tx",
    ],
)
def test_a_prefix_rule_only_matches_plain_commands(command: str) -> None:
    assert not _bash("bash(git status*)", command)


@pytest.mark.parametrize(
    "command",
    ["git status --short", "git log --format=%h -n 5", "git diff a.py b/c.py", "make test-all", "uv run pytest tests/a_b.py::t -q", "git show @", "echo a,b+c:d"],
)
def test_plain_arguments_after_a_prefix_still_match(command: str) -> None:
    prefix = command.split(" ", 1)[0]
    assert _bash(f"bash({prefix} *)", command)


def test_a_prefix_rule_for_term_start_is_held_to_the_same_remainder_rule() -> None:
    rule = "term_start(npm run dev*)"
    assert _match(rule, "term_start", {"command": "npm run dev -- --port 3000"}, "command")
    assert not _match(rule, "term_start", {"command": "npm run dev; rm -rf x"}, "command")
    assert not _match(rule, "term_start", {"command": "npm run dev =(touch x)"}, "command")


def test_a_non_shell_prefix_rule_allows_any_remainder() -> None:
    assert _match("web_fetch(https://a.org/*)", "web_fetch", {"url": "https://a.org/x?a=1&b=2;c#d"}, "url")


# deny rules are matched the other way round: anything that could hide the denied command still matches


@pytest.mark.parametrize("op", [";", "&&", "||", "|", "`", "$(", "&", "\n"])  # `>` and `<` are followed by a file name
def test_a_deny_prefix_matches_a_denied_command_after_any_command_starting_operator(op: str) -> None:
    assert _denies("bash(rm*)", f"echo hi{op}rm -rf x")


@pytest.mark.parametrize("op", OPS)
def test_a_deny_prefix_matches_a_denied_command_before_any_shell_operator(op: str) -> None:
    assert _denies("bash(rm*)", f"rm -rf x{op}echo hi")


@pytest.mark.parametrize(
    "command",
    ["  rm -rf x", "\trm x", "echo a\n\n  rm x", "(rm x)", "echo $(rm x)", "echo `rm x`", "{ rm x; }", "cat <(rm x)", "a;rm x", "a&&  rm x"],
)
def test_a_deny_prefix_sees_through_leading_whitespace_and_nesting(command: str) -> None:
    assert _denies("bash(rm*)", command)


def test_a_deny_prefix_still_does_not_match_an_unrelated_command() -> None:
    assert not _denies("bash(rm*)", "echo hi; ls -la")
    assert not _denies("bash(rm*)", "farm x")


def test_a_deny_prefix_with_a_chain_in_it_matches_the_whole_command() -> None:
    assert _denies("bash(git push; *)", "  git push; rm x")


@pytest.mark.parametrize("rule", ["bash(*)", "bash"])
@pytest.mark.parametrize("command", ["ls", "", "  ", "a; b", "x\ny"])
def test_denying_every_bash_command_denies_every_command(rule: str, command: str) -> None:
    assert _denies(rule, command)


def test_an_exact_deny_rule_matches_the_command_in_any_segment_of_a_chain() -> None:
    assert _denies("bash(rm -rf build)", "rm -rf build")
    assert _denies("bash(rm -rf build)", "  rm -rf build  ")
    assert _denies("bash(rm -rf build)", "echo hi; rm -rf build")
    assert not _denies("bash(rm -rf build)", "rm -rf build2")


def test_a_deny_for_term_start_is_shell_aware_too() -> None:
    assert _denies("term_start(rm*)", "echo x; rm y", tool="term_start")


def test_a_deny_for_another_tool_is_a_plain_prefix() -> None:
    assert _denies("web_fetch(https://evil.org/*)", "https://evil.org/x", tool="web_fetch")
    assert not _denies("web_fetch(https://evil.org/*)", "https://ok.org/ https://evil.org/x", tool="web_fetch")


@pytest.mark.parametrize(
    ("text", "tool", "value", "prefix", "server"),
    [
        ("bash", "bash", None, False, False),
        ("bash(make test)", "bash", "make test", False, False),
        ("bash(git status*)", "bash", "git status", True, False),
        ("bash(*)", "bash", "", True, False),
        ("mcp__gh__*", "mcp__gh__", None, False, True),
        ("mcp__my-srv_1__*", "mcp__my-srv_1__", None, False, True),
        ("mcp__gh__issues", "mcp__gh__issues", None, False, False),
        ("bash(echo (hi))", "bash", "echo (hi)", False, False),
    ],
)
def test_parsing_gives_the_parts(text: str, tool: str, value: str | None, prefix: bool, server: bool) -> None:
    rule = parse_rule(text)
    assert (rule.text, rule.tool, rule.value, rule.prefix, rule.server_wildcard) == (text, tool, value, prefix, server)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("bash()", "empty value"),
        ("bash(a*b)", "* only at the end"),
        ("bash(*a)", "* only at the end"),
        ("bash(a**)", "* only at the end"),
        ("ba*sh", "* only"),
        ("*", "* only"),
        ("mcp__*", "* only"),
        ("mcp____*", "* only"),
        ("bash(ls", "parenthes"),
        ("bash(ls)x", "parenthes"),
        ("bash)", "parenthes"),
        ("ba sh", "name"),
        ("(ls)", "name"),
        ("bash!(ls)", "name"),
    ],
)
def test_an_invalid_rule_is_refused_with_a_reason(text: str, reason: str) -> None:
    with pytest.raises(RuleError, match=re.escape(reason)):
        parse_rule(text)


def test_matching_500_rules_takes_under_a_millisecond_per_decision() -> None:
    rules = [parse_rule(f"bash(command-{i}*)") for i in range(250)] + [parse_rule(f"web_fetch(https://h{i}.org/)") for i in range(250)]
    calls = [("bash", {"command": f"other {i}"}, None) for i in range(500)] + [
        ("web_fetch", {"url": f"https://none{i}.org/"}, "url") for i in range(500)
    ]
    start = time.perf_counter()
    for tool, args, key in calls:
        assert not any(rule.matches(tool, args, key) for rule in rules)
    per_decision = (time.perf_counter() - start) / len(calls)
    assert per_decision < 0.001


def test_a_rule_is_hashable_and_equal_to_the_same_text_parsed_again() -> None:
    rule = parse_rule("bash(ls*)")
    assert isinstance(rule, Rule)
    assert rule in {parse_rule("bash(ls*)")}


def test_the_exact_rule_for_a_call_matches_that_call_and_only_it() -> None:
    text = exact_rule_text("bash", "uv run pytest -q")

    assert text == "bash(uv run pytest -q)"
    assert _bash(text, "uv run pytest -q")
    assert not _bash(text, "uv run pytest -q -x")


def test_the_exact_rule_without_a_value_is_the_bare_tool() -> None:
    assert exact_rule_text("write", None) == "write"
    assert exact_rule_text("mcp__gh__issues", None) == "mcp__gh__issues"


@pytest.mark.parametrize("value", ["rm x*", "ls *.py", "*", "", "a**"])
def test_a_value_a_rule_cannot_express_exactly_has_no_rule(value: str) -> None:
    """`bash(rm x*)` would be a prefix rule, broader than the one command the user approved."""
    with pytest.raises(RuleError):
        exact_rule_text("bash", value)
