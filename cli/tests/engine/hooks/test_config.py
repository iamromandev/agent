"""Reading the `hooks` key of a settings file: entries, warnings, tool matchers."""

from __future__ import annotations

import pytest
from src.engine.hooks.config import DEFAULT_TIMEOUT, HookEntry, read_hooks


def test_each_bad_entry_gives_one_escaped_warning_and_the_valid_ones_load() -> None:
    data = {
        "hooks": {
            "PreToolUse": [
                {"command": "ok"},
                {"command": ""},
                {"command": "neg", "timeout": -1},
                {"command": 5},
                "not an object",
                {"command": "zero", "timeout": 0},
                {"command": "str", "timeout": "5"},
                {"command": "flag", "timeout": True},
                {},
            ],
            "Bogus\x1b[31m": [{"command": "x"}],
            "PostToolUse": "not a list",
        }
    }

    entries, warnings = read_hooks(data, "global")

    assert [e.command for e in entries] == ["ok"]
    assert len(warnings) == 8 + 1 + 1  # 8 bad entries, the unknown event, the non-list
    assert all("\x1b" not in w for w in warnings)
    assert any("Bogus" in w and "unknown event" in w for w in warnings)
    assert all(w.startswith("hooks (global)") for w in warnings)


def test_hooks_that_is_not_an_object_is_one_warning() -> None:
    entries, warnings = read_hooks({"hooks": ["x"]}, "project")

    assert entries == []
    assert len(warnings) == 1 and "object" in warnings[0]


def test_a_matcher_on_an_event_without_tools_is_ignored_with_a_warning() -> None:
    entries, warnings = read_hooks({"hooks": {"Stop": [{"command": "x", "matcher": "edit"}]}}, "global")

    assert [(e.command, e.matcher) for e in entries] == [("x", None)]
    assert len(warnings) == 1 and "matcher" in warnings[0]


@pytest.mark.parametrize("matcher", ["bash(ls)", 5])
def test_an_invalid_matcher_skips_the_entry_with_a_warning(matcher: object) -> None:
    entries, warnings = read_hooks({"hooks": {"PreToolUse": [{"command": "x", "matcher": matcher}]}}, "global")

    assert entries == []
    assert len(warnings) == 1


def entry(matcher: str | None) -> HookEntry:
    return HookEntry("PreToolUse", "x", matcher, DEFAULT_TIMEOUT, "global")


def test_matchers_match_exact_prefix_and_server_wildcard() -> None:
    assert entry("edit").matches("edit") and not entry("edit").matches("write")
    assert entry("mcp__gh__*").matches("mcp__gh__issue") and not entry("mcp__gh__*").matches("mcp__other__issue")
    assert entry("te*").matches("term_start") and not entry("te*").matches("write")


def test_a_lone_star_is_stored_as_every_tool() -> None:
    entries, _ = read_hooks({"hooks": {"PostToolUse": [{"command": "x", "matcher": "*"}]}}, "global")

    assert [e.matcher for e in entries] == [None]
