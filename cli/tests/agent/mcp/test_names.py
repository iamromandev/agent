from __future__ import annotations

import pytest
from src.agent.mcp.names import tool_name


def test_plain_names_are_joined_with_the_prefix() -> None:
    assert tool_name("fs-notes", "search_notes") == "mcp__fs-notes__search_notes"


@pytest.mark.parametrize(
    ("server", "tool", "expected"),
    [
        ("my.srv", "do/thing", "mcp__my_srv__do_thing"),
        ("a b", "c:d", "mcp__a_b__c_d"),
        ("srv", "naïve", "mcp__srv__na_ve"),
    ],
)
def test_characters_the_model_does_not_allow_become_underscores(server: str, tool: str, expected: str) -> None:
    assert tool_name(server, tool) == expected


def test_names_that_clean_to_the_same_text_collide() -> None:
    assert tool_name("s", "do/thing") == tool_name("s", "do_thing")


def test_double_underscores_in_a_name_are_kept_and_still_prefixed() -> None:
    assert tool_name("a__b", "c__d") == "mcp__a__b__c__d"
