from __future__ import annotations

from typing import Any

from src.agent.provider.base import ToolSpec
from src.agent.tools.base import Prepared, Tool, ToolContext, ToolResult, check_args

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "limit": {"type": "integer"},
        "all": {"type": "boolean"},
    },
    "required": ["path"],
}


def test_check_args_accepts_valid_arguments() -> None:
    assert check_args(SCHEMA, {"path": "a.py", "limit": 3, "all": True}) is None


def test_check_args_reports_a_missing_required_key() -> None:
    assert check_args(SCHEMA, {}) == 'missing "path"'


def test_check_args_reports_a_wrong_string_type() -> None:
    assert check_args(SCHEMA, {"path": 3}) == '"path" must be a string'


def test_check_args_reports_a_wrong_integer_type() -> None:
    assert check_args(SCHEMA, {"path": "a", "limit": "3"}) == '"limit" must be an integer'


def test_check_args_rejects_a_bool_for_an_integer() -> None:
    assert check_args(SCHEMA, {"path": "a", "limit": True}) == '"limit" must be an integer'


def test_check_args_reports_a_wrong_boolean_type() -> None:
    assert check_args(SCHEMA, {"path": "a", "all": "yes"}) == '"all" must be a boolean'


def test_check_args_ignores_unknown_keys() -> None:
    assert check_args(SCHEMA, {"path": "a", "extra": object()}) is None


def test_check_args_reports_arguments_that_were_not_valid_json() -> None:
    optional: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    assert check_args(optional, {"__invalid__": "{oops"}) == "arguments were not valid JSON"


class _Dummy(Tool):
    name = "dummy"
    description = "A dummy tool"
    parameters = SCHEMA

    async def prepare(self, args: dict[str, Any], ctx: ToolContext) -> Prepared | ToolResult:
        return ToolResult("ok")


def test_tool_spec_and_defaults() -> None:
    tool = _Dummy()
    assert tool.spec() == ToolSpec("dummy", "A dummy tool", SCHEMA)
    assert tool.mutating is False
    assert tool.needs_approval is False


def test_tool_label_defaults_to_name_and_first_argument() -> None:
    assert _Dummy().label({"path": "src/a.py"}) == "dummy src/a.py"
    assert _Dummy().label({}) == "dummy"
