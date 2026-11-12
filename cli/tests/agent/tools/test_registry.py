from __future__ import annotations

from src.agent.tools import default_tools


def test_default_tools_are_exactly_the_six_in_order() -> None:
    assert [t.name for t in default_tools()] == ["read", "glob", "grep", "write", "edit", "bash"]


def test_default_tools_returns_fresh_instances() -> None:
    assert default_tools()[0] is not default_tools()[0]


def test_only_write_and_edit_are_mutating() -> None:
    assert {t.name for t in default_tools() if t.mutating} == {"write", "edit"}


def test_only_write_edit_and_bash_need_approval() -> None:
    assert {t.name for t in default_tools() if t.needs_approval} == {"write", "edit", "bash"}


def test_every_spec_is_an_object_schema_with_required_properties() -> None:
    for tool in default_tools():
        spec = tool.spec()
        assert spec.name == tool.name
        assert spec.description
        assert spec.parameters["type"] == "object"
        required = spec.parameters["required"]
        assert required, tool.name
        assert set(required) <= set(spec.parameters["properties"]), tool.name
        for prop in spec.parameters["properties"].values():
            assert prop["type"] in {"string", "integer", "boolean"}
            assert prop["description"]
