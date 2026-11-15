from __future__ import annotations

from src.engine.tools import default_tools, read_only_tools

from tests.fakes import FakeFetcher


def test_default_tools_are_exactly_the_seven_in_order() -> None:
    assert [t.name for t in default_tools()] == ["read", "glob", "grep", "write", "edit", "bash", "todo"]


def test_todo_needs_no_approval_and_is_not_mutating() -> None:
    todo = next(t for t in default_tools() if t.name == "todo")
    assert not todo.needs_approval and not todo.mutating
    assert todo.spec().parameters["required"] == ["items"]


def test_default_tools_returns_fresh_instances() -> None:
    assert default_tools()[0] is not default_tools()[0]


def test_only_write_and_edit_are_mutating() -> None:
    assert {t.name for t in default_tools() if t.mutating} == {"write", "edit"}


def test_only_write_edit_and_bash_need_approval() -> None:
    assert {t.name for t in default_tools() if t.needs_approval} == {"write", "edit", "bash"}


def test_every_spec_is_an_object_schema_with_required_properties() -> None:
    for tool in default_tools():
        if tool.name == "todo":  # its one property is an array of objects: see the test below
            continue
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


def test_web_fetch_is_added_last_only_with_a_fetcher() -> None:
    from tests.fakes import FakeFetcher

    tools = default_tools(fetcher=FakeFetcher())
    assert [t.name for t in tools][-1] == "web_fetch"
    assert len(tools) == 8
    web = tools[-1]
    assert web.needs_approval and not web.mutating
    assert web.spec().parameters["required"] == ["url"]
    assert "web_fetch" not in [t.name for t in default_tools()]


def test_read_only_tools_are_read_glob_grep_and_web_fetch_with_a_fetcher() -> None:
    assert [t.name for t in read_only_tools()] == ["read", "glob", "grep"]
    assert [t.name for t in read_only_tools(FakeFetcher())] == ["read", "glob", "grep", "web_fetch"]


def test_no_read_only_tool_mutates_or_runs_commands() -> None:
    tools = read_only_tools(FakeFetcher())
    assert not any(t.mutating for t in tools)
    assert {t.name for t in tools}.isdisjoint({"write", "edit", "bash", "task", "todo"})


def test_todo_items_schema_is_an_array_of_objects_with_content_and_status() -> None:
    todo = next(t for t in default_tools() if t.name == "todo")
    parameters = todo.spec().parameters
    items = parameters["properties"]["items"]

    assert parameters["type"] == "object" and set(parameters["properties"]) == {"items"}
    assert items["type"] == "array" and items["description"]
    assert items["items"]["type"] == "object"
    assert items["items"]["required"] == ["content", "status"]
    assert items["items"]["properties"]["content"]["type"] == "string"
    assert items["items"]["properties"]["status"] == {"type": "string", "enum": ["pending", "in_progress", "done"]}


def test_web_search_is_added_last_only_with_a_searcher() -> None:
    from tests.fakes import FakeSearcher

    assert "web_search" not in [t.name for t in default_tools(fetcher=FakeFetcher())]
    tools = default_tools(fetcher=FakeFetcher(), searcher=FakeSearcher())
    assert [t.name for t in tools][-2:] == ["web_fetch", "web_search"]
    search = tools[-1]
    assert not search.needs_approval and not search.mutating
    assert search.spec().parameters["required"] == ["query"]


def test_read_only_tools_include_web_search_with_a_searcher() -> None:
    from tests.fakes import FakeSearcher

    assert [t.name for t in read_only_tools()] == ["read", "glob", "grep"]
    names = [t.name for t in read_only_tools(searcher=FakeSearcher())]
    assert names == ["read", "glob", "grep", "web_search"]
    assert [t.name for t in read_only_tools(FakeFetcher(), FakeSearcher())][-2:] == ["web_fetch", "web_search"]
