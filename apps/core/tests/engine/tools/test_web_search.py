from __future__ import annotations

from pathlib import Path

import pytest
from core.engine.search_web import SearchError, SearchHit
from core.engine.tools.base import Prepared, ToolContext, ToolResult
from core.engine.tools.web_search import WebSearch
from core.testing.fakes import FakeSearcher


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path, bash_timeout=5.0)


def hit(n: int, title: str = "", snippet: str = "snip") -> SearchHit:
    return SearchHit(title or f"Title {n}", f"https://example.com/{n}", snippet)


async def run(searcher: FakeSearcher, ctx: ToolContext, **args: object) -> ToolResult:
    args.setdefault("query", "textual")
    prepared = await WebSearch(searcher).prepare(args, ctx)
    assert isinstance(prepared, Prepared)
    return await prepared.execute()


async def test_one_result_summary_is_singular_and_empty_snippet_line_is_omitted(ctx: ToolContext) -> None:
    result = await run(FakeSearcher([hit(1, snippet="")]), ctx)
    assert result.summary == "1 result"
    assert result.output.endswith("1. Title 1\n   https://example.com/1")


async def test_output_over_the_limit_is_cut_with_a_note(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.engine.tools.web_search.TOOL_OUTPUT_LIMIT", 200)
    result = await run(FakeSearcher([hit(n, snippet="s" * 250) for n in range(1, 6)]), ctx, limit=5)
    assert len(result.output) <= 200 and result.output.endswith("[truncated]")
    assert result.output.startswith("Search results (untrusted")


async def test_no_hits_is_a_non_error_no_results_text(ctx: ToolContext) -> None:
    result = await run(FakeSearcher([]), ctx, query="zzz")
    assert not result.is_error
    assert result.output == 'No results for "zzz".' and result.summary == "no results"


async def test_a_search_error_is_an_error_result_with_its_summary(ctx: ToolContext) -> None:
    error = SearchError("search failed: timeout after 15s", "timeout")
    result = await run(FakeSearcher(error), ctx)
    assert result.is_error and result.output == "search failed: timeout after 15s"
    assert result.summary == "timeout"


@pytest.mark.parametrize("query", [""])
async def test_empty_query_is_refused_without_a_search(ctx: ToolContext, query: str) -> None:
    searcher = FakeSearcher([hit(1)])
    result = await WebSearch(searcher).prepare({"query": query}, ctx)
    assert isinstance(result, ToolResult) and result.is_error and "must not be empty" in result.output
    assert searcher.calls == []


async def test_501_char_query_is_refused_and_500_is_accepted(ctx: ToolContext) -> None:
    searcher = FakeSearcher([hit(1)])
    result = await WebSearch(searcher).prepare({"query": "q" * 501}, ctx)
    assert isinstance(result, ToolResult) and result.is_error and "longer than 500" in result.output
    assert searcher.calls == []
    assert isinstance(await WebSearch(searcher).prepare({"query": "q" * 500}, ctx), Prepared)


async def test_missing_query_is_invalid_arguments(ctx: ToolContext) -> None:
    result = await WebSearch(FakeSearcher()).prepare({}, ctx)
    assert isinstance(result, ToolResult) and result.is_error and result.summary == "invalid arguments"
