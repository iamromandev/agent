from __future__ import annotations

from src.engine.mentions_index import Suggestion
from src.tui.widgets.path_suggestions import PathSuggestions
from textual.app import App, ComposeResult

ITEMS = [Suggestion("src/", True), Suggestion("src/a.py", False), Suggestion("x\x1b[31m.py", False)]


class Host(App[None]):
    def compose(self) -> ComposeResult:
        yield PathSuggestions()


def prompts(widget: PathSuggestions) -> list[str]:
    return [str(widget.get_option_at_index(i).prompt) for i in range(widget.option_count)]


async def test_it_opens_with_rows_and_escapes_paths() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(PathSuggestions)
        assert not widget.is_open
        widget.update_for(ITEMS)
        assert widget.is_open
        assert prompts(widget)[:2] == ["src/", "src/a.py"]
        assert "\x1b" not in prompts(widget)[2]
        assert widget.chosen() == Suggestion("src/", True)


async def test_it_closes_when_empty_and_moves_within_bounds() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(PathSuggestions)
        widget.update_for(ITEMS)
        widget.move(5)
        assert widget.chosen() == ITEMS[2]
        widget.move(-9)
        assert widget.chosen() == ITEMS[0]
        widget.update_for([])
        assert not widget.is_open
        assert widget.chosen() is None


async def test_the_same_suggestions_keep_the_highlight() -> None:
    async with Host().run_test() as pilot:
        widget = pilot.app.query_one(PathSuggestions)
        widget.update_for(ITEMS)
        widget.move(1)
        widget.update_for(list(ITEMS))
        assert widget.chosen() == ITEMS[1]
        widget.update_for(ITEMS[:2])
        assert widget.chosen() == ITEMS[0]
