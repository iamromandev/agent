"""NFR1: a keystroke in `@token` stays fast on a 20,000 file workspace."""

from __future__ import annotations

from pathlib import Path

from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.path_suggestions import PathSuggestions

from tests.engine.test_mentions_perf import BUDGET, best_of_two, make_tree
from tests.tui.conftest import AppFactory, until


async def test_a_keystroke_updates_the_list_in_under_50_ms_on_20000_files(make_app: AppFactory, tmp_path: Path) -> None:
    make_tree(tmp_path)
    app = make_app(workspace=tmp_path)
    async with app.run_test() as pilot:
        await until(pilot, lambda: app.file_index.ready, timeout=30)
        box = app.query_one(ChatInput)
        popup = app.query_one(PathSuggestions)
        for text in ("@", "@pkg1", "@pkg042/", "@pkg042/mod0", "@pkg042/od"):
            box.load_text(text)
            box.move_cursor(box.document.end)
            await pilot.pause()
            app._update_suggestions(box)
            assert popup.is_open

            def keystroke() -> None:
                popup.hide()  # as if the previous token's rows were on screen
                app._update_suggestions(box)

            took = best_of_two(keystroke)
            print(f"keystroke {text!r}: {took * 1000:.2f} ms")
            assert took < BUDGET, text
            assert popup.is_open
