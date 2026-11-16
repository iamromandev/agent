"""The bare `/model` picker: rows, the switch, Esc, failures, and that the input stays usable while it loads."""

from __future__ import annotations

import asyncio

import pytest
from src.engine.provider.models import ModelInfo
from src.tui.screens.models import ModelPicker
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry
from textual.screen import ModalScreen
from textual.widgets import OptionList

from tests.fakes import FakeCatalog
from tests.tui.conftest import AppFactory, type_and_send, until

GIB = 1024**3


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _rows(app) -> list[str]:
    picker = app.screen.query_one(OptionList)
    return [str(picker.get_option_at_index(i).prompt) for i in range(picker.option_count)]


async def _open(app, pilot, command: str = "/model") -> None:
    await type_and_send(pilot, command)
    await until(pilot, lambda: isinstance(app.screen, ModelPicker))


def _catalog(**kwargs) -> FakeCatalog:
    models = [ModelInfo("fake", 2 * GIB, "7.6B"), ModelInfo("other", None, None)]
    return FakeCatalog(models, lengths={"fake": 32768}, **kwargs)


async def test_rows_show_name_size_parameters_context_and_mark_the_current(make_app: AppFactory) -> None:
    catalog = _catalog()
    catalog.fail_length.add("other")
    app = make_app(catalog=catalog)
    async with app.run_test() as pilot:
        await _open(app, pilot)
        first, second = _rows(app)
        assert first.startswith("* fake") and "2.0 GiB" in first and "7.6B" in first and "ctx 32,768" in first
        assert second.startswith("  other") and second.count("?") == 3


async def test_a_name_with_control_characters_is_escaped(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "ev\x1bil"]))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        assert all("\x1b" not in row for row in _rows(app))


async def test_enter_switches_with_the_length_already_read(make_app: AppFactory) -> None:
    catalog = FakeCatalog(["fake", "other"], lengths={"other": 4096})
    app = make_app(catalog=catalog, context_window=8192)
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await pilot.press("down", "enter")
        await until(pilot, lambda: app.query_one(StatusBar).model == "other")
        assert app.query_one(StatusBar).window == 4096
        assert catalog.length_calls.count("other") == 1
        assert not app._switching


async def test_escape_changes_nothing(make_app: AppFactory) -> None:
    app = make_app(catalog=_catalog())
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await pilot.press("down")
        await pilot.press("escape")
        await pilot.pause()
        await until(pilot, lambda: not isinstance(app.screen, ModelPicker))
        assert app.query_one(StatusBar).model == "fake"
        assert not app._switching


async def test_no_models_installed(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog([]))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model   ")
        await until(pilot, lambda: "No models installed." in _notices(app)[-1:] or any("No models installed." in n for n in _notices(app)))


async def test_a_list_failure_is_one_error_line(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake"], fail=True))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model")
        await until(pilot, lambda: any("Could not list" in n for n in _notices(app)))
        assert not isinstance(app.screen, ModelPicker)
        assert app.query_one(StatusBar).model == "fake"


async def test_the_input_stays_usable_while_the_list_loads(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "other"], delay=0.3))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model")
        await pilot.pause(0.05)
        assert not app._switching
        await pilot.press(*"abc")
        assert app.query_one(ChatInput).text == "abc"


def _busy(app) -> bool:
    return any("busy" in n for n in _notices(app))


async def test_a_second_bare_model_while_loading_is_refused(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "other"], delay=0.3))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model")
        await pilot.pause(0.05)
        await app._command("/model")
        await until(pilot, lambda: any(isinstance(s, ModelPicker) for s in app.screen_stack))
        await pilot.pause(0.2)
        assert sum(isinstance(s, ModelPicker) for s in app.screen_stack) == 1
        assert any("already" in n for n in _notices(app))


async def test_a_second_bare_model_while_the_picker_is_open_is_refused(make_app: AppFactory) -> None:
    app = make_app(catalog=_catalog())
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await app._command("/model")
        await pilot.pause(0.1)
        assert sum(isinstance(s, ModelPicker) for s in app.screen_stack) == 1
        assert any("already" in n for n in _notices(app))


async def test_a_turn_started_while_loading_skips_the_picker(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "other"], delay=0.3))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model")
        await pilot.pause(0.05)
        app._turn_active = True
        await pilot.pause(0.8)
        assert not isinstance(app.screen, ModelPicker)
        assert _busy(app)
        assert app.query_one(StatusBar).model == "fake"
        app._turn_active = False


async def test_a_switch_started_while_the_picker_is_open_makes_enter_busy(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "other"]))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        app._switching = True
        await pilot.press("down", "enter")
        await until(pilot, lambda: _busy(app))
        assert app.query_one(StatusBar).model == "fake"
        assert app._switching  # not ours to release
        app._switching = False


async def test_a_model_removed_before_enter_is_not_installed(make_app: AppFactory) -> None:
    catalog = FakeCatalog(["fake", "other"])
    app = make_app(catalog=catalog)
    async with app.run_test() as pilot:
        await _open(app, pilot)
        catalog.models = [m for m in catalog.models if m.name != "other"]
        await pilot.press("down", "enter")
        await until(pilot, lambda: any("not installed" in n for n in _notices(app)))
        assert app.query_one(StatusBar).model == "fake"
        assert not app._switching


async def test_a_failed_lookup_shows_a_question_mark_and_a_warning_on_switch(make_app: AppFactory) -> None:
    catalog = FakeCatalog(["fake", "other"])
    catalog.fail_length.add("other")
    app = make_app(catalog=catalog, context_window=8192)
    async with app.run_test() as pilot:
        await _open(app, pilot)
        assert "ctx ?" in _rows(app)[1]
        await pilot.press("down", "enter")
        await until(pilot, lambda: app.query_one(StatusBar).model == "other")
        assert any("Could not read the context length" in n for n in _notices(app))
        assert app.query_one(StatusBar).window == 8192


class _Slow(FakeCatalog):
    async def context_length(self, name: str) -> int | None:
        if name == "other":
            await asyncio.sleep(1)
        return await super().context_length(name)


async def test_a_lookup_timeout_shows_a_question_mark_and_a_warning(make_app: AppFactory, monkeypatch) -> None:
    import src.tui.app as app_module

    monkeypatch.setattr(app_module, "SHOW_TIMEOUT", 0.05)
    catalog = _Slow(["fake", "other"])
    app = make_app(catalog=catalog)
    async with app.run_test() as pilot:
        await _open(app, pilot)
        assert "ctx ?" in _rows(app)[1]
        await pilot.press("down", "enter")
        await until(pilot, lambda: app.query_one(StatusBar).model == "other")
        assert any("did not answer" in n for n in _notices(app))


class _Boom(FakeCatalog):
    where = "list"

    async def list(self):
        if self.where == "list":
            raise RuntimeError("boom")
        return await super().list()

    async def context_length(self, name: str):
        raise RuntimeError("boom")


@pytest.mark.parametrize("where", ["list", "length"])
async def test_a_generic_error_leaves_the_state_unchanged(make_app: AppFactory, where: str) -> None:
    catalog = _Boom(["fake", "other"])
    catalog.where = where
    app = make_app(catalog=catalog)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model")
        await until(pilot, lambda: any("Model switch failed" in n for n in _notices(app)))
        assert not isinstance(app.screen, ModelPicker)
        assert not app._switching
        assert app.query_one(StatusBar).model == "fake"


async def test_escape_while_the_list_loads_opens_no_picker(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "other"], delay=0.3))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model")
        await pilot.pause(0.05)
        await pilot.press("escape")
        await pilot.pause(0.8)
        assert not isinstance(app.screen, ModelPicker)
        assert not app._switching
        await app._command("/model")  # and a new one can start
        await until(pilot, lambda: isinstance(app.screen, ModelPicker))


async def test_a_picker_is_not_opened_over_another_dialog(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "other"], delay=0.3))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model")
        await pilot.pause(0.05)
        app.push_screen(ModalScreen())
        await pilot.pause(0.8)
        assert not any(isinstance(s, ModelPicker) for s in app.screen_stack)


async def test_long_parameters_are_cut_in_the_row(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog([ModelInfo("fake", None, "9" * 40), ModelInfo("o", None, "7B\x1b" * 10)]))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        first, second = _rows(app)
        assert "9" * 13 not in first and "9" * 12 in first
        assert "\x1b" not in second and not second.split("  ")[-2].endswith("\\")


async def test_the_row_name_is_written_like_the_notices(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "é-ünï"]))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        assert "é-ünï" in _rows(app)[1]
