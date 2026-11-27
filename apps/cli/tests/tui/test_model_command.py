"""`/model <name>` in the TUI: the switch, the window, the busy rule, the saved choice and its trust, the session record."""

from __future__ import annotations

import json
import time
from pathlib import Path

from core.engine.mcp.trust import TrustStore, trust_store_path
from core.engine.permissions.config import project_settings_path
from core.engine.permissions.setup import prepare
from core.engine.provider.base import Done, TextDelta
from core.testing.fakes import FakeCatalog, Gate
from src.tui.screens.permission_trust import PermissionTrustScreen
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry

from tests.tui.conftest import AppFactory, reached, scripted, settled, type_and_send, until


def _notices(app) -> list[str]:
    return [e.text for e in app.query(NoticeEntry)]


def _reply(text: str = "ok"):
    return [TextDelta(text), Done(1, 1)]


def _rules(tmp_path: Path):
    """The rules `src.tui.run` builds for the folder: the only ones that can save to its project file."""
    store = TrustStore(trust_store_path(name="permission-trust.json"))
    return prepare(tmp_path.resolve(), tmp_path / "none.json", store).rules


async def _switch(app, pilot, name: str) -> None:
    await type_and_send(pilot, f"/model {name}")
    await settled(app)
    await pilot.pause()


# --- the switch ----------------------------------------------------------------------------------


async def test_a_failed_length_lookup_still_switches_with_the_cap_and_a_warning(make_app: AppFactory) -> None:
    catalog = FakeCatalog(["fake", "b"])
    catalog.fail_length.add("b")
    app = make_app(catalog=catalog, window_cap=8192)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

        assert (scripted(app).model, scripted(app).context_window) == ("b", 8192)
        assert any("Could not read the context length of b; using AGENT_NUM_CTX (8192)" in n for n in _notices(app))


async def test_hostile_names_are_shown_escaped(make_app: AppFactory) -> None:
    hostile = "ev\x1b[2Jil‮"
    app = make_app(catalog=FakeCatalog(["fake", hostile]))
    async with app.run_test():
        await settled(app)
        for name in ("bad\x1b[31m", hostile):  # typed keys drop such characters: hand the command over as a paste would
            await app._command(f"/model {name}")
            await settled(app)

        text = "\n".join(_notices(app))
        assert "\x1b" not in text and "‮" not in text
        assert "bad\\x1b[31m" in text  # the typed name, in the error
        assert "ev\\x1b[2Jil\\u202e" in text  # the installed one, in the list and in the switch notice


async def test_switching_to_the_current_model_says_so_and_does_nothing_else(make_app: AppFactory, tmp_path: Path) -> None:
    catalog = FakeCatalog(["fake", "b"])
    app = make_app(catalog=catalog)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "fake")

        assert any(n.endswith("Already using fake.") for n in _notices(app))
        assert catalog.length_calls == []
    assert not project_settings_path(tmp_path.resolve()).exists()


async def test_an_ollama_error_during_the_switch_keeps_the_old_model_and_window(make_app: AppFactory, tmp_path: Path) -> None:
    catalog = FakeCatalog(["fake", "b"])
    app = make_app(catalog=catalog, context_window=777)
    async with app.run_test() as pilot:
        catalog.fail = True
        await _switch(app, pilot, "b")

        assert (scripted(app).model, scripted(app).context_window) == ("fake", 777)
        assert (app.query_one(StatusBar).model, app.query_one(StatusBar).window) == ("fake", 777)
        assert any(n.startswith("✗") and "Cannot reach Ollama" in n for n in _notices(app))
    assert not project_settings_path(tmp_path.resolve()).exists()


async def test_a_switch_is_refused_while_a_turn_runs(make_app: AppFactory) -> None:
    gate = Gate()
    app = make_app([[gate, *_reply()]], catalog=FakeCatalog(["fake", "b"]))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "go")
        await reached(gate)
        await type_and_send(pilot, "/model b")
        await pilot.pause()

        assert any("busy" in n for n in _notices(app))
        assert scripted(app).model == "fake"
        gate.release.set()
        await settled(app)
        assert scripted(app).model == "fake"


# --- saving the choice -------------------------------------------------------------------------


async def test_an_untrusted_project_stays_untrusted_after_a_switch(make_app: AppFactory, tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    path = project_settings_path(workspace)
    path.parent.mkdir()
    path.write_text(json.dumps({"permissions": {"allow": ["bash(*)"]}}))
    store = TrustStore(trust_store_path(name="permission-trust.json"))
    first = prepare(workspace, tmp_path / "none.json", store)
    app = make_app(catalog=FakeCatalog(["fake", "b"]), permissions=first.rules)
    async with app.run_test() as pilot:
        await until(pilot, lambda: isinstance(app.screen, PermissionTrustScreen))
        await until(pilot, lambda: time.monotonic() >= app._keys_ignored_until)
        await pilot.press("escape")  # Don't trust
        await until(pilot, lambda: not isinstance(app.screen, PermissionTrustScreen))
        await settled(app)
        await _switch(app, pilot, "b")

    assert prepare(workspace, tmp_path / "none.json", store).rules.pending_trust is True


async def test_a_trust_store_that_fails_says_the_next_launch_will_ask(make_app: AppFactory, tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    rules = prepare(workspace, tmp_path / "none.json", TrustStore(workspace / "trust.json")).rules  # a store inside the workspace
    app = make_app(catalog=FakeCatalog(["fake", "b"]), permissions=rules)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

        assert scripted(app).model == "b"
        assert any("next launch will ask" in n for n in _notices(app))
    assert json.loads(project_settings_path(workspace).read_text())["model"] == "b"


# --- the session record ------------------------------------------------------------------------


# --- the command itself ------------------------------------------------------------------------


# --- review fixes ------------------------------------------------------------------------------------


async def test_launch_notices_are_shown_as_info_lines_with_no_prompt(make_app: AppFactory) -> None:
    app = make_app(launch_notices=["Model z (from .agent/settings.json)"])
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert any(n.endswith("Model z (from .agent/settings.json)") for n in _notices(app))
        assert not isinstance(app.screen, PermissionTrustScreen)
