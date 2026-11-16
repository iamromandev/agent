"""`/model <name>` in the TUI: the switch, the window, the busy rule, the saved choice and its trust, the session record."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from src.engine.mcp.trust import TrustStore, trust_store_path
from src.engine.permissions.config import project_settings_path
from src.engine.permissions.setup import prepare
from src.engine.provider.base import Done, TextDelta
from src.engine.session import FileRecorder, SessionStore, sessions_root, workspace_dir
from src.tui import StartChoice
from src.tui.app import HELP
from src.tui.screens.permission_trust import PermissionTrustScreen
from src.tui.widgets.chat_input import ChatInput
from src.tui.widgets.status import StatusBar
from src.tui.widgets.transcript import NoticeEntry

from tests.fakes import FakeCatalog, Gate
from tests.tui.conftest import AppFactory, chat, reached, scripted, settled, type_and_send, until


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


async def test_the_next_request_goes_to_the_new_model_with_its_window_and_the_full_history(make_app: AppFactory) -> None:
    catalog = FakeCatalog(["fake", "other"], lengths={"other": 4096})
    app = make_app([_reply("one"), _reply("two")], context_window=8192, catalog=catalog)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await settled(app)
        await _switch(app, pilot, "other")
        await type_and_send(pilot, "second")
        await settled(app)

        provider = scripted(app)
        assert provider.used == [("fake", 8192), ("other", 4096)]
        messages, _tools = provider.requests[-1]
        assert [(m.role, m.content) for m in messages[1:]] == [
            ("user", "first"),
            ("assistant", "one"),
            ("user", "second"),
        ]
        status = app.query_one(StatusBar)
        assert (status.model, status.window) == ("other", 4096)
        assert any(n.endswith("Model: other (context 4,096)") for n in _notices(app))


@pytest.mark.parametrize(("length", "window"), [(40960, 8192), (4096, 4096), (None, 8192)])
async def test_the_window_is_the_smaller_of_the_models_length_and_the_cap(
    make_app: AppFactory, length: int | None, window: int
) -> None:
    catalog = FakeCatalog(["fake", "b"], lengths={} if length is None else {"b": length})
    app = make_app(catalog=catalog, window_cap=8192)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

        assert scripted(app).context_window == window
        assert app.query_one(StatusBar).window == window


async def test_a_failed_length_lookup_still_switches_with_the_cap_and_a_warning(make_app: AppFactory) -> None:
    catalog = FakeCatalog(["fake", "b"])
    catalog.fail_length.add("b")
    app = make_app(catalog=catalog, window_cap=8192)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

        assert (scripted(app).model, scripted(app).context_window) == ("b", 8192)
        assert any("Could not read the context length of b; using AGENT_NUM_CTX (8192)" in n for n in _notices(app))


async def test_a_name_with_a_missing_tag_matches_the_latest_one(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "b:latest"], lengths={"b:latest": 1000}))
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

        assert scripted(app).model == "b:latest"


async def test_the_name_is_stripped(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "b"]))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model   b   ")
        await settled(app)

        assert scripted(app).model == "b"


async def test_an_unknown_model_changes_nothing_and_lists_the_installed_ones(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "b"]))
    async with app.run_test() as pilot:
        await _switch(app, pilot, "c")

        assert scripted(app).model == "fake"
        assert app.query_one(StatusBar).model == "fake"
        (error,) = [n for n in _notices(app) if n.startswith("✗")]
        assert "Model c is not installed. Installed: fake, b" in error
    assert not project_settings_path(tmp_path.resolve()).exists()


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


async def test_a_slow_lookup_leaves_the_input_usable_and_a_message_waits_for_it(make_app: AppFactory) -> None:
    catalog = FakeCatalog(["fake", "b"], delay=0.3)
    app = make_app(catalog=catalog)
    async with app.run_test() as pilot:
        await type_and_send(pilot, "/model b")
        await pilot.press("h", "i")
        assert app.query_one(ChatInput).text == "hi"  # still typing while Ollama answers
        await type_and_send(pilot, "")  # Enter while switching: refused, the text stays
        assert app.query_one(ChatInput).text == "hi"
        assert any("busy" in n for n in _notices(app))
        await settled(app)

        assert scripted(app).model == "b"


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


async def test_the_switch_keeps_the_history_and_the_session_file(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app([_reply("one")], catalog=FakeCatalog(["fake", "b"]))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await settled(app)
        before = list(app.agent.history)
        await _switch(app, pilot, "b")

        assert app.agent.history == before
    store = SessionStore(sessions_root(None), tmp_path.resolve())
    assert len(store.list()) == 1


async def test_the_task_child_and_instructions_follow_the_provider(make_app: AppFactory) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "b"], lengths={"b": 2000}))
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

        assert app.agent.provider is scripted(app)
        assert app.agent.provider.context_window == 2000


async def test_the_history_is_compacted_before_the_next_request_when_the_new_window_is_smaller(
    make_app: AppFactory, save_session
) -> None:
    long = "x" * 3000
    save_session([m for n in range(4) for m in chat(f"{n} {long}", f"a{n} {long}")])
    catalog = FakeCatalog(["fake", "b"], lengths={"b": 2000})
    app = make_app(
        [_reply("the summary"), _reply("answer")],
        context_window=8192,
        catalog=catalog,
        start=StartChoice("continue"),
        tools=False,
    )
    async with app.run_test() as pilot:
        await settled(app)
        await _switch(app, pilot, "b")
        await type_and_send(pilot, "next")
        await settled(app)

        provider = scripted(app)
        assert len(provider.requests) == 2
        assert provider.used[0] == ("b", 2000)  # the summary request already used the new model
        assert any("Context compacted" in n for n in _notices(app))
        messages, _ = provider.requests[-1]
        assert len(messages) < 10  # shorter than the 8 saved messages plus the new one plus the system prompt


async def test_no_compaction_when_the_history_still_fits(make_app: AppFactory, save_session) -> None:
    save_session([m for n in range(4) for m in chat(f"{n} {'x' * 3000}", f"a{n} {'x' * 3000}")])
    catalog = FakeCatalog(["fake", "b"], lengths={"b": 100000})
    app = make_app([_reply("answer")], context_window=8192, catalog=catalog, start=StartChoice("continue"), tools=False)
    async with app.run_test() as pilot:
        await settled(app)
        await _switch(app, pilot, "b")
        await type_and_send(pilot, "next")
        await settled(app)

        assert len(scripted(app).requests) == 1


# --- saving the choice -------------------------------------------------------------------------


async def test_the_choice_is_saved_in_the_project_file_next_to_the_permissions(make_app: AppFactory, tmp_path: Path) -> None:
    path = project_settings_path(tmp_path.resolve())
    path.parent.mkdir()
    path.write_text(json.dumps({"permissions": {"allow": ["bash(ls)"]}, "theme": "x"}))
    rules = _rules(tmp_path)
    rules.trust_project()  # the trust prompt is not what this test is about
    app = make_app(catalog=FakeCatalog(["fake", "b"]), permissions=rules)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

    assert json.loads(path.read_text()) == {"permissions": {"allow": ["bash(ls)"]}, "theme": "x", "model": "b"}


async def test_a_missing_project_file_is_created(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app(catalog=FakeCatalog(["fake", "b"]), permissions=_rules(tmp_path))
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

    assert json.loads(project_settings_path(tmp_path.resolve()).read_text()) == {"model": "b"}


async def test_a_trusted_project_stays_trusted_after_a_switch(make_app: AppFactory, tmp_path: Path) -> None:
    """The risk: a write that changes the file's bytes must not make its allow rules and hooks ask again."""
    workspace = tmp_path.resolve()
    path = project_settings_path(workspace)
    path.parent.mkdir()
    path.write_text(json.dumps({"permissions": {"allow": ["bash(ls)"]}, "hooks": {"Stop": [{"command": "true"}]}}))
    store = TrustStore(trust_store_path(name="permission-trust.json"))
    first = prepare(workspace, tmp_path / "none.json", store)
    assert first.rules.pending_trust
    first.rules.trust_project()  # the user said yes at an earlier launch
    app = make_app(catalog=FakeCatalog(["fake", "b"]), permissions=first.rules, permission_warnings=first.warnings)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

    assert json.loads(path.read_text())["model"] == "b"
    again = prepare(workspace, tmp_path / "none.json", store)  # the next launch
    assert again.rules.pending_trust is False
    assert again.rules.untrusted_allow == ()
    assert [r.text for r in again.rules.untrusted_allow] == []
    assert again.rules.allow("bash", {"command": "ls"}, None) is not None


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


async def test_a_symlinked_project_file_is_not_written_but_the_switch_applies(make_app: AppFactory, tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text('{"a": 1}')
    path = project_settings_path(tmp_path.resolve())
    path.parent.mkdir()
    path.symlink_to(real)
    workspace = tmp_path.resolve()
    rules = prepare(workspace, tmp_path / "none.json", TrustStore(trust_store_path(name="permission-trust.json"))).rules
    app = make_app(catalog=FakeCatalog(["fake", "b"]), permissions=rules)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

        assert scripted(app).model == "b"
        assert any("Model saved for the run only:" in n and "symlink" in n for n in _notices(app))
    assert real.read_text() == '{"a": 1}'


async def test_an_unparseable_project_file_is_left_alone_with_a_warning(make_app: AppFactory, tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    path = project_settings_path(workspace)
    path.parent.mkdir()
    path.write_text("{oops")
    rules = prepare(workspace, tmp_path / "none.json", TrustStore(trust_store_path(name="permission-trust.json"))).rules
    app = make_app(catalog=FakeCatalog(["fake", "b"]), permissions=rules)
    async with app.run_test() as pilot:
        await _switch(app, pilot, "b")

        assert scripted(app).model == "b"
        assert any("Model saved for the run only:" in n for n in _notices(app))
    assert path.read_text() == "{oops"


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


async def test_the_switch_is_recorded_in_the_session_and_replays_as_a_notice(make_app: AppFactory, tmp_path: Path) -> None:
    app = make_app([_reply("one")], catalog=FakeCatalog(["fake", "b"], lengths={"b": 4096}))
    async with app.run_test() as pilot:
        await type_and_send(pilot, "first")
        await settled(app)
        await _switch(app, pilot, "b")

    store = SessionStore(sessions_root(None), tmp_path.resolve())
    (info,) = store.list()
    loaded = store.load(info.id)
    assert [(i.kind, i.text) for i in loaded.items][-1] == ("notice", "Model: b (context 4,096)")
    assert [m.content for m in loaded.history] == ["first", "one"]


async def test_resuming_shows_the_model_line_and_does_not_change_the_model(make_app: AppFactory, tmp_path: Path) -> None:
    root = tmp_path.resolve()
    recorder = FileRecorder(workspace_dir(sessions_root(None), root), root)
    for message, meta in chat("q", "a"):
        recorder.message(message, meta)
    recorder.model("old-model", 2048)
    recorder.close()
    app = make_app(start=StartChoice("continue"), catalog=FakeCatalog(["fake", "old-model"]))
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert any(n.endswith("Model: old-model (context 2,048)") for n in _notices(app))
        assert (scripted(app).model, scripted(app).context_window) == ("fake", 1000)
        assert app.query_one(StatusBar).model == "fake"


# --- the command itself ------------------------------------------------------------------------


async def test_help_lists_model() -> None:
    assert "/model" in HELP


def test_a_custom_command_named_model_is_skipped_with_a_warning(tmp_path: Path) -> None:
    from src.engine.commands.loader import load_commands
    from src.tui.app import BUILTINS

    folder = tmp_path / ".agent" / "commands"
    folder.mkdir(parents=True)
    (folder / "model.md").write_text("hello")
    loaded = load_commands(tmp_path, set(BUILTINS), {"XDG_CONFIG_HOME": str(tmp_path / "cfg")})

    assert "model" not in loaded.by_name
    assert any("model" in w for w in loaded.warnings)


# --- review fixes ------------------------------------------------------------------------------------


async def test_a_project_file_changed_on_disk_after_launch_is_not_trusted_by_a_switch(
    make_app: AppFactory, tmp_path: Path
) -> None:
    """AC8: the new bytes hold more than the user saw, so the switch must not vouch for them."""
    workspace = tmp_path.resolve()
    path = project_settings_path(workspace)
    path.parent.mkdir()
    path.write_text(json.dumps({"permissions": {"allow": ["bash(ls)"]}}))
    store = TrustStore(trust_store_path(name="permission-trust.json"))
    first = prepare(workspace, tmp_path / "none.json", store)
    first.rules.trust_project()
    app = make_app(catalog=FakeCatalog(["fake", "b"]), permissions=first.rules)
    async with app.run_test() as pilot:
        await settled(app)
        path.write_text(json.dumps({"permissions": {"allow": ["bash(ls)", "bash(*)"]}}))  # edited behind our back
        await _switch(app, pilot, "b")

    written = path.read_bytes()
    assert json.loads(written)["model"] == "b"
    assert not store.is_trusted(workspace, written)
    again = prepare(workspace, tmp_path / "none.json", store)
    assert again.rules.pending_trust is True


async def test_launch_notices_are_shown_as_info_lines_with_no_prompt(make_app: AppFactory) -> None:
    app = make_app(launch_notices=["Model z (from .agent/settings.json)"])
    async with app.run_test() as pilot:
        await settled(app)
        await pilot.pause()

        assert any(n.endswith("Model z (from .agent/settings.json)") for n in _notices(app))
        assert not isinstance(app.screen, PermissionTrustScreen)


async def test_the_status_bar_writes_out_control_characters_of_the_model_name(make_app: AppFactory) -> None:
    app = make_app(model="b\x1b[2J\u202e")
    async with app.run_test():
        await settled(app)

        shown = app.query_one(StatusBar).model
        assert "\x1b" not in shown
        assert "\u202e" not in shown
        assert "\\x1b" in shown
