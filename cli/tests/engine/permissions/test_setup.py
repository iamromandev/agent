"""`prepare` and `Rules`: what a launch reads, which project rules are active, and what `/permissions` says."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from src.config import get_settings
from src.engine.mcp.trust import TrustStore, trust_store_path
from src.engine.permissions import setup as permission_setup
from src.engine.permissions.config import project_settings_path
from src.engine.permissions.setup import Rules, RuleSavedNotTrusted, prepare


class Launch:
    def __init__(self, tmp_path: Path) -> None:
        self.workspace = tmp_path / "ws"
        self.workspace.mkdir()
        self.global_path = tmp_path / "config" / "settings.json"
        self.store = TrustStore(tmp_path / "data" / "permission-trust.json")
        self.project_path = project_settings_path(self.workspace)

    def write(self, path: Path, allow: object = None, deny: object = None, **extra: object) -> bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        permissions = {k: v for k, v in (("allow", allow), ("deny", deny)) if v is not None}
        raw = json.dumps({"permissions": permissions, **extra}).encode()
        path.write_bytes(raw)
        return raw

    def project(self, allow: list[str] | None = None, deny: list[str] | None = None) -> bytes:
        return self.write(self.project_path, allow, deny)

    def prepare(self):
        return prepare(self.workspace, self.global_path, self.store)


@pytest.fixture
def launch(tmp_path: Path) -> Launch:
    return Launch(tmp_path)


def _allow(rules: Rules, tool: str, **args: object) -> str | None:
    verdict = rules.allow(tool, args, None)
    return verdict.rule.text if verdict else None


def _deny(rules: Rules, tool: str, **args: object) -> str | None:
    verdict = rules.deny(tool, args, None)
    return verdict.rule.text if verdict else None


def test_without_any_file_there_are_no_rules_and_no_warnings(launch: Launch) -> None:
    setup = launch.prepare()

    assert setup.warnings == ()
    assert _allow(setup.rules, "bash", command="ls") is None
    assert _deny(setup.rules, "bash", command="ls") is None
    assert setup.rules.listing() == []
    assert setup.rules.untrusted_allow == ()


def test_global_rules_are_active_without_any_trust(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(make test)"], deny=["bash(rm*)"])
    rules = launch.prepare().rules

    assert _allow(rules, "bash", command="make test") == "bash(make test)"
    assert _deny(rules, "bash", command="rm x") == "bash(rm*)"
    assert rules.allow("bash", {"command": "make test"}, None).source == "global"  # type: ignore[union-attr]
    assert rules.deny("bash", {"command": "rm x"}, None).source == "global"  # type: ignore[union-attr]


def test_a_deny_rule_refuses_a_chained_or_padded_command_with_the_same_rule(launch: Launch) -> None:
    launch.write(launch.global_path, deny=["bash(rm*)"])
    rules = launch.prepare().rules

    assert _deny(rules, "bash", command="echo hi; rm x") == "bash(rm*)"
    assert _deny(rules, "bash", command="   rm x") == "bash(rm*)"


def test_an_allow_rule_does_not_approve_a_chained_command(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(git status*)"])
    rules = launch.prepare().rules

    assert _allow(rules, "bash", command="git status --short") == "bash(git status*)"
    assert _allow(rules, "bash", command="git status =(touch x)") is None


def test_an_untrusted_project_allow_rule_is_inactive_but_its_deny_rule_applies(launch: Launch) -> None:
    launch.project(allow=["bash"], deny=["bash(rm*)"])
    setup = launch.prepare()

    assert _allow(setup.rules, "bash", command="ls") is None
    assert _deny(setup.rules, "bash", command="rm x") == "bash(rm*)"
    assert [r.text for r in setup.rules.untrusted_allow] == ["bash"]


def test_a_project_file_with_only_deny_rules_needs_no_trust(launch: Launch) -> None:
    launch.project(deny=["bash(rm*)"])

    assert launch.prepare().rules.untrusted_allow == ()


def test_trusting_activates_the_project_allow_rules_and_remembers_the_bytes(launch: Launch) -> None:
    raw = launch.project(allow=["bash(ls)"])
    rules = launch.prepare().rules

    rules.trust_project()

    assert _allow(rules, "bash", command="ls") == "bash(ls)"
    assert rules.untrusted_allow == ()
    assert launch.store.is_trusted(launch.workspace, raw)
    again = launch.prepare().rules  # the next launch
    assert _allow(again, "bash", command="ls") == "bash(ls)"
    assert again.untrusted_allow == ()


def test_editing_a_trusted_project_file_makes_it_untrusted_again(launch: Launch) -> None:
    launch.project(allow=["bash(ls)"])
    launch.prepare().rules.trust_project()

    launch.project(allow=["bash(ls)", "bash"])

    rules = launch.prepare().rules
    assert _allow(rules, "bash", command="anything") is None
    assert [r.text for r in rules.untrusted_allow] == ["bash(ls)", "bash"]


def test_trust_is_per_folder(launch: Launch, tmp_path: Path) -> None:
    launch.project(allow=["bash(ls)"])
    launch.prepare().rules.trust_project()
    other = tmp_path / "other"
    other.mkdir()
    project_settings_path(other).parent.mkdir()
    project_settings_path(other).write_bytes(launch.project_path.read_bytes())  # the very same bytes

    rules = prepare(other, launch.global_path, launch.store).rules

    assert [r.text for r in rules.untrusted_allow] == ["bash(ls)"]


def test_deny_wins_whatever_the_source_and_both_lists_are_always_checked(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash"])
    launch.project(deny=["bash(rm*)"])
    rules = launch.prepare().rules

    verdict = rules.deny("bash", {"command": "rm x"}, None)
    assert verdict is not None
    assert (verdict.rule.text, verdict.source) == ("bash(rm*)", "project")
    assert _allow(rules, "bash", command="rm x") == "bash"  # the loop asks deny first


def test_a_global_deny_applies_to_a_project_allow(launch: Launch) -> None:
    launch.write(launch.global_path, deny=["bash(rm*)"])
    launch.project(allow=["bash"])
    rules = launch.prepare().rules
    rules.trust_project()

    verdict = rules.deny("bash", {"command": "rm x"}, None)
    assert verdict is not None
    assert verdict.source == "global"


def test_a_store_inside_the_workspace_is_refused_with_a_warning(launch: Launch) -> None:
    raw = launch.project(allow=["bash"])
    inside = TrustStore(launch.workspace / "data" / "permission-trust.json")
    inside.path.parent.mkdir()
    inside.path.write_text(json.dumps({str(launch.workspace): hashlib.sha256(raw).hexdigest()}))

    setup = prepare(launch.workspace, launch.global_path, inside)

    assert [r.text for r in setup.rules.untrusted_allow] == ["bash"]
    assert any("inside the workspace" in w for w in setup.warnings)
    with pytest.raises(OSError, match="inside the workspace"):
        setup.rules.trust_project()
    assert _allow(setup.rules, "bash", command="x") == "bash"  # still active for this launch (the caller warns)


def test_a_corrupt_store_trusts_nothing(launch: Launch) -> None:
    launch.project(allow=["bash"])
    launch.store.path.parent.mkdir(parents=True)
    launch.store.path.write_bytes(b"{not json")

    assert [r.text for r in launch.prepare().rules.untrusted_allow] == ["bash"]


def test_the_store_does_not_follow_the_workspace_dot_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `.env` in the workspace names a store that already trusts the project file: `prepare` must not use it."""
    workspace = tmp_path / "ws"
    (workspace / ".agent").mkdir(parents=True)
    raw = json.dumps({"permissions": {"allow": ["bash"]}}).encode()
    (workspace / ".agent" / "settings.json").write_bytes(raw)
    planted = tmp_path / "planted"
    (planted / "agent").mkdir(parents=True)
    (planted / "agent" / "permission-trust.json").write_text(
        json.dumps({str(workspace): hashlib.sha256(raw).hexdigest()})
    )
    (workspace / ".env").write_text(f"XDG_DATA_HOME={planted}\nAGENT_SESSIONS_DIR={planted}\nAGENT_XDG_DATA_HOME={planted}\n")
    monkeypatch.chdir(workspace)
    get_settings.cache_clear()
    try:
        store = TrustStore(trust_store_path(name="permission-trust.json"))
        rules = prepare(workspace, tmp_path / "none.json", store).rules
    finally:
        get_settings.cache_clear()

    assert store.path == Path(os.environ["XDG_DATA_HOME"]) / "agent" / "permission-trust.json"
    assert not store.path.is_relative_to(planted)
    assert [r.text for r in rules.untrusted_allow] == ["bash"]


def test_the_permission_store_is_not_the_mcp_store(tmp_path: Path) -> None:
    env = {"XDG_DATA_HOME": str(tmp_path)}
    assert trust_store_path(env, name="permission-trust.json") != trust_store_path(env)


def test_an_invalid_rule_is_one_warning_naming_the_file_and_the_rule_and_the_rest_still_apply(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(a*b)", "bash()", "bash(ok)"])
    setup = launch.prepare()

    invalid = [w for w in setup.warnings if str(launch.global_path) in w]
    assert len(invalid) == 2
    assert any("bash(a*b)" in w for w in invalid)
    assert any("bash()" in w for w in invalid)
    assert _allow(setup.rules, "bash", command="ok") == "bash(ok)"


def test_a_global_file_with_bad_json_is_a_warning_and_adds_no_rules(launch: Launch) -> None:
    launch.global_path.parent.mkdir(parents=True)
    launch.global_path.write_text("{oops")

    setup = launch.prepare()

    assert len(setup.warnings) == 1
    assert str(launch.global_path) in setup.warnings[0]
    assert _allow(setup.rules, "bash", command="x") is None


def test_a_project_file_with_bad_json_warns_that_its_deny_rules_are_not_applied_either(launch: Launch) -> None:
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.write_text("{oops")

    setup = launch.prepare()

    assert len(setup.warnings) == 1
    assert "deny rules are not applied" in setup.warnings[0]
    assert setup.rules.untrusted_allow == ()


def test_a_symlinked_project_file_is_a_warning_and_adds_no_rules(launch: Launch, tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    launch.write(real, allow=["bash"], deny=["bash(rm*)"])
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.symlink_to(real)

    setup = launch.prepare()

    assert any("symlink" in w for w in setup.warnings)
    assert _deny(setup.rules, "bash", command="rm x") is None
    assert setup.rules.untrusted_allow == ()


def test_a_symlinked_global_file_is_followed(launch: Launch, tmp_path: Path) -> None:
    real = tmp_path / "dotfiles" / "settings.json"
    launch.write(real, allow=["bash(ls)"])
    launch.global_path.parent.mkdir(parents=True)
    launch.global_path.symlink_to(real)

    assert _allow(launch.prepare().rules, "bash", command="ls") == "bash(ls)"


@pytest.mark.parametrize("rule", ["web_fetch(https://docs.python.org*)", "web_fetch(https://docs.python.org:8080*)"])
def test_a_web_fetch_prefix_without_a_slash_after_the_host_warns_but_still_applies(launch: Launch, rule: str) -> None:
    launch.write(launch.global_path, allow=[rule])
    setup = launch.prepare()

    assert any("web_fetch" in w and "/" in w for w in setup.warnings)
    assert len(setup.rules.listing()) > 0


def test_a_web_fetch_prefix_that_ends_with_a_slash_does_not_warn(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["web_fetch(https://docs.python.org/*)", "web_fetch(https://a.org/x)"])

    assert launch.prepare().warnings == ()


def test_the_listing_shows_every_rule_with_its_source_inactive_ones_invalid_ones_and_the_trust_state(
    launch: Launch,
) -> None:
    launch.write(launch.global_path, allow=["bash(make test)", "bash(a*b)"])
    launch.project(allow=["bash"], deny=["bash(rm*)"])

    lines = launch.prepare().rules.listing()

    assert lines == [
        "allow bash(make test) (global)",
        "invalid bash(a*b) (global): * only at the end",
        "allow bash (project) inactive: untrusted",
        "deny bash(rm*) (project)",
        "project file: untrusted",
    ]


def test_the_listing_marks_the_project_file_trusted_once_it_is(launch: Launch) -> None:
    launch.project(allow=["bash"])
    rules = launch.prepare().rules
    rules.trust_project()

    assert rules.listing() == ["allow bash (project)", "project file: trusted"]


def test_the_listing_reports_a_file_that_could_not_be_used(launch: Launch) -> None:
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.write_text("{oops")
    launch.write(launch.global_path, allow=["bash"])

    lines = launch.prepare().rules.listing()

    assert lines[0] == "allow bash (global)"
    assert any(line.startswith("error (project):") for line in lines)
    assert lines[-1] == "project file: unusable"


def test_the_empty_rules_decide_nothing_and_cannot_save() -> None:
    rules = Rules.empty()

    assert rules.deny("bash", {"command": "x"}, None) is None
    assert rules.allow("bash", {"command": "x"}, None) is None
    assert rules.listing() == []
    assert rules.can_save is False
    with pytest.raises(OSError):
        rules.save_project("bash")


def test_a_prepared_launch_can_save(launch: Launch) -> None:
    assert launch.prepare().rules.can_save is True


# Always here


def _saved(launch: Launch) -> dict:
    return json.loads(launch.project_path.read_text())


def test_saving_adds_the_rule_to_the_file_trusts_the_new_bytes_and_applies_it_at_once(launch: Launch) -> None:
    rules = launch.prepare().rules

    rules.save_project("bash(uv run pytest -q)")

    assert _saved(launch)["permissions"]["allow"] == ["bash(uv run pytest -q)"]
    assert _allow(rules, "bash", command="uv run pytest -q") == "bash(uv run pytest -q)"
    assert launch.store.is_trusted(launch.workspace, launch.project_path.read_bytes())
    again = launch.prepare()  # the next launch: no trust prompt, the rule applies
    assert again.rules.untrusted_allow == ()
    assert _allow(again.rules, "bash", command="uv run pytest -q") == "bash(uv run pytest -q)"


def test_saving_next_to_trusted_rules_keeps_them_and_other_keys(launch: Launch) -> None:
    launch.write(launch.project_path, allow=["bash(ls)"], deny=["bash(rm*)"], theme="x")
    rules = launch.prepare().rules
    rules.trust_project()

    rules.save_project("write")

    data = _saved(launch)
    assert data["permissions"] == {"allow": ["bash(ls)", "write"], "deny": ["bash(rm*)"]}
    assert data["theme"] == "x"
    assert launch.prepare().rules.untrusted_allow == ()


def test_saving_does_not_trust_rules_of_an_untrusted_file(launch: Launch) -> None:
    """A cloned repo's `bash(*)` must not be trusted just because the user chose "Always here" for one call."""
    launch.project(allow=["bash(*)"])
    rules = launch.prepare().rules  # untrusted, the user declined

    rules.save_project("bash(ls)")

    assert _allow(rules, "bash", command="ls") == "bash(ls)"  # the saved rule works now
    assert _allow(rules, "bash", command="rm x") is None  # the repo's own rule stays off
    again = launch.prepare().rules
    assert [r.text for r in again.untrusted_allow] == ["bash(*)", "bash(ls)"]  # asked again next launch


def test_saving_after_the_file_was_edited_during_the_run_does_not_trust_the_edit(launch: Launch) -> None:
    launch.project(allow=["bash(ls)"])
    rules = launch.prepare().rules
    rules.trust_project()
    launch.project(allow=["bash(ls)", "bash"])  # edited by someone (the model, a script) after launch

    rules.save_project("write")

    assert [r.text for r in launch.prepare().rules.untrusted_allow] == ["bash(ls)", "bash", "write"]


def test_saving_the_same_rule_twice_adds_no_duplicate(launch: Launch) -> None:
    rules = launch.prepare().rules

    rules.save_project("bash(ls)")
    rules.save_project("bash(ls)")

    assert _saved(launch)["permissions"]["allow"] == ["bash(ls)"]
    assert rules.listing().count("allow bash(ls) (project)") == 1


def test_a_file_that_cannot_be_parsed_is_never_overwritten(launch: Launch) -> None:
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.write_text("{oops")
    rules = launch.prepare().rules

    with pytest.raises(ValueError, match="not valid JSON"):
        rules.save_project("bash(ls)")

    assert launch.project_path.read_text() == "{oops"
    assert _allow(rules, "bash", command="ls") is None


def test_a_symlinked_project_file_is_not_written_through(launch: Launch, tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    before = launch.write(real, allow=["bash(x)"])
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.symlink_to(real)
    rules = launch.prepare().rules

    with pytest.raises(OSError, match="symlink"):
        rules.save_project("bash(ls)")

    assert real.read_bytes() == before


def test_a_failing_trust_store_still_activates_the_rule_and_reports_the_error(launch: Launch) -> None:
    inside = TrustStore(launch.workspace / "trust.json")
    rules = prepare(launch.workspace, launch.global_path, inside).rules

    with pytest.raises(OSError, match="inside the workspace"):
        rules.save_project("bash(ls)")

    assert _allow(rules, "bash", command="ls") == "bash(ls)"
    assert _saved(launch)["permissions"]["allow"] == ["bash(ls)"]


def test_a_saved_rule_does_not_trust_a_file_changed_between_launch_and_the_save_itself(
    launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The change lands after `Rules` looked and before `add_allow` read: only add_allow's own read counts."""
    launch.project(allow=["bash(ls)"])
    rules = launch.prepare().rules
    rules.trust_project()
    real = permission_setup.add_allow

    def edited_first(path: Path, rule_text: str) -> tuple[bytes | None, bytes]:
        launch.project(allow=["bash(ls)", "bash"])  # someone else's edit, in that window
        return real(path, rule_text)

    monkeypatch.setattr(permission_setup, "add_allow", edited_first)

    rules.save_project("write")

    assert not launch.store.is_trusted(launch.workspace, launch.project_path.read_bytes())
    assert [r.text for r in launch.prepare().rules.untrusted_allow] == ["bash(ls)", "bash", "write"]


def test_a_trust_store_that_fails_after_the_file_was_written_is_reported_as_exactly_that(launch: Launch) -> None:
    rules = prepare(launch.workspace, launch.global_path, TrustStore(launch.workspace / "trust.json")).rules

    with pytest.raises(RuleSavedNotTrusted) as caught:
        rules.save_project("bash(ls)")

    message = str(caught.value)
    assert "saved to .agent/settings.json" in message
    assert "could not be remembered as trusted" in message
    assert "next launch will ask" in message
    assert _saved(launch)["permissions"]["allow"] == ["bash(ls)"]
    assert _allow(rules, "bash", command="ls") == "bash(ls)"


# text from the project file is escaped wherever it is shown


def test_the_listing_escapes_rule_text_and_reasons_from_the_file(launch: Launch) -> None:
    launch.project(allow=["bash(\x1b[2Jevil\nfake line*)", "bad\x1b[31m rule"], deny=["bash(rm\x1b]0;x\x07*)"])
    launch.prepare().rules.trust_project()

    lines = launch.prepare().rules.listing()

    assert all(line.isprintable() for line in lines)
    assert any("\\x1b[2Jevil\\x0afake line*" in line for line in lines)
    assert any(line.startswith("invalid bad\\x1b[31m rule") for line in lines)


def test_the_listing_escapes_the_error_of_an_unreadable_file(launch: Launch) -> None:
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.write_text("{\x1b[2J")

    lines = launch.prepare().rules.listing()

    assert all(line.isprintable() for line in lines)


def test_warnings_escape_the_rule_text_and_the_path(launch: Launch) -> None:
    launch.project(allow=["bash(a\x1b[2J*b)"])
    launch.project_path.write_text(json.dumps({"permissions": {"allow": ["bash(a\x1b[2J*b)", "web_fetch(https://a.org\n*)"]}}))

    warnings = launch.prepare().warnings

    assert warnings
    assert all(w.isprintable() for w in warnings)
    assert any("\\x1b[2J" in w for w in warnings)


def test_a_bad_list_is_a_warning_naming_the_list_and_the_file_and_the_other_list_still_applies(launch: Launch) -> None:
    launch.write(launch.global_path, allow="bash", deny=["bash(rm*)"])

    setup = launch.prepare()

    (warning,) = setup.warnings
    assert str(launch.global_path) in warning
    assert "permissions.allow" in warning
    assert _deny(setup.rules, "bash", command="rm x") == "bash(rm*)"
    assert any(line.startswith("error (global): ") and "permissions.allow" in line for line in setup.rules.listing())


def test_a_project_file_with_a_bad_allow_list_keeps_its_deny_rules_and_asks_for_no_trust(launch: Launch) -> None:
    launch.write(launch.project_path, allow=5, deny=["bash(rm*)"])

    rules = launch.prepare().rules

    assert _deny(rules, "bash", command="rm x") == "bash(rm*)"
    assert rules.untrusted_allow == ()


def test_many_warnings_are_cut_to_twenty_and_a_count(launch: Launch) -> None:
    launch.write(launch.global_path, allow=[f"bash(a*{i}*)" for i in range(45)])

    warnings = launch.prepare().warnings

    assert len(warnings) == 21
    assert warnings[-1] == "and 25 more warnings"


def test_exactly_twenty_warnings_are_all_shown(launch: Launch) -> None:
    launch.write(launch.global_path, allow=[f"bash(a*{i}*)" for i in range(20)])

    warnings = launch.prepare().warnings

    assert len(warnings) == 20
    assert not any("more" in w for w in warnings)


# a valued rule for a tool without a key argument


KEYS = {"bash": "command", "web_fetch": "url", "read": None, "write": None, "edit": None}


@pytest.mark.parametrize("rule", ["write(a.py)", "edit(src/*)", "read(secret.txt)", "mcp__gh__issues(x)"])
def test_a_valued_rule_for_a_tool_without_a_key_argument_warns_that_it_never_applies(launch: Launch, rule: str) -> None:
    launch.write(launch.global_path, allow=[rule])

    setup = prepare(launch.workspace, launch.global_path, launch.store, KEYS)

    (warning,) = setup.warnings
    assert warning == f"{launch.global_path}: {rule}: {rule.split('(')[0]} has no argument to match; this rule never applies"


def test_the_same_warning_for_a_deny_rule_and_a_project_rule(launch: Launch) -> None:
    launch.write(launch.global_path, deny=["read(secret.txt)"])
    launch.project(allow=["write(a.py)"])

    warnings = prepare(launch.workspace, launch.global_path, launch.store, KEYS).warnings

    assert len(warnings) == 2
    assert all("has no argument to match" in w for w in warnings)


@pytest.mark.parametrize("rule", ["write", "read", "bash(ls)", "bash(ls*)", "web_fetch(https://a.org/*)", "mcp__gh__*", "other(x)"])
def test_rules_that_can_match_do_not_warn(launch: Launch, rule: str) -> None:
    launch.write(launch.global_path, allow=[rule])

    assert prepare(launch.workspace, launch.global_path, launch.store, KEYS).warnings == ()


def test_without_the_tool_list_no_rule_is_checked_against_tools(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["write(a.py)"])

    assert launch.prepare().warnings == ()
