"""`prepare` and `Rules`: what a launch reads, which project rules are active, and what `/permissions` says."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from core.engine.mcp.trust import TrustStore
from core.engine.permissions import setup as permission_setup
from core.engine.permissions.config import project_settings_path
from core.engine.permissions.rules import parse_rule
from core.engine.permissions.setup import Rules, prepare


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


def test_an_allow_rule_does_not_approve_a_chained_command(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(git status*)"])
    rules = launch.prepare().rules

    assert _allow(rules, "bash", command="git status --short") == "bash(git status*)"
    assert _allow(rules, "bash", command="git status =(touch x)") is None


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


@pytest.mark.parametrize("rule", ["web_fetch(https://docs.python.org:8080*)"])
def test_a_web_fetch_prefix_without_a_slash_after_the_host_warns_but_still_applies(launch: Launch, rule: str) -> None:
    launch.write(launch.global_path, allow=[rule])
    setup = launch.prepare()

    assert any("web_fetch" in w and "/" in w for w in setup.warnings)
    assert len(setup.rules.listing()) > 0


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


# Always here


def _saved(launch: Launch) -> dict:
    return json.loads(launch.project_path.read_text())


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


def test_a_symlinked_project_file_is_not_written_through(launch: Launch, tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    before = launch.write(real, allow=["bash(x)"])
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.symlink_to(real)
    rules = launch.prepare().rules

    with pytest.raises(OSError, match="symlink"):
        rules.save_project("bash(ls)")

    assert real.read_bytes() == before


# save_project_key: the model choice must not cost the project its trust


def test_saving_a_key_after_the_file_was_edited_during_the_run_does_not_trust_the_edit(launch: Launch) -> None:
    launch.project(allow=["bash(ls)"])
    rules = launch.prepare().rules
    rules.trust_project()
    launch.project(allow=["bash(ls)", "bash"])  # edited after launch

    rules.save_project_key("model", "a")

    assert not launch.store.is_trusted(launch.workspace, launch.project_path.read_bytes())
    assert [r.text for r in launch.prepare().rules.untrusted_allow] == ["bash(ls)", "bash"]


def test_saving_a_key_never_overwrites_an_unparseable_or_symlinked_file(launch: Launch, tmp_path: Path) -> None:
    launch.project_path.parent.mkdir(parents=True)
    launch.project_path.write_text("{oops")
    with pytest.raises(ValueError, match="not valid JSON"):
        launch.prepare().rules.save_project_key("model", "a")
    assert launch.project_path.read_text() == "{oops"

    launch.project_path.unlink()
    real = tmp_path / "real.json"
    before = launch.write(real, allow=["bash(x)"])
    launch.project_path.symlink_to(real)
    with pytest.raises(OSError, match="symlink"):
        launch.prepare().rules.save_project_key("model", "a")
    assert real.read_bytes() == before


# text from the project file is escaped wherever it is shown


def test_a_bad_list_is_a_warning_naming_the_list_and_the_file_and_the_other_list_still_applies(launch: Launch) -> None:
    launch.write(launch.global_path, allow="bash", deny=["bash(rm*)"])

    setup = launch.prepare()

    (warning,) = setup.warnings
    assert str(launch.global_path) in warning
    assert "permissions.allow" in warning
    assert _deny(setup.rules, "bash", command="rm x") == "bash(rm*)"
    assert any(line.startswith("error (global): ") and "permissions.allow" in line for line in setup.rules.listing())


def test_many_warnings_are_cut_to_twenty_and_a_count(launch: Launch) -> None:
    launch.write(launch.global_path, allow=[f"bash(a*{i}*)" for i in range(45)])

    warnings = launch.prepare().warnings

    assert len(warnings) == 21
    assert warnings[-1] == "and 25 more warnings"


# a valued rule for a tool without a key argument


KEYS = {"bash": "command", "web_fetch": "url", "read": None, "write": None, "edit": None}


@pytest.mark.parametrize("rule", ["mcp__gh__issues(x)"])
def test_a_valued_rule_for_a_tool_without_a_key_argument_warns_that_it_never_applies(launch: Launch, rule: str) -> None:
    launch.write(launch.global_path, allow=[rule])

    setup = prepare(launch.workspace, launch.global_path, launch.store, KEYS)

    (warning,) = setup.warnings
    assert warning == f"{launch.global_path}: {rule}: {rule.split('(')[0]} has no argument to match; this rule never applies"


HOOKS = {"hooks": {"PreToolUse": [{"command": "echo hi"}]}}


def test_a_run_allow_rule_is_active_but_never_saved_or_remembered(launch: Launch) -> None:
    rules = launch.prepare().rules

    rules.add_run_allow([parse_rule("bash(make test)")])

    assert _allow(rules, "bash", command="make test") == "bash(make test)"
    assert _allow(rules, "bash", command="rm x") is None
    assert not launch.project_path.exists()
    assert rules.allow("bash", {"command": "make test"}, "command").source == "run"


def test_the_listing_shows_run_rules_as_run(launch: Launch) -> None:
    launch.write(launch.global_path, allow=["bash(ls)"])
    rules = launch.prepare().rules
    rules.add_run_allow([parse_rule("edit")])

    assert rules.listing() == ["allow bash(ls) (global)", "allow edit (run)", "project file: none"]


def test_run_rule_warnings_name_unknown_tools_and_valued_rules_on_keyless_tools() -> None:
    keys = {"bash": "command", "read": None, "write": None}
    rules = [parse_rule(t) for t in ("bash(ls)", "read(x)", "nosuch", "mcp__gh__*", "write")]

    warnings = permission_setup.run_rule_warnings(rules, keys)

    assert warnings == [
        "--allow read(x): read has no argument to match; this rule never applies",
        "--allow nosuch: no tool named nosuch; this rule never applies",
    ]
