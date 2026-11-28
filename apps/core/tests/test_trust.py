from datetime import UTC, datetime

import pytest
from core.engine.configguard import HeldGlobal, MarkerInfo
from core.engine.configwatch import TrustRevoked
from core.engine.hooks.config import HookEntry
from core.engine.lsp.config import ServerSpec
from core.engine.mcp.config import ServerConfig
from core.engine.permissions.rules import parse_rule
from core.trust import apply_trust, pending_trust, trust_lines

HOOK = HookEntry("PreToolUse", "echo ok" + " " * 200 + "; curl x", "bash", 30.0, "project")
LSP = ServerSpec("my\x1bdsl", "/opt/bin/ls p", ("--stdio", "a" + " " * 5 + "b"), frozenset({".m"}), {}, "project")
SERVER = ServerConfig("db\x1b", "run", ("a b",), {"K": "v"}, "project", True)


class Guard:
    def __init__(self, error: OSError | None = None) -> None:
        self.error, self.calls = error, 0

    def trust_global(self) -> None:
        self.calls += 1
        if self.error:
            raise self.error


class Permissions(Guard):
    def trust_project(self) -> None:
        self.trust_global()


def test_lines_escape_odd_characters_and_collapse_space_runs() -> None:
    ask = trust_lines("permission_trust", rules=[parse_rule("bash(a‮b*)")], hooks=[HOOK])
    assert ask.lines[0] == "allow bash(a\\u202eb*)"
    assert ask.lines[1] == "hook PreToolUse [bash] (timeout 30s) echo ok\u2420\u00d7200; curl x"
    assert all(line.isascii() or "\u2420" in line for line in ask.lines)


def test_mcp_and_global_lines_keep_the_screens_texts() -> None:
    mcp = trust_lines("mcp_trust", servers=[SERVER])
    assert mcp.lines == ["db\\x1b  run 'a b'  (env: K)\n  overrides the global 'db\\x1b'"]
    marker = MarkerInfo(datetime(2026, 1, 2, tzinfo=UTC), ("a.json",))
    glob = trust_lines(
        "global_trust",
        rules=[parse_rule("bash(ls)"), parse_rule("read")],
        hooks=[HOOK],
        servers=[SERVER],
        marker=marker,
        deny=[parse_rule("bash(rm*)")],
        held=HeldGlobal(keep_env=("TOKEN",), skills=("s",)),
    )
    assert glob.lines[0] == "allow read"  # the broadest allow first
    assert glob.lines[-3:] == ["keepEnv TOKEN", "skill s", "deny rule bash(rm*) (applies either way)"]
    assert "a.json" in (glob.note or "")


def test_pending_trust_is_global_then_mcp_then_permissions() -> None:
    asks = pending_trust(
        global_waiting=True,
        marker=None,
        global_rules=[parse_rule("read")],
        global_hooks=[],
        global_servers=[],
        global_deny=[],
        held=HeldGlobal(),
        project_servers=[SERVER],
        project_rules=[parse_rule("bash")],
        project_hooks=[],
    )
    assert [a.kind for a in asks] == ["global_trust", "mcp_trust", "permission_trust"]
    assert pending_trust(
        global_waiting=False, marker=None, global_rules=[], global_hooks=[], global_servers=[], global_deny=[],
        held=HeldGlobal(), project_servers=[], project_rules=[], project_hooks=[],
    ) == []  # fmt: skip


def test_global_yes_trusts_starts_servers_and_says_so() -> None:
    guard = Guard()
    out = apply_trust("global_trust", True, guard=guard, servers=["a"], held_any=True)
    assert guard.calls == 1 and out.start == ["a"]
    assert out.notices == [
        (
            "info",
            "global config trusted: global allow rules, hooks, MCP servers and language servers are on; "
            "keepEnv and the global commands, skills, agents and workflows return at the next launch (they are read once at launch)",
        )
    ]


def test_global_no_changes_nothing() -> None:
    guard = Guard()
    out = apply_trust("global_trust", False, guard=guard, servers=["a"])
    assert guard.calls == 0 and out.start == []
    assert out.notices == [
        ("info", "global config not trusted: global allow rules, hooks, MCP servers and language servers are off for this run")
    ]


@pytest.mark.parametrize("error", [TrustRevoked("revoked"), OSError("could not remove the marker")])
def test_global_failure_is_one_warning_and_starts_nothing(error: OSError) -> None:
    out = apply_trust("global_trust", True, guard=Guard(error), servers=["a"])
    assert out.start == [] and out.notices == [("warn", f"global config: {error}")]


def test_mcp_no_names_the_servers_and_yes_starts_them() -> None:
    no = apply_trust("mcp_trust", False, servers=["a", "b"])
    assert no.notices == [("info", "mcp: project servers not trusted, so not started: a, b")] and no.start == []
    calls: list[int] = []
    yes = apply_trust("mcp_trust", True, servers=["a"], mcp_trust=lambda: calls.append(1))
    assert calls == [1] and yes.start == ["a"] and yes.notices == []


def test_mcp_trust_revoked_starts_nothing_but_oserror_still_starts() -> None:
    def revoked() -> None:
        raise TrustRevoked("gone")

    def broken() -> None:
        raise OSError("disk")

    assert apply_trust("mcp_trust", True, servers=["a"], mcp_trust=revoked).notices == [
        ("warn", f"mcp: {TrustRevoked('gone')}")
    ]
    out = apply_trust("mcp_trust", True, servers=["a"], mcp_trust=broken)
    assert out.start == ["a"] and out.notices == [("warn", "mcp: could not remember the trust: disk")]


@pytest.mark.parametrize(
    ("rules", "hooks", "text"),
    [
        (True, True, "project permission rules and hooks not trusted: allow rules and hooks are off for this run"),
        (False, True, "project hooks not trusted: hooks are off for this run"),
        (True, False, "project permission rules not trusted: allow rules are off for this run"),
    ],
)
def test_permission_no_texts(rules: bool, hooks: bool, text: str) -> None:
    perms = Permissions()
    out = apply_trust("permission_trust", False, permissions=perms, has_rules=rules, has_hooks=hooks)
    assert perms.calls == 0 and out.notices == [("info", text)]


def test_permission_yes_and_failures() -> None:
    perms = Permissions()
    assert apply_trust("permission_trust", True, permissions=perms).notices == [] and perms.calls == 1
    revoked = apply_trust("permission_trust", True, permissions=Permissions(TrustRevoked("r")))
    assert revoked.notices == [("warn", f"permissions: {TrustRevoked('r')}")]
    broken = apply_trust("permission_trust", True, permissions=Permissions(OSError("x")))
    assert broken.notices == [("warn", "permissions: could not remember the trust: x")]


def test_lsp_servers_are_listed_after_the_hooks_in_the_project_and_global_prompts() -> None:
    ask = trust_lines("permission_trust", rules=[parse_rule("bash(a)")], hooks=[HOOK], lsp=[LSP])
    assert ask.lines[-1] == "lsp my\\x1bdsl: '/opt/bin/ls p' --stdio 'a\u2420\u00d75b'"
    assert ask.lines[:2] == [ask.lines[0], ask.lines[1]] and ask.lines[1].startswith("hook ")
    assert "1 language server" in ask.title
    only = trust_lines("permission_trust", lsp=[LSP])
    assert only.lines == [ask.lines[-1]] and only.accept == "Trust these language servers"

    glob = trust_lines("global_trust", lsp=[LSP], marker=None)
    assert glob.lines == [ask.lines[-1]] and "1 language server" in glob.title


def test_pending_trust_asks_the_permission_prompt_for_lsp_alone() -> None:
    asks = pending_trust(
        global_waiting=False, marker=None, global_rules=[], global_hooks=[], global_servers=[], global_deny=[],
        held=HeldGlobal(), project_servers=[], project_rules=[], project_hooks=[], project_lsp=[LSP],
    )
    assert [a.kind for a in asks] == ["permission_trust"] and asks[0].lines[0].startswith("lsp ")


def test_an_lsp_line_shows_the_env_keys_like_the_mcp_prompt_does() -> None:
    spec = ServerSpec("e", "/bin/ls", (), frozenset({".m"}), {"LD_PRELOAD": "/tmp/x.so", "NODE_\x1bOPTIONS": "-r y"}, "project")
    assert trust_lines("permission_trust", lsp=[spec]).lines == ["lsp e: /bin/ls  (env: LD_PRELOAD, NODE_\\x1bOPTIONS)"]
    assert trust_lines("permission_trust", lsp=[LSP]).lines[0].endswith("'a\u2420\u00d75b'")  # no env: no note


@pytest.mark.parametrize(
    ("rules", "hooks", "text"),
    [
        (True, False, "project settings not trusted: allow rules, language servers are off for this run"),
        (False, True, "project settings not trusted: hooks, language servers are off for this run"),
        (True, True, "project settings not trusted: allow rules, hooks, language servers are off for this run"),
        (False, False, "project settings not trusted: language servers are off for this run"),
    ],
)
def test_declining_the_project_prompt_with_language_servers_names_what_stays_off(rules: bool, hooks: bool, text: str) -> None:
    out = apply_trust("permission_trust", False, has_rules=rules, has_hooks=hooks, has_lsp=True)
    assert out.notices == [("info", text)]


def test_the_global_prompt_footer_names_language_servers() -> None:
    ask = trust_lines("global_trust", lsp=[LSP], marker=None, held=HeldGlobal(commands=("c",)))
    assert ask.footer[-1].startswith("Trust turns the allow rules, hooks, MCP servers and language servers on now;")
