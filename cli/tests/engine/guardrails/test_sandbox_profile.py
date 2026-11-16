"""The sandbox profile text and `make_sandbox` (R11, R13). Pure text: no `sandbox-exec` needed, so every platform runs it."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.engine.guardrails import sandbox as sandbox_module
from src.engine.guardrails.config import GuardConfig
from src.engine.guardrails.paths import ProtectedPaths
from src.engine.guardrails.sandbox import SANDBOX_EXEC, Sandbox, build_profile, make_sandbox


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    return folder.resolve()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    folder = tmp_path / "ws"
    folder.mkdir()
    return folder.resolve()


def profile(workspace: Path, home: Path, *, writable: list[Path] | None = None, deny_write: list[Path] | None = None,
            extra: tuple[str, ...] = (), network: bool = True) -> str:
    return build_profile(workspace, writable if writable is not None else [workspace], deny_write or [],
                         ProtectedPaths(home, extra), network)


def index(text: str, needle: str) -> int:
    assert needle in text, f"{needle!r} not in the profile"
    return text.index(needle)


def test_wrap_puts_sandbox_exec_and_the_profile_in_front() -> None:
    sandbox = Sandbox("(version 1)", "on", True, ())

    assert sandbox.wrap(["sh", "-c", "true"]) == [str(SANDBOX_EXEC), "-p", "(version 1)", "sh", "-c", "true"]


def test_wrap_without_a_profile_leaves_the_command_alone() -> None:
    assert Sandbox(None, "off", True, ()).wrap(["sh", "-c", "true"]) == ["sh", "-c", "true"]


def test_the_profile_starts_with_allow_default_and_denies_writes(workspace: Path, home: Path) -> None:
    text = profile(workspace, home)

    assert text.startswith("(version 1)\n(allow default)\n(deny file-write*)\n")


def test_writable_roots_and_devices_are_allowed(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, writable=[workspace, home / ".cache"])

    allow = next(line for line in text.splitlines() if line.startswith("(allow file-write*"))
    assert f'(subpath "{workspace}")' in allow
    assert f'(subpath "{home / ".cache"}")' in allow
    for device in ('(literal "/dev/null")', '(literal "/dev/stdout")', '(literal "/dev/stderr")', '(literal "/dev/ptmx")',
                   '(literal "/dev/tty")', '(regex #"^/dev/fd/")', '(regex #"^/dev/ttys[0-9]+$")', '(regex #"^/dev/pty")'):
        assert device in allow


def test_only_the_pseudo_terminal_devices_are_writable_not_every_tty_name(workspace: Path, home: Path) -> None:
    allow = next(line for line in profile(workspace, home).splitlines() if line.startswith("(allow file-write*"))

    assert '(regex #"^/dev/tty")' not in allow  # it would also match /dev/tty.usbserial-* and /dev/tty.Bluetooth-*


def test_a_symlinked_writable_root_is_written_as_its_real_path(workspace: Path, home: Path, tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(home / "target-cache", target_is_directory=False)

    text = profile(workspace, home, writable=[workspace, link])

    assert f'(subpath "{home / "target-cache"}")' in text
    assert f'(subpath "{link}")' not in text


def test_quote_and_backslash_in_a_path_are_escaped_in_a_string(tmp_path: Path, home: Path) -> None:
    odd = tmp_path / 'we"ird\\ws'
    odd.mkdir()

    text = profile(odd.resolve(), home)

    assert f'(subpath "{str(odd.resolve()).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}")' in text
    assert 'we"ird' not in text


def test_write_denials_come_after_the_allows_and_cover_the_agent_files(workspace: Path, home: Path) -> None:
    config = home / ".config" / "agent"
    settings = workspace / ".agent" / "settings.json"

    text = profile(workspace, home, deny_write=[config, settings])

    deny = next(line for line in text.splitlines() if line.startswith("(deny file-write* (subpath"))
    assert f'(subpath "{config}")' in deny
    assert f'(subpath "{settings}")' in deny
    assert index(text, "(allow file-write*") < index(text, "(deny file-write* (subpath")


def test_the_workspace_and_agent_folder_are_denied_only_for_removal_and_rename(workspace: Path, home: Path) -> None:
    text = profile(workspace, home)

    unlink = next(line for line in text.splitlines() if line.startswith("(deny file-write-unlink (literal"))
    assert f'(literal "{workspace}")' in unlink  # `mv <workspace>` away and back must fail
    assert f'(literal "{workspace / ".agent"}")' in unlink
    assert f'(literal "{workspace}")' not in "".join(
        line for line in text.splitlines() if line.startswith("(deny file-write* ")
    )  # `touch .`, `tar -x` and `cp -R x/. .` must work
    assert index(text, "(allow file-write*") < index(text, "(deny file-write-unlink (literal")


def test_the_agent_commands_and_skills_folders_are_denied_for_writing(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, deny_write=[workspace / ".agent" / "commands", workspace / ".agent" / "skills"])

    deny = next(line for line in text.splitlines() if line.startswith("(deny file-write* (subpath"))
    assert f'(subpath "{workspace / ".agent" / "commands"}")' in deny
    assert f'(subpath "{workspace / ".agent" / "skills"}")' in deny


def test_protected_home_roots_are_denied_for_writing_in_every_spelling_after_the_allows(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, writable=[workspace, home], extra=("~/secrets/**",))

    for root in (home / ".ssh", home / ".docker" / "config.json", home / "secrets"):
        for spelling in (root, Path("/System/Volumes/Data") / str(root).lstrip("/")):
            line = f'(deny file-write* (subpath "{spelling}"))'
            assert line in text
            assert index(text, "(allow file-write*") < index(text, line)


def test_protected_names_and_globs_are_denied_for_writing_except_the_templates(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=("*.token", "secrets/**", "~/logs/*.secret"))

    write = [line for line in text.splitlines() if line.startswith("(deny file-write* (require-all")]
    assert len(write) == 1
    assert '#"/\\.[eE][nN][vV]$"' in write[0]
    assert '#"/[^/]*\\.[pP][eE][mM]$"' in write[0]
    assert "(require-not (regex" in write[0] and "[eE][xX][aA][mM][pP][lL][eE]$" in write[0]
    glob = next(line for line in text.splitlines() if line.startswith("(deny file-write* (regex"))
    assert '#"^(.*/)?[sS][eE][cC][rR][eE][tT][sS]/.*$"' in glob
    assert '[^/]*\\.[sS][eE][cC][rR][eE][tT]$"' in glob
    assert '#"/[^/]*\\.[tT][oO][kK][eE][nN]$"' in glob  # a settings name has no template exception


def test_removing_or_renaming_a_protected_name_or_folder_is_denied(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=("secrets/**",))

    unlink = [line for line in text.splitlines() if line.startswith("(deny file-write-unlink (regex")]
    assert len(unlink) == 1
    assert '#"^(.*/)?[sS][eE][cC][rR][eE][tT][sS]$"' in unlink[0]  # the folder itself, not only what is in it
    assert '#"^/System/Volumes/Data' not in unlink[0]  # relative patterns have no firmlink spelling


def test_the_folder_of_an_absolute_wildcard_pattern_is_denied_for_removal_with_its_firmlink_spelling(
    workspace: Path, home: Path
) -> None:
    text = profile(workspace, home, extra=("~/vault/*/keys/**",))

    unlink = next(line for line in text.splitlines() if line.startswith("(deny file-write-unlink (regex"))
    base = "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in f"{home}/vault/")
    body = f'{base.replace(".", chr(92) + ".")}[^/]*/[kK][eE][yY][sS]$"'
    assert f'#"^{body}' in unlink
    assert f'#"^/System/Volumes/Data{body}' in unlink


def test_denials_are_also_written_with_the_macos_firmlink_prefix(workspace: Path, home: Path) -> None:
    text = profile(workspace, home)

    assert f'(subpath "{home / ".ssh"}")' in text
    assert f'(subpath "/System/Volumes/Data{home / ".ssh"}")' in text


def test_protected_folders_and_files_are_denied_for_reading(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=("~/secrets/**",))

    for root in (home / ".ssh", home / ".aws", home / ".netrc", home / ".docker" / "config.json", home / "secrets"):
        assert f'(deny file-read* (subpath "{root}"))' in text


def test_built_in_names_are_denied_by_regex_ignoring_case(workspace: Path, home: Path) -> None:
    text = profile(workspace, home)

    assert '(deny file-read* (regex ' in text
    assert '#"/\\.[eE][nN][vV]$"' in text
    assert '#"/\\.[eE][nN][vV]\\.[^/]*$"' in text
    assert '#"/[^/]*\\.[pP][eE][mM]$"' in text
    assert '#"/[iI][dD]_[rR][sS][aA][^/]*$"' in text


def test_the_env_templates_are_allowed_again_after_the_name_denial(workspace: Path, home: Path) -> None:
    text = profile(workspace, home)

    allow = '(allow file-read* (regex #"/\\.[eE][nN][vV]\\.[eE][xX][aA][mM][pP][lL][eE]$"'
    assert index(text, '#"/\\.[eE][nN][vV]\\.[^/]*$"') < index(text, allow)


def test_settings_name_patterns_are_denied_after_the_template_exceptions(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=("*.token",))

    assert index(text, "[eE][xX][aA][mM][pP][lL][eE]$") < index(text, '#"/[^/]*\\.[tT][oO][kK][eE][nN]$"')


def test_system_folders_and_the_certifi_bundle_stay_readable_before_the_path_denials(workspace: Path, home: Path) -> None:
    text = profile(workspace, home)

    allow = next(line for line in text.splitlines() if line.startswith("(allow file-read* (subpath"))
    for folder in ("/etc", "/private/etc", "/usr", "/opt/homebrew", "/System", "/Library"):
        assert f'(subpath "{folder}")' in allow
    assert '(regex #"/certifi/cacert\\.pem$")' in allow
    assert index(text, '#"/[^/]*\\.[pP][eE][mM]$"') < index(text, allow) < index(text, f'(deny file-read* (subpath "{home / ".ssh"}"))')


def test_a_relative_settings_pattern_becomes_a_regex_under_any_folder(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=("secrets/**",))

    assert '#"^(.*/)?[sS][eE][cC][rR][eE][tT][sS]/.*$"' in text


def test_a_wildcard_path_pattern_becomes_an_anchored_regex(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=("~/logs/*.secret",))

    expected = "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else c for c in f"{home}/logs/")
    body = f'{expected.replace(".", chr(92) + ".")}[^/]*\\.[sS][eE][cC][rR][eE][tT]$"'
    assert f'#"^{body}' in text
    assert f'#"^/System/Volumes/Data{body}' in text


def test_a_quote_in_a_regex_is_written_as_a_dot_because_it_cannot_be_escaped(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=('we"ird/**',))

    assert '#"^(.*/)?[wW][eE].[iI][rR][dD]/.*$"' in text
    assert all(line.count('"') % 2 == 0 for line in text.splitlines() if "#" in line)


def test_a_backslash_in_a_regex_is_a_regex_escape(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, extra=("a\\b/**",))

    assert '[aA]\\\\[bB]/.*$"' in text


def test_a_glob_bracket_range_folds_each_endpoint_pair(workspace: Path, home: Path) -> None:
    from src.engine.guardrails.sandbox import glob_regex

    assert glob_regex("[a-c]") == "[a-cA-C]"
    assert glob_regex("[a-z0-9]") == "[a-zA-Z0-9]"
    assert glob_regex("[A-Z]") == "[A-Za-z]"
    assert glob_regex("[!a-c]") == "[^a-cA-C]"
    assert glob_regex("[abc]") == "[aAbBcC]"
    assert glob_regex("[_a-c-]") == "[_a-cA-C-]"


def test_a_glob_bracket_range_matches_the_same_letters_in_both_cases(workspace: Path, home: Path) -> None:
    import re

    from src.engine.guardrails.sandbox import glob_regex

    pattern = re.compile(glob_regex("f[a-c]x"))
    assert all(pattern.fullmatch(name) for name in ("fax", "FBX", "fCx", "Fcx"))
    assert not any(pattern.fullmatch(name) for name in ("fdx", "FDX"))


def test_network_on_adds_no_network_rule(workspace: Path, home: Path) -> None:
    assert "(deny network" not in profile(workspace, home, network=True)


def test_network_off_denies_outgoing_ip_except_loopback_and_allows_unix_sockets(workspace: Path, home: Path) -> None:
    text = profile(workspace, home, network=False)

    assert '(deny network*)(allow network* (remote ip "localhost:*") (remote unix-socket))' in text
    assert "(local ip)" not in text  # it matches every outgoing connection


# make_sandbox (AC13)


def make(cfg: GuardConfig, workspace: Path, home: Path, *, system: str = "Darwin", env: dict | None = None,
         probe=lambda argv: True, sessions: Path | None = None):
    return make_sandbox(cfg, workspace, home, env or {}, system, probe, sessions=sessions)


@pytest.fixture
def fake_exec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "sandbox-exec"
    path.write_text("")
    monkeypatch.setattr(sandbox_module, "SANDBOX_EXEC", path)
    return path


def test_make_sandbox_turns_it_on_and_lists_the_writable_folders(workspace: Path, home: Path, fake_exec: Path) -> None:
    sandbox, warning = make(GuardConfig(), workspace, home, env={"TMPDIR": str(workspace.parent / "tmpdir") + "/"})

    assert warning is None
    assert sandbox.profile is not None
    assert sandbox.reason == "on (macOS sandbox-exec)"
    assert sandbox.network is True
    assert sandbox.writable[0] == workspace
    for relative in (".cache", "Library/Caches", ".npm", ".bun", ".cargo/registry", ".local/share/uv", ".local/state"):
        assert home / relative in sandbox.writable
    for fixed in ("/tmp", "/private/tmp", "/private/var/folders"):
        assert Path(fixed).resolve() in sandbox.writable
    assert workspace.parent / "tmpdir" in sandbox.writable
    assert sandbox.wrap(["x"])[:3] == [str(fake_exec), "-p", sandbox.profile]


def test_make_sandbox_adds_the_global_write_list_expanding_the_home_folder(workspace: Path, home: Path, fake_exec: Path) -> None:
    cfg = GuardConfig(extra_write=("~/extra/cache", str(workspace.parent / "elsewhere")))

    sandbox, warning = make(cfg, workspace, home)

    assert warning is None
    assert home / "extra" / "cache" in sandbox.writable
    assert workspace.parent / "elsewhere" in sandbox.writable


def test_make_sandbox_ignores_a_relative_write_entry_with_a_warning(workspace: Path, home: Path, fake_exec: Path) -> None:
    sandbox, warning = make(GuardConfig(extra_write=("cache",)), workspace, home)

    assert sandbox.profile is not None
    assert warning is not None and "cache" in warning
    assert Path("cache") not in sandbox.writable


def test_make_sandbox_never_lets_the_agent_folders_be_written(workspace: Path, home: Path, fake_exec: Path) -> None:
    sessions = home / "elsewhere" / "sessions"

    sandbox, _ = make(GuardConfig(), workspace, home, sessions=sessions)

    assert sandbox.profile is not None
    for denied in (home / ".config" / "agent", home / ".local" / "share" / "agent", sessions,
                   workspace / ".agent" / "settings.json", workspace / ".agent" / "mcp.json"):
        assert f'(subpath "{denied}")' in sandbox.profile


def test_make_sandbox_follows_xdg_folders_for_the_agent_files(workspace: Path, home: Path, fake_exec: Path) -> None:
    env = {"XDG_CONFIG_HOME": str(home / "cfg"), "XDG_DATA_HOME": str(home / "data")}

    sandbox, _ = make(GuardConfig(), workspace, home, env=env)

    assert sandbox.profile is not None
    assert f'(subpath "{home / "cfg" / "agent"}")' in sandbox.profile
    assert f'(subpath "{home / "data" / "agent"}")' in sandbox.profile


def test_make_sandbox_passes_the_network_setting_to_the_profile(workspace: Path, home: Path, fake_exec: Path) -> None:
    sandbox, _ = make(GuardConfig(network=False), workspace, home)

    assert sandbox.network is False
    assert sandbox.profile is not None and "(deny network*)" in sandbox.profile


def test_make_sandbox_probes_with_the_real_profile_and_a_trivial_command(workspace: Path, home: Path, fake_exec: Path) -> None:
    seen: list[list[str]] = []

    def probe(argv: list[str]) -> bool:
        seen.append(argv)
        return True

    sandbox, _ = make(GuardConfig(), workspace, home, probe=probe)

    assert seen == [[str(fake_exec), "-p", sandbox.profile, "/usr/bin/true"]]


def test_disabled_in_settings_runs_unsandboxed_without_a_warning(workspace: Path, home: Path, fake_exec: Path) -> None:
    sandbox, warning = make(GuardConfig(sandbox_enabled=False), workspace, home)

    assert sandbox.profile is None
    assert sandbox.reason == "off: disabled in settings"
    assert warning is None


def test_not_macos_runs_unsandboxed_with_one_warning(workspace: Path, home: Path, fake_exec: Path) -> None:
    def probe(argv: list[str]) -> bool:
        raise AssertionError("must not probe")

    sandbox, warning = make(GuardConfig(), workspace, home, system="Linux", probe=probe)

    assert sandbox.profile is None
    assert sandbox.reason == "unavailable: not macOS"
    assert warning is not None and "not macOS" in warning
    assert sandbox.wrap(["ls"]) == ["ls"]


def test_a_missing_sandbox_exec_runs_unsandboxed_with_one_warning(workspace: Path, home: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_module, "SANDBOX_EXEC", tmp_path / "missing")

    sandbox, warning = make(GuardConfig(), workspace, home)

    assert sandbox.profile is None
    assert sandbox.reason == "unavailable: sandbox-exec not found"
    assert warning is not None and "sandbox-exec" in warning


def test_a_failing_probe_runs_unsandboxed_with_one_warning(workspace: Path, home: Path, fake_exec: Path) -> None:
    sandbox, warning = make(GuardConfig(), workspace, home, probe=lambda argv: False)

    assert sandbox.profile is None
    assert sandbox.reason == "unavailable: sandbox-exec could not start a test command"
    assert warning is not None and "test command" in warning


def test_a_probe_that_raises_is_a_failed_probe(workspace: Path, home: Path, fake_exec: Path) -> None:
    def probe(argv: list[str]) -> bool:
        raise OSError("boom")

    sandbox, warning = make(GuardConfig(), workspace, home, probe=probe)

    assert sandbox.profile is None
    assert warning is not None


def test_a_profile_that_cannot_be_built_falls_back_with_a_warning(workspace: Path, home: Path, fake_exec: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> str:
        raise ValueError("bad pattern")

    monkeypatch.setattr(sandbox_module, "build_profile", broken)

    sandbox, warning = make(GuardConfig(), workspace, home)

    assert sandbox.profile is None
    assert warning is not None and "sandbox" in warning


def test_the_warnings_escape_control_characters_from_settings(workspace: Path, home: Path, fake_exec: Path) -> None:
    _, warning = make(GuardConfig(extra_write=("bad\x1b[31m",)), workspace, home)

    assert warning is not None and "\x1b" not in warning


def test_a_tmpdir_that_is_the_root_the_home_folder_or_above_it_is_not_made_writable(
    workspace: Path, home: Path, fake_exec: Path
) -> None:
    for temp in ("/", str(home), str(home.parent), str(home.parent.parent), str(home) + "/"):
        sandbox, warning = make(GuardConfig(), workspace, home, env={"TMPDIR": temp})

        assert sandbox.profile is not None
        assert Path(temp).resolve() not in sandbox.writable, temp
        assert warning is not None and "TMPDIR" in warning, temp


def test_a_tmpdir_below_the_home_folder_is_still_made_writable(workspace: Path, home: Path, fake_exec: Path) -> None:
    sandbox, warning = make(GuardConfig(), workspace, home, env={"TMPDIR": str(home / "tmp")})

    assert warning is None
    assert home / "tmp" in sandbox.writable


def test_a_workspace_that_is_the_home_folder_or_holds_a_protected_folder_is_a_warning(
    home: Path, fake_exec: Path
) -> None:
    (home / ".ssh").mkdir()
    for workspace in (home, home.parent):
        _, warning = make(GuardConfig(), workspace, home)

        assert warning is not None and "workspace" in warning and ".ssh" in warning, workspace


def test_an_ordinary_workspace_gets_no_protected_folder_warning(workspace: Path, home: Path, fake_exec: Path) -> None:
    (home / ".ssh").mkdir()

    _, warning = make(GuardConfig(), workspace, home)

    assert warning is None


def test_the_agent_commands_and_skills_folders_are_never_writable(workspace: Path, home: Path, fake_exec: Path) -> None:
    sandbox, _ = make(GuardConfig(), workspace, home)

    assert sandbox.profile is not None
    for name in ("commands", "skills"):
        assert f'(subpath "{workspace / ".agent" / name}")' in sandbox.profile
