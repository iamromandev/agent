"""Protected paths (R3): built-in patterns, names anywhere, settings patterns, matching after resolution."""

from __future__ import annotations

import os
import sys
import time
import unicodedata
from pathlib import Path

import pytest
from src.engine.guardrails import paths as paths_module
from src.engine.guardrails.paths import BUILTIN_HOME, BUILTIN_NAMES, MAX_TRACKED_FILES, ProtectedPaths


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    return folder.resolve()


def protected(home: Path, *extra: str) -> ProtectedPaths:
    return ProtectedPaths(home, extra)


@pytest.mark.parametrize(
    "relative",
    [
        ".ssh/config",
        ".ssh/keys/deploy",
        ".aws/credentials",
        ".gnupg/pubring.kbx",
        ".config/gh/hosts.yml",
        ".docker/config.json",
        ".netrc",
        ".kube/config",
        "Library/Keychains/login.keychain-db",
    ],
)
def test_the_builtin_home_paths_are_protected(home: Path, relative: str) -> None:
    assert protected(home).is_protected(home / relative)


@pytest.mark.parametrize(
    "relative",
    [".docker/other.json", ".kube/cache/x", ".config/other/x", ".sshx/config", "notes/ssh", ".aws"],
)
def test_other_paths_in_the_home_folder_are_not(home: Path, relative: str) -> None:
    assert not protected(home).is_protected(home / relative)


@pytest.mark.parametrize(
    "path",
    ["/ws/.env", "/ws/config/.env.local", "/ws/a/b/server.pem", "/ws/x.key", "/ws/id_rsa", "/ws/id_rsa.pub", "/ws/id_ed25519", "/ws/id_ecdsa.bak"],
)
def test_file_names_are_protected_anywhere(home: Path, path: str) -> None:
    assert protected(home).is_protected(Path(path))


@pytest.mark.parametrize("name", [".env.example", ".env.sample", ".env.template"])
def test_ac4_template_env_files_are_not_protected(home: Path, name: str) -> None:
    assert not protected(home).is_protected(Path("/ws") / name)


@pytest.mark.parametrize("path", ["/ws/app.py", "/ws/environment", "/ws/keys.txt", "/ws/.envrc", "/ws/id_dsa", "/ws/monkey"])
def test_ordinary_files_are_not_protected(home: Path, path: str) -> None:
    assert not protected(home).is_protected(Path(path))


def test_a_settings_pattern_with_a_home_prefix_is_protected(home: Path) -> None:
    paths = protected(home, "~/secrets/**")

    assert paths.is_protected(home / "secrets" / "a" / "b.txt")
    assert not paths.is_protected(home / "other" / "b.txt")


def test_a_settings_pattern_without_a_slash_matches_the_name_anywhere(home: Path) -> None:
    paths = protected(home, "*.token", "vault.json")

    assert paths.is_protected(Path("/ws/deep/a.token"))
    assert paths.is_protected(Path("/ws/vault.json"))
    assert not paths.is_protected(Path("/ws/vault.json.bak"))


def test_a_settings_name_pattern_has_no_template_exception(home: Path) -> None:
    assert protected(home, ".env.example").is_protected(Path("/ws/.env.example"))


def test_a_relative_settings_pattern_with_a_slash_matches_under_any_folder(home: Path) -> None:
    paths = protected(home, "secrets/**")

    assert paths.is_protected(Path("/ws/secrets/key.txt"))
    assert paths.is_protected(home / "x" / "secrets" / "deep" / "key.txt")
    assert not paths.is_protected(Path("/ws/public/key.txt"))


def test_matching_ignores_case_because_the_file_system_may(home: Path) -> None:
    paths = protected(home)

    assert paths.is_protected(home / ".SSH" / "id")
    assert paths.is_protected(Path("/ws/.ENV"))


def test_ac4_a_symlink_to_a_protected_file_is_protected_once_resolved(home: Path, tmp_path: Path) -> None:
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text("x")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link.txt").symlink_to(home / ".aws" / "credentials")
    paths = protected(home)

    assert paths.is_protected(workspace / "link.txt")  # the file system says it is the same file, resolved or not
    assert paths.is_protected((workspace / "link.txt").resolve())


def test_a_home_that_is_a_symlink_is_matched_in_both_forms(tmp_path: Path) -> None:
    real = tmp_path / "real"
    (real / ".ssh").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    paths = ProtectedPaths(link, ())

    assert paths.is_protected(real / ".ssh" / "id")
    assert paths.is_protected(link / ".ssh" / "id")


def test_patterns_list_the_builtin_ones_then_the_settings_ones(home: Path) -> None:
    listed = protected(home, "~/secrets/**", "*.token").patterns()

    assert listed[: len(BUILTIN_HOME) + len(BUILTIN_NAMES)] == [*BUILTIN_HOME, *BUILTIN_NAMES]
    assert listed[-2:] == ["~/secrets/**", "*.token"]


def test_the_counts_split_builtin_from_settings_patterns(home: Path) -> None:
    paths = protected(home, "a", "b/**")

    assert paths.builtin_count == len(BUILTIN_HOME) + len(BUILTIN_NAMES)
    assert paths.extra_count == 2


def test_home_roots_are_the_absolute_folders_and_files_for_the_sandbox(home: Path) -> None:
    roots = protected(home, "~/secrets/**", "*.token").home_roots()

    assert home / ".ssh" in roots
    assert home / ".docker" / "config.json" in roots
    assert home / ".netrc" in roots
    assert home / "secrets" in roots
    assert all(root.is_absolute() for root in roots)


def test_matching_one_path_is_fast(home: Path) -> None:
    paths = protected(home, "~/a/**", "*.token")
    target = Path("/ws/some/deep/folder/file.txt")
    paths.is_protected(target)

    start = time.perf_counter()
    for _ in range(200):
        paths.is_protected(target)
    per_call = (time.perf_counter() - start) / 200

    assert per_call < 0.001, f"{per_call * 1000:.3f} ms per path"


# --- the same file under another spelling (firmlinks, hard links, Unicode forms) -------


def test_a_second_path_to_a_protected_folder_is_protected_by_identity(home: Path, tmp_path: Path) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "config").write_text("Host x\n")
    alias = tmp_path / "alias"
    alias.symlink_to(home / ".ssh")  # stands in for a firmlink: a spelling the patterns do not know

    paths = protected(home)

    assert paths.is_protected(alias / "config")
    assert paths.is_protected(alias / "not-yet" / "deep")  # anything under the folder, existing or not
    assert not paths.is_protected(tmp_path / "other" / "config")


def test_a_second_path_to_a_protected_file_is_protected_by_identity(home: Path, tmp_path: Path) -> None:
    (home / ".netrc").write_text("machine x\n")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(home / ".netrc")

    assert protected(home).is_protected(alias)


def test_a_hard_link_to_a_protected_exact_file_is_protected(home: Path, tmp_path: Path) -> None:
    (home / ".kube").mkdir()
    (home / ".kube" / "config").write_text("token: x\n")
    link = tmp_path / "notes.txt"
    os.link(home / ".kube" / "config", link)

    assert protected(home).is_protected(link)


def test_a_hard_link_to_a_file_under_a_protected_folder_is_protected(home: Path, tmp_path: Path) -> None:
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text("x\n")
    link = tmp_path / "innocent.txt"
    os.link(home / ".aws" / "credentials", link)

    assert protected(home).is_protected(link)


def test_a_hard_link_under_a_settings_folder_is_protected(home: Path, tmp_path: Path) -> None:
    (home / "secrets" / "deep").mkdir(parents=True)
    (home / "secrets" / "deep" / "a.txt").write_text("x\n")
    link = tmp_path / "copy.txt"
    os.link(home / "secrets" / "deep" / "a.txt", link)

    assert protected(home, "~/secrets/**").is_protected(link)


def test_a_file_next_to_the_hard_link_is_not_protected(home: Path, tmp_path: Path) -> None:
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text("x\n")
    other = tmp_path / "other.txt"
    other.write_text("x\n")

    assert not protected(home).is_protected(other)


def test_hard_links_are_tracked_for_a_bounded_number_of_files(home: Path, tmp_path: Path) -> None:
    folder = home / ".aws"
    folder.mkdir()
    for index in range(MAX_TRACKED_FILES + 5):
        (folder / f"f{index}").write_text("x")

    paths = protected(home)

    assert paths.tracking_truncated
    assert not protected(tmp_path / "empty-home").tracking_truncated


def test_files_deeper_than_the_depth_bound_are_not_tracked_as_hard_links(home: Path, tmp_path: Path) -> None:
    deep = home / ".aws" / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / "x.txt").write_text("x")
    link = tmp_path / "deep.txt"
    os.link(deep / "x.txt", link)

    paths = protected(home)

    assert paths.tracking_truncated
    assert not paths.is_protected(link)


def test_an_nfd_home_spelling_matches_the_nfc_path(tmp_path: Path) -> None:
    nfd = unicodedata.normalize("NFD", "h\u00f6me")
    nfc = unicodedata.normalize("NFC", "h\u00f6me")
    assert nfd != nfc
    paths = ProtectedPaths(Path("/Users") / nfd, ())

    assert paths.is_protected(Path("/Users") / nfc / ".ssh" / "id")
    assert ProtectedPaths(Path("/Users") / nfc, ()).is_protected(Path("/Users") / nfd / ".ssh" / "id")


def test_an_nfd_settings_pattern_and_name_match_the_nfc_path(home: Path) -> None:
    nfd = unicodedata.normalize("NFD", "caf\u00e9")
    nfc = unicodedata.normalize("NFC", "caf\u00e9")
    paths = protected(home, f"~/{nfd}/**", f"{nfd}.token")

    assert paths.is_protected(home / nfc / "x")
    assert paths.is_protected(Path("/ws") / f"{nfc}.token")


def test_the_firmlink_prefix_is_removed_before_matching_on_macos(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths_module, "STRIP_FIRMLINK", True)
    spelled = Path("/System/Volumes/Data" + str(home / ".ssh" / "config"))

    assert protected(home).is_protected(spelled)
    assert protected(home, "~/secrets/**").is_protected(Path("/System/Volumes/Data" + str(home / "secrets" / "a")))
    assert protected(home).is_protected(Path("/System/Volumes/Data/ws/.env"))
    assert not protected(home).is_protected(Path("/System/Volumes/Data/ws/app.py"))


@pytest.mark.parametrize("prefix", ["/system/volumes/data", "/System/volumes/DATA", "/SYSTEM/VOLUMES/DATA"])
def test_the_firmlink_prefix_is_removed_whatever_its_case(home: Path, monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    monkeypatch.setattr(paths_module, "STRIP_FIRMLINK", True)
    paths = protected(home)  # built before the files below exist

    (home / ".ssh").mkdir()
    (home / ".ssh" / "config").write_text("Host x\n")  # created after launch: only the pattern can catch it

    assert paths.is_protected(Path(prefix + str(home / ".ssh" / "config")))
    assert paths.is_protected(Path(prefix + "/ws/.env"))
    assert not paths.is_protected(Path(prefix + "/ws/app.py"))
    assert not paths.is_protected(Path(prefix))


@pytest.mark.parametrize("prefix", ["/system/volumes/data", "/System/Volumes/DATA"])
def test_a_folder_pattern_whose_folder_appears_later_matches_under_a_cased_firmlink(
    home: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    monkeypatch.setattr(paths_module, "STRIP_FIRMLINK", True)
    paths = protected(home, "~/secrets")  # the folder does not exist yet
    (home / "secrets").mkdir()
    (home / "secrets" / "a.txt").write_text("x")

    assert paths.is_protected(Path(prefix + str(home / "secrets" / "a.txt")))
    assert paths.is_protected(Path(prefix + str(home / "secrets")))


def test_the_firmlink_prefix_is_kept_elsewhere(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths_module, "STRIP_FIRMLINK", False)

    assert not protected(home).is_protected(Path("/System/Volumes/Data" + str(home / ".ssh" / "config")))


@pytest.mark.skipif(sys.platform != "darwin", reason="firmlinks are a macOS file system feature")
def test_the_real_firmlink_form_of_a_protected_file_is_protected(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (home / ".ssh").mkdir()
    (home / ".ssh" / "config").write_text("Host x\n")
    spelled = Path("/System/Volumes/Data" + str(home / ".ssh" / "config"))
    assert spelled.exists()  # the firmlink really is another way to the same file
    monkeypatch.setattr(paths_module, "STRIP_FIRMLINK", False)  # identity alone must catch it

    assert protected(home).is_protected(spelled)
    assert protected(home).is_protected(Path("/System/Volumes/Data" + str(home / ".ssh" / "new-file")))


# --- settings patterns that name a folder ----------------------------------------------


@pytest.mark.parametrize("pattern", ["~/secrets", "~/secrets/"])
def test_a_settings_pattern_naming_a_folder_covers_what_is_inside(home: Path, pattern: str) -> None:
    paths = protected(home, pattern)

    assert paths.is_protected(home / "secrets" / "a" / "b.txt")
    assert paths.is_protected(home / "secrets")
    assert not paths.is_protected(home / "secrets2" / "a.txt")
    assert home / "secrets" in paths.home_roots()


def test_a_relative_settings_pattern_naming_a_folder_covers_what_is_inside(home: Path) -> None:
    paths = protected(home, "priv/")

    assert paths.is_protected(Path("/ws/priv/key.txt"))
    assert not paths.is_protected(Path("/ws/public/key.txt"))


def test_a_settings_pattern_with_a_wildcard_in_the_last_segment_is_not_widened(home: Path) -> None:
    paths = protected(home, "~/logs/*.secret")

    assert paths.is_protected(home / "logs" / "a.secret")
    assert not paths.is_protected(home / "logs" / "a.txt")


def test_a_settings_pattern_with_a_dotdot_segment_is_ignored(home: Path) -> None:
    paths = protected(home, "~/a/../b/**", "../x")

    assert not paths.is_protected(home / "b" / "c")
    assert not paths.is_protected(home / "a" / ".." / "b" / "c")
    assert not paths.is_protected(Path("/ws/../x/y"))


# --- the open file itself (a path can change between a check and the open) --------------


def test_an_open_protected_file_is_protected_by_its_descriptor(home: Path) -> None:
    (home / ".netrc").write_text("machine x\n")
    paths = protected(home)

    with open(home / ".netrc", "rb") as handle:
        assert paths.is_protected_fd(handle.fileno())


def test_an_open_file_with_a_protected_name_is_protected_by_its_descriptor(home: Path, tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / ".env").write_text("A=1\n")
    link = tmp_path / "ws" / "link.txt"
    link.symlink_to(tmp_path / "ws" / ".env")

    with open(link, "rb") as handle:  # opened through the link: the descriptor knows where it really is
        assert protected(home).is_protected_fd(handle.fileno())


def test_a_file_made_after_launch_inside_a_protected_folder_is_protected_by_its_descriptor(home: Path) -> None:
    paths = protected(home)
    (home / ".ssh").mkdir()
    (home / ".ssh" / "late").write_text("x")

    with open(home / ".ssh" / "late", "rb") as handle:
        assert paths.is_protected_fd(handle.fileno())


def test_an_ordinary_open_file_is_not_protected_by_its_descriptor(home: Path, tmp_path: Path) -> None:
    (tmp_path / "plain.txt").write_text("x")

    with open(tmp_path / "plain.txt", "rb") as handle:
        assert not protected(home).is_protected_fd(handle.fileno())


def test_a_descriptor_that_cannot_be_examined_counts_as_protected(home: Path) -> None:
    assert protected(home).is_protected_fd(10_000)


# --- scan bounds, workspace names, trimmed patterns, second home -----------------------


def test_the_scan_of_a_protected_folder_stops_at_a_bound_on_folders_visited(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths_module, "MAX_TRACKED_DIRS", 5)
    for index in range(20):
        (home / ".aws" / f"d{index}").mkdir(parents=True)

    paths = protected(home)

    assert paths.tracking_truncated


def test_a_few_folders_stay_under_the_bound(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths_module, "MAX_TRACKED_DIRS", 5)
    for index in range(3):
        (home / ".aws" / f"d{index}").mkdir(parents=True)

    assert not protected(home).tracking_truncated


def test_the_folder_bound_covers_the_scan_of_the_workspace_too(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths_module, "MAX_TRACKED_DIRS", 5)
    workspace = tmp_path / "ws"
    for index in range(20):
        (workspace / f"d{index}").mkdir(parents=True)

    assert ProtectedPaths(home, (), workspace).tracking_truncated


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    folder = tmp_path / "ws"
    (folder / "config" / "deep").mkdir(parents=True)
    return folder


@pytest.mark.parametrize("name", [".env", ".env.local", "server.pem", "x.key", "id_rsa", "id_ed25519.pub"])
def test_a_hard_link_to_a_workspace_file_with_a_protected_name_is_protected(
    home: Path, workspace: Path, tmp_path: Path, name: str
) -> None:
    secret = workspace / "config" / "deep" / name
    secret.write_text("A=1\n")
    paths = ProtectedPaths(home, (), workspace)
    link = tmp_path / "innocent.txt"
    os.link(secret, link)  # made after launch, to a file that existed at launch

    assert paths.is_protected(link)


def test_a_hard_link_to_a_workspace_file_matching_a_settings_name_pattern_is_protected(
    home: Path, workspace: Path, tmp_path: Path
) -> None:
    (workspace / "a.token").write_text("t\n")
    paths = ProtectedPaths(home, ("*.token",), workspace)
    link = tmp_path / "copy.txt"
    os.link(workspace / "a.token", link)

    assert paths.is_protected(link)


def test_a_hard_link_to_a_workspace_template_or_ordinary_file_is_not_protected(
    home: Path, workspace: Path, tmp_path: Path
) -> None:
    (workspace / ".env.example").write_text("A=\n")
    (workspace / "app.py").write_text("x\n")
    paths = ProtectedPaths(home, (), workspace)
    os.link(workspace / ".env.example", tmp_path / "t.txt")
    os.link(workspace / "app.py", tmp_path / "a.txt")

    assert not paths.is_protected(tmp_path / "t.txt")
    assert not paths.is_protected(tmp_path / "a.txt")


def test_the_workspace_scan_does_not_enter_dependency_and_vcs_folders(home: Path, workspace: Path, tmp_path: Path) -> None:
    (workspace / "node_modules" / "pkg").mkdir(parents=True)
    (workspace / "node_modules" / "pkg" / ".env").write_text("A=1\n")
    paths = ProtectedPaths(home, (), workspace)
    link = tmp_path / "copy.txt"
    os.link(workspace / "node_modules" / "pkg" / ".env", link)

    assert not paths.is_protected(link)
    assert not paths.tracking_truncated


def test_without_a_workspace_nothing_in_it_is_tracked(home: Path, workspace: Path, tmp_path: Path) -> None:
    (workspace / ".env").write_text("A=1\n")
    link = tmp_path / "copy.txt"
    os.link(workspace / ".env", link)

    assert not ProtectedPaths(home, ()).is_protected(link)


def test_the_workspace_scan_records_a_bounded_number_of_files(
    home: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths_module, "MAX_TRACKED_FILES", 3)
    for index in range(6):
        (workspace / f"k{index}.pem").write_text("x")

    assert ProtectedPaths(home, (), workspace).tracking_truncated


def test_settings_patterns_are_trimmed(home: Path) -> None:
    paths = protected(home, "  ~/secrets/**  ", " *.token ", "   ")

    assert paths.is_protected(home / "secrets" / "a")
    assert paths.is_protected(Path("/ws/a.token"))
    assert paths.extra_count == 2
    assert not paths.is_protected(Path("/ws/anything"))


def test_a_second_home_is_protected_too(home: Path, tmp_path: Path) -> None:
    other = tmp_path / "other-home"
    other.mkdir()
    paths = ProtectedPaths(home, ("~/secrets/**",), also_home=other.resolve())

    assert paths.is_protected(home / ".ssh" / "id")
    assert paths.is_protected(other.resolve() / ".ssh" / "id")
    assert paths.is_protected(other.resolve() / "secrets" / "x")
    assert paths.home_folders == [home, other.resolve()]


def test_login_home_is_none_when_it_is_the_current_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(paths_module.pwd, "getpwuid", lambda uid: type("P", (), {"pw_dir": str(tmp_path)})())

    assert paths_module.login_home() is None


def test_login_home_is_the_account_home_when_home_points_elsewhere(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "fake-home"
    real = tmp_path / "real-home"
    fake.mkdir()
    real.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake)
    monkeypatch.setattr(paths_module.pwd, "getpwuid", lambda uid: type("P", (), {"pw_dir": str(real)})())

    assert paths_module.login_home() == real


def test_login_home_is_none_when_the_account_has_no_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(uid: int) -> object:
        raise KeyError(uid)

    monkeypatch.setattr(paths_module.pwd, "getpwuid", missing)

    assert paths_module.login_home() is None
