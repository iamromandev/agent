"""Settings files: where they are, how they are read (a bad file is an error value, never an exception)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from src.engine.permissions.config import (
    RuleFile,
    add_allow,
    global_settings_path,
    project_settings_path,
    read_rules,
)


def _write(path: Path, data: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return raw


def _texts(rules: tuple) -> list[str]:
    return [r.text for r in rules]


def test_the_global_file_is_under_xdg_config_home() -> None:
    assert global_settings_path({"XDG_CONFIG_HOME": "/xdg"}) == Path("/xdg/agent/settings.json")


def test_the_global_file_falls_back_to_the_home_config_folder() -> None:
    assert global_settings_path({}) == Path.home() / ".config" / "agent" / "settings.json"


def test_a_relative_xdg_config_home_is_ignored() -> None:
    assert global_settings_path({"XDG_CONFIG_HOME": "rel"}) == Path.home() / ".config" / "agent" / "settings.json"


def test_the_project_file_is_in_the_agent_folder_of_the_workspace(tmp_path: Path) -> None:
    assert project_settings_path(tmp_path) == tmp_path / ".agent" / "settings.json"


def test_a_missing_file_is_no_rules_and_no_error(tmp_path: Path) -> None:
    assert read_rules(tmp_path / "settings.json") == RuleFile((), (), (), None, None)


def test_allow_and_deny_lists_are_read_in_order_with_the_exact_bytes(tmp_path: Path) -> None:
    raw = _write(
        tmp_path / "s.json", {"permissions": {"allow": ["bash(make test)", "write"], "deny": ["bash(rm*)"]}}
    )

    loaded = read_rules(tmp_path / "s.json")

    assert _texts(loaded.allow) == ["bash(make test)", "write"]
    assert _texts(loaded.deny) == ["bash(rm*)"]
    assert loaded.invalid == ()
    assert loaded.error is None
    assert loaded.raw == raw


def test_other_top_level_keys_and_other_permission_keys_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "s.json", {"theme": "x", "permissions": {"allow": ["bash"], "ask": ["write"]}})

    loaded = read_rules(tmp_path / "s.json")

    assert _texts(loaded.allow) == ["bash"]
    assert loaded.error is None


def test_a_file_without_permissions_is_valid_and_has_no_rules(tmp_path: Path) -> None:
    raw = _write(tmp_path / "s.json", {"theme": "x"})

    loaded = read_rules(tmp_path / "s.json")

    assert (loaded.allow, loaded.deny, loaded.error) == ((), (), None)
    assert loaded.raw == raw


def test_an_invalid_rule_is_kept_with_its_reason_and_the_valid_ones_still_apply(tmp_path: Path) -> None:
    _write(tmp_path / "s.json", {"permissions": {"allow": ["bash(a*b)", "bash()", "bash(ok)"], "deny": ["x y"]}})

    loaded = read_rules(tmp_path / "s.json")

    assert _texts(loaded.allow) == ["bash(ok)"]
    assert [text for text, _ in loaded.invalid] == ["bash(a*b)", "bash()", "x y"]
    assert all(reason for _, reason in loaded.invalid)
    assert loaded.error is None


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"[1, 2]",
        b'"text"',
        b'{"permissions": []}',
        b"\xff\xfe",
        b"",
    ],
)
def test_a_file_with_bad_json_or_shape_is_an_error_and_contributes_no_rules(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "s.json"
    path.write_bytes(content)

    loaded = read_rules(path)

    assert loaded.error
    assert (loaded.allow, loaded.deny) == ((), ())


@pytest.mark.parametrize("bad", ["x", ["bash", 5], {"a": 1}, 3])
def test_a_bad_deny_list_does_not_drop_the_allow_rules_and_is_reported(tmp_path: Path, bad: object) -> None:
    _write(tmp_path / "s.json", {"permissions": {"allow": ["bash"], "deny": bad}})

    loaded = read_rules(tmp_path / "s.json")

    assert loaded.error is None
    assert _texts(loaded.allow) == ["bash"]
    assert loaded.deny == ()
    assert [e.split(" ")[0] for e in loaded.list_errors] == ['"permissions.deny"']


@pytest.mark.parametrize("bad", ["x", ["bash", 5], {"a": 1}, 3])
def test_a_bad_allow_list_does_not_drop_the_deny_rules_and_is_reported(tmp_path: Path, bad: object) -> None:
    _write(tmp_path / "s.json", {"permissions": {"allow": bad, "deny": ["bash(rm*)"]}})

    loaded = read_rules(tmp_path / "s.json")

    assert loaded.error is None
    assert loaded.allow == ()
    assert _texts(loaded.deny) == ["bash(rm*)"]
    assert [e.split(" ")[0] for e in loaded.list_errors] == ['"permissions.allow"']


def test_two_bad_lists_are_reported_once_each(tmp_path: Path) -> None:
    _write(tmp_path / "s.json", {"permissions": {"allow": "x", "deny": "y"}})

    assert len(read_rules(tmp_path / "s.json").list_errors) == 2


def test_a_symlinked_file_is_an_error_and_contributes_no_rules(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    _write(real, {"permissions": {"allow": ["bash"]}})
    link = tmp_path / "link.json"
    link.symlink_to(real)

    loaded = read_rules(link)

    assert loaded.error is not None
    assert "symlink" in loaded.error
    assert loaded.allow == ()


def test_a_symlink_can_be_followed_when_the_caller_says_so(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    _write(real, {"permissions": {"allow": ["bash"]}})
    link = tmp_path / "link.json"
    link.symlink_to(real)

    loaded = read_rules(link, follow_symlink=True)

    assert loaded.error is None
    assert _texts(loaded.allow) == ["bash"]


def test_an_unreadable_path_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "s.json").mkdir()  # a directory where the file should be

    loaded = read_rules(tmp_path / "s.json")

    assert loaded.error is not None
    assert loaded.allow == ()


DEEP = b"[" * 200_000 + b"]" * 200_000


def test_deeply_nested_json_is_a_parse_failure_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_bytes(DEEP)

    loaded = read_rules(path)

    assert loaded.error is not None
    assert loaded.raw == DEEP
    assert (loaded.allow, loaded.deny) == ((), ())


def test_a_file_over_the_size_cap_is_an_error_and_is_not_parsed(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_bytes(b'{"permissions": {"allow": []}, "pad": "' + b"x" * (1 << 20) + b'"}')

    loaded = read_rules(path)

    assert loaded.error is not None
    assert "larger than" in loaded.error


def test_a_fifo_is_refused_without_reading_it(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    os.mkfifo(path)  # reading it would block until someone writes

    loaded = read_rules(path)

    assert loaded.error is not None
    assert "regular file" in loaded.error


def test_a_symlinked_agent_folder_is_refused_for_the_project_file(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    _write(elsewhere / "settings.json", {"permissions": {"allow": ["bash"], "deny": ["bash(rm*)"]}})
    (tmp_path / ".agent").symlink_to(elsewhere)

    loaded = read_rules(tmp_path / ".agent" / "settings.json")

    assert loaded.error is not None
    assert "symlink" in loaded.error
    assert (loaded.allow, loaded.deny) == ((), ())


# add_allow


def _allow_list(path: Path) -> list[str]:
    return json.loads(path.read_text())["permissions"]["allow"]


def test_add_allow_creates_the_file_and_the_agent_folder(tmp_path: Path) -> None:
    path = tmp_path / ".agent" / "settings.json"

    read, new = add_allow(path, "bash(make test)")

    assert read is None  # there was no file to read
    assert json.loads(path.read_text()) == {"permissions": {"allow": ["bash(make test)"]}}
    assert path.read_bytes() == new
    assert path.read_text().endswith("\n")


def test_add_allow_keeps_every_other_key_and_rule(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    _write(path, {"theme": {"a": [1, 2]}, "permissions": {"allow": ["bash(ls)"], "deny": ["bash(rm*)"], "ask": ["x"]}})

    add_allow(path, "write")

    assert json.loads(path.read_text()) == {
        "theme": {"a": [1, 2]},
        "permissions": {"allow": ["bash(ls)", "write"], "deny": ["bash(rm*)"], "ask": ["x"]},
    }


def test_add_allow_creates_the_missing_parts_of_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    _write(path, {"theme": "x"})

    add_allow(path, "write")

    assert json.loads(path.read_text()) == {"theme": "x", "permissions": {"allow": ["write"]}}


def test_add_allow_adds_no_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    add_allow(path, "write")
    first = path.read_bytes()

    assert add_allow(path, "write") == (first, first)

    assert _allow_list(path) == ["write"]


@pytest.mark.parametrize("content", [b"{oops", b"[1]", b'{"permissions": []}', b'{"permissions": {"allow": "x"}}', b"\xff"])
def test_add_allow_never_overwrites_a_file_it_cannot_understand(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "s.json"
    path.write_bytes(content)

    with pytest.raises(ValueError):
        add_allow(path, "write")

    assert path.read_bytes() == content


def test_add_allow_returns_the_bytes_it_read_next_to_the_bytes_it_wrote(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    before = _write(path, {"permissions": {"allow": ["bash(ls)"]}})

    read, new = add_allow(path, "write")

    assert read == before
    assert new == path.read_bytes()


def test_add_allow_writes_non_ascii_as_is(tmp_path: Path) -> None:
    path = tmp_path / "s.json"

    add_allow(path, "bash(echo é)")

    assert "bash(echo é)" in path.read_text(encoding="utf-8")


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a read-only file")
def test_add_allow_refuses_a_file_that_is_not_writable(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    before = _write(path, {"permissions": {"allow": []}})
    path.chmod(0o400)

    with pytest.raises(OSError, match="not writable"):
        add_allow(path, "write")

    assert path.read_bytes() == before


def test_add_allow_refuses_a_symlinked_file_and_leaves_its_target_alone(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    before = _write(real, {"permissions": {"allow": ["bash(x)"]}})
    link = tmp_path / "s.json"
    link.symlink_to(real)

    with pytest.raises(OSError, match="symlink"):
        add_allow(link, "write")

    assert real.read_bytes() == before
    assert link.is_symlink()


def test_add_allow_refuses_a_symlinked_agent_folder(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / ".agent").symlink_to(elsewhere)

    with pytest.raises(OSError, match="symlink"):
        add_allow(tmp_path / ".agent" / "settings.json", "write")

    assert list(elsewhere.iterdir()) == []


def test_a_crash_while_writing_leaves_the_old_file_whole_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s.json"
    before = _write(path, {"theme": "x", "permissions": {"allow": ["bash(ls)"]}})

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        add_allow(path, "write")

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]


def test_a_crash_while_creating_the_file_leaves_nothing_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".agent" / "settings.json"

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        add_allow(path, "write")

    assert not path.exists()
    assert list(path.parent.iterdir()) == []


def test_add_allow_refuses_deeply_nested_json_an_oversized_file_and_a_fifo(tmp_path: Path) -> None:
    deep, big, fifo = tmp_path / "deep.json", tmp_path / "big.json", tmp_path / "fifo.json"
    deep.write_bytes(DEEP)
    big.write_bytes(b'{"pad": "' + b"x" * (1 << 20) + b'"}')
    os.mkfifo(fifo)

    with pytest.raises(ValueError):
        add_allow(deep, "write")
    with pytest.raises(OSError, match="larger than"):
        add_allow(big, "write")
    with pytest.raises(OSError, match="regular file"):
        add_allow(fifo, "write")

    assert deep.read_bytes() == DEEP


def test_the_file_is_private_to_the_user_when_created(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    add_allow(path, "write")

    assert path.stat().st_mode & 0o077 == 0


def test_the_parsed_top_level_object_is_kept_with_its_other_keys(tmp_path: Path) -> None:
    _write(tmp_path / "s.json", {"permissions": {"allow": ["write"]}, "guardrails": {"protected": ["*.key"]}})

    loaded = read_rules(tmp_path / "s.json")

    assert loaded.data == {"permissions": {"allow": ["write"]}, "guardrails": {"protected": ["*.key"]}}


def test_a_missing_or_unusable_file_has_no_data(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{nope")
    (tmp_path / "list.json").write_text("[]")

    assert read_rules(tmp_path / "missing.json").data is None
    assert read_rules(tmp_path / "bad.json").data is None
    assert read_rules(tmp_path / "list.json").data is None
