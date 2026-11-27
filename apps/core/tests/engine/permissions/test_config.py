"""Settings files: where they are, how they are read (a bad file is an error value, never an exception)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from core.engine.permissions.config import (
    add_allow,
    read_rules,
    set_top_key,
)


def _write(path: Path, data: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return raw


def _texts(rules: tuple) -> list[str]:
    return [r.text for r in rules]


@pytest.mark.parametrize(
    "content",
    [
        b'{"permissions": []}',
        b"\xff\xfe",
    ],
)
def test_a_file_with_bad_json_or_shape_is_an_error_and_contributes_no_rules(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "s.json"
    path.write_bytes(content)

    loaded = read_rules(path)

    assert loaded.error
    assert (loaded.allow, loaded.deny) == ((), ())


DEEP = b"[" * 200_000 + b"]" * 200_000


# add_allow


def _allow_list(path: Path) -> list[str]:
    return json.loads(path.read_text())["permissions"]["allow"]


@pytest.mark.parametrize("content", [b'{"permissions": []}', b'{"permissions": {"allow": "x"}}'])
def test_add_allow_never_overwrites_a_file_it_cannot_understand(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "s.json"
    path.write_bytes(content)

    with pytest.raises(ValueError):
        add_allow(path, "write")

    assert path.read_bytes() == content


def test_a_missing_or_unusable_file_has_no_data(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{nope")
    (tmp_path / "list.json").write_text("[]")

    assert read_rules(tmp_path / "missing.json").data is None
    assert read_rules(tmp_path / "bad.json").data is None
    assert read_rules(tmp_path / "list.json").data is None


# set_top_key


def test_set_top_key_with_the_same_value_leaves_the_file_byte_for_byte(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    raw = _write(path, {"model": "m"})

    assert set_top_key(path, "model", "m") == (raw, raw)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("content", [b"[1]"])
def test_set_top_key_never_overwrites_a_file_it_cannot_understand(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "s.json"
    path.write_bytes(content)

    with pytest.raises(ValueError):
        set_top_key(path, "model", "m")

    assert path.read_bytes() == content


def test_set_top_key_refuses_deeply_nested_json_an_oversized_file_and_a_fifo(tmp_path: Path) -> None:
    deep, big, fifo = tmp_path / "deep.json", tmp_path / "big.json", tmp_path / "fifo.json"
    deep.write_bytes(DEEP)
    big.write_bytes(b'{"pad": "' + b"x" * (1 << 20) + b'"}')
    os.mkfifo(fifo)

    with pytest.raises(ValueError):
        set_top_key(deep, "model", "m")
    with pytest.raises(OSError, match="larger than"):
        set_top_key(big, "model", "m")
    with pytest.raises(OSError, match="regular file"):
        set_top_key(fifo, "model", "m")

    assert deep.read_bytes() == DEEP


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write to a read-only file")
def test_set_top_key_refuses_a_file_that_is_not_writable(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    before = _write(path, {"a": 1})
    path.chmod(0o400)

    with pytest.raises(OSError, match="not writable"):
        set_top_key(path, "model", "m")

    assert path.read_bytes() == before


def test_a_crash_while_setting_a_key_leaves_the_old_file_whole_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "s.json"
    before = _write(path, {"a": 1})

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        set_top_key(path, "model", "m")

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]
