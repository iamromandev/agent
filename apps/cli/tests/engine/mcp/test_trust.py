from __future__ import annotations

import os
from pathlib import Path

import pytest
from src.engine.mcp.trust import TrustStore

CONTENT = b'{"servers": {"db": {"command": "x"}}}'


@pytest.mark.parametrize("junk", [b'{"/ws": 5}'])
def test_a_corrupt_file_means_untrusted_and_the_next_trust_overwrites_it(tmp_path: Path, junk: bytes) -> None:
    path = tmp_path / "trust.json"
    path.write_bytes(junk)
    store = TrustStore(path)

    assert store.is_trusted(tmp_path / "ws", CONTENT) is False
    store.trust(tmp_path / "ws", CONTENT)

    assert store.is_trusted(tmp_path / "ws", CONTENT) is True


def test_forget_removes_one_folder_and_keeps_the_others(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    store.trust(tmp_path / "a", CONTENT)
    store.trust(tmp_path / "b", CONTENT)

    store.forget(tmp_path / "a")

    assert store.is_trusted(tmp_path / "a", CONTENT) is False
    assert store.is_trusted(tmp_path / "b", CONTENT) is True


def test_forget_says_whether_it_wrote_and_holds_says_whether_a_folder_is_listed(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")
    store.trust(tmp_path / "a", CONTENT)
    assert store.holds(tmp_path / "a") and not store.holds(tmp_path / "b")

    assert store.forget(tmp_path / "a") is True
    assert store.forget(tmp_path / "a") is False
    assert not store.holds(tmp_path / "a")


def test_forget_with_nothing_to_forget_writes_nothing(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "trust.json")

    store.forget(tmp_path / "a")
    assert not (tmp_path / "trust.json").exists()
    store.trust(tmp_path / "b", CONTENT)
    before = (tmp_path / "trust.json").stat().st_mtime_ns
    store.forget(tmp_path / "a")

    assert (tmp_path / "trust.json").stat().st_mtime_ns == before


def test_a_failed_forget_leaves_the_old_file_and_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = TrustStore(tmp_path / "trust.json")
    store.trust(tmp_path / "ws", CONTENT)
    before = (tmp_path / "trust.json").read_bytes()

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        store.forget(tmp_path / "ws")

    assert (tmp_path / "trust.json").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["trust.json"]


def test_a_failed_write_leaves_the_old_file_and_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "trust.json"
    store = TrustStore(path)
    store.trust(tmp_path / "ws", CONTENT)
    before = path.read_bytes()

    def broken(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="disk full"):
        store.trust(tmp_path / "other", b"{}")

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["trust.json"]
