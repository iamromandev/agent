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
