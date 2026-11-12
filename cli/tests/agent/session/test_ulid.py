from __future__ import annotations

import re

from src.agent.session.ulid import new_ulid

CROCKFORD = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def test_ulid_is_26_crockford_characters() -> None:
    assert CROCKFORD.match(new_ulid())


def test_ulid_encodes_the_timestamp_in_the_first_10_characters() -> None:
    assert new_ulid(0)[:10] == "0000000000"
    assert new_ulid(1)[:10] == "0000000001"
    assert new_ulid(32)[:10] == "0000000010"


def test_ulids_sort_by_time() -> None:
    times = [0, 1, 31, 32, 1_000, 1_700_000_000_000, 2**48 - 1]
    ids = [new_ulid(t) for t in times]
    assert ids == sorted(ids)


def test_ulid_defaults_to_the_current_time() -> None:
    import time

    before = int(time.time() * 1000)
    first = new_ulid()
    after = int(time.time() * 1000)
    assert new_ulid(before)[:10] <= first[:10] <= new_ulid(after)[:10]


def test_ulids_are_unique_across_10000_calls() -> None:
    assert len({new_ulid() for _ in range(10_000)}) == 10_000


def test_ulids_in_the_same_millisecond_differ() -> None:
    assert new_ulid(5) != new_ulid(5)
