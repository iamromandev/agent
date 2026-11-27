from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from core.engine.workflows.timings import MAX_BYTES, MAX_RECORDS, TIMINGS_FILE, TimingStore


def test_a_missing_file_has_no_values(tmp_path: Path) -> None:
    assert TimingStore(tmp_path / "t.json").values("speak") == []


def test_added_values_are_saved_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "data" / "t.json"
    TimingStore(path).add("speak", 4.5)
    TimingStore(path).add("speak", 5.5)

    assert TimingStore(path).values("speak") == [4.5, 5.5]
    assert json.loads(path.read_text())["timings"] == 1
    assert [p.name for p in path.parent.iterdir()] == ["t.json"]  # no temp file left


def test_only_the_last_twenty_are_kept(tmp_path: Path) -> None:
    store = TimingStore(tmp_path / "t.json")
    for i in range(25):
        store.add("speak", float(i + 1))

    assert TimingStore(tmp_path / "t.json").values("speak") == [float(i) for i in range(6, 26)]
    assert MAX_RECORDS == 20


def test_none_path_keeps_values_in_memory_and_never_writes(tmp_path: Path) -> None:
    store = TimingStore(None)
    store.add("speak", 3.0)

    assert store.values("speak") == [3.0]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[]",
        '{"timings": 2, "keys": {}}',
        '{"timings": 1, "keys": []}',
        '{"timings": 1, "keys": {"speak": "x"}}',
        "\x00" * 10,
    ],
)
def test_a_corrupt_or_wrongly_shaped_file_gives_no_values(tmp_path: Path, content: str) -> None:
    path = tmp_path / "t.json"
    path.write_text(content)

    assert TimingStore(path).values("speak") == []


def test_an_oversized_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text('{"timings": 1, "keys": {"speak": [1.0]}}' + " " * MAX_BYTES)

    assert TimingStore(path).values("speak") == []


def test_bad_values_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text('{"timings": 1, "keys": {"speak": [1.0, -2, 0, "x", true, null, NaN, Infinity, 3.0]}}')

    assert TimingStore(path).values("speak") == [1.0, 3.0]


def test_non_finite_and_non_positive_adds_are_ignored(tmp_path: Path) -> None:
    store = TimingStore(tmp_path / "t.json")
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        store.add("speak", bad)

    assert store.values("speak") == []
    assert not (tmp_path / "t.json").exists()


HUGE = "1" + "0" * 400


def test_a_huge_integer_in_the_file_is_dropped_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text('{"timings": 1, "keys": {"speak": [' + HUGE + ", 2.0]}}")

    assert TimingStore(path).values("speak") == [2.0]


def test_add_survives_a_huge_integer_in_the_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text('{"timings": 1, "keys": {"speak": [' + HUGE + '], "edit_image": [' + HUGE + "]}}")

    TimingStore(path).add("speak", 3.0)

    assert TimingStore(path).values("speak") == [3.0]


def test_a_huge_integer_add_is_ignored(tmp_path: Path) -> None:
    store = TimingStore(tmp_path / "t.json")

    store.add("speak", 10**400)

    assert store.values("speak") == []


def test_an_unwritable_folder_raises_oserror(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")

    with pytest.raises(OSError):
        TimingStore(blocker / "sub" / "t.json").add("speak", 1.0)


def test_a_symlinked_file_is_not_read(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text('{"timings": 1, "keys": {"speak": [1.0]}}')
    link = tmp_path / "t.json"
    os.symlink(real, link)

    assert TimingStore(link).values("speak") == []


def test_default_lives_in_the_data_folder(tmp_path: Path) -> None:
    store = TimingStore.default({"XDG_DATA_HOME": str(tmp_path)})

    store.add("speak", 2.0)

    assert (tmp_path / "agent" / TIMINGS_FILE).is_file()
