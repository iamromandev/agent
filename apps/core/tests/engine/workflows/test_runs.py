from __future__ import annotations

import json
from pathlib import Path

from core.engine.workflows.runs import RunEntry, RunIndex


def index(tmp_path: Path) -> RunIndex:
    return RunIndex(tmp_path / "data" / "agent" / "workflows" / "abc.json")


def test_for_workspace_lives_in_the_data_folder(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    runs = RunIndex.for_workspace(workspace, {"XDG_DATA_HOME": str(tmp_path / "data")})

    assert runs.path.parent == tmp_path / "data" / "agent" / "workflows"
    assert runs.path.suffix == ".json" and len(runs.path.stem) == 16
    assert RunIndex.for_workspace(workspace / "x", {"XDG_DATA_HOME": str(tmp_path / "data")}).path != runs.path


def test_empty_index_finds_nothing(tmp_path: Path) -> None:
    runs = index(tmp_path)
    assert runs.latest_unfinished() is None
    assert runs.find("x") is None


def test_record_and_find(tmp_path: Path) -> None:
    runs = index(tmp_path)

    runs.record("r1", "reel", "workflows/reel/r1", "failed")

    assert runs.find("r1") == RunEntry("r1", "reel", "workflows/reel/r1", "failed")
    assert RunIndex(runs.path).find("r1") == RunEntry("r1", "reel", "workflows/reel/r1", "failed")  # on disk


def test_record_updates_the_state_of_a_run(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.record("r1", "reel", "f", "running")
    runs.record("r1", "reel", "f", "done")

    entry = runs.find("r1")
    assert entry is not None and entry.state == "done"
    assert len(json.loads(runs.path.read_text())["runs"]) == 1


def test_latest_unfinished_is_the_newest_not_done(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.record("r1", "reel", "f1", "failed")
    runs.record("r2", "reel", "f2", "running")
    runs.record("r3", "reel", "f3", "done")

    latest = runs.latest_unfinished()
    assert latest is not None and latest.run == "r2"


def test_latest_unfinished_skips_entries_the_caller_cannot_use(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.record("r1", "reel", "f1", "failed")
    runs.record("r2", "reel", "f2", "running")

    latest = runs.latest_unfinished(lambda entry: entry.run != "r2")

    assert latest is not None and latest.run == "r1"
    assert runs.latest_unfinished(lambda entry: False) is None


def test_latest_unfinished_is_none_when_all_done(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.record("r1", "reel", "f1", "done")
    assert runs.latest_unfinished() is None


def test_latest_is_the_newest_run_of_any_state(tmp_path: Path) -> None:
    runs = index(tmp_path)
    assert runs.latest() is None
    runs.record("r1", "reel", "f1", "failed")
    runs.record("r2", "reel", "f2", "done")

    latest = runs.latest()
    assert latest is not None and latest.run == "r2"


def test_latest_skips_entries_the_caller_cannot_use(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.record("r1", "reel", "f1", "done")
    runs.record("r2", "reel", "f2", "done")

    latest = runs.latest(lambda entry: entry.run != "r2")

    assert latest is not None and latest.run == "r1"
    assert runs.latest(lambda entry: False) is None


def test_updating_an_old_run_makes_it_the_newest(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.record("r1", "reel", "f1", "failed")
    runs.record("r2", "reel", "f2", "failed")
    runs.record("r1", "reel", "f1", "failed")

    latest = runs.latest_unfinished()
    assert latest is not None and latest.run == "r1"


def test_only_the_last_fifty_are_kept(tmp_path: Path) -> None:
    runs = index(tmp_path)
    for n in range(60):
        runs.record(f"r{n:02}", "reel", f"f{n}", "failed")

    assert runs.find("r09") is None
    assert runs.find("r10") is not None
    assert len(json.loads(runs.path.read_text())["runs"]) == 50


def test_find_takes_a_unique_prefix(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.record("20261120-1000-cats", "reel", "f1", "failed")
    runs.record("20261121-1000-dogs", "reel", "f2", "failed")

    found = runs.find("20261121")
    assert found is not None and found.folder == "f2"
    assert runs.find("2026112") is None  # ambiguous


def test_a_corrupt_index_is_an_empty_one(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.path.parent.mkdir(parents=True)
    runs.path.write_text("{nope")

    assert runs.find("x") is None
    runs.record("r1", "reel", "f", "failed")
    assert runs.find("r1") is not None


def test_the_same_run_id_in_two_folders_keeps_both(tmp_path: Path) -> None:
    runs = index(tmp_path)

    runs.record("r1", "a", "workflows/a/r1", "failed")
    runs.record("r1", "b", "workflows/b/r1", "failed")

    assert [(e.name, e.folder) for e in runs._read()] == [("a", "workflows/a/r1"), ("b", "workflows/b/r1")]


def test_recording_a_run_again_updates_it_in_place_of_adding(tmp_path: Path) -> None:
    runs = index(tmp_path)

    runs.record("r1", "a", "workflows/a/r1", "running")
    runs.record("r1", "a", "workflows/a/r1", "done")

    assert [e.state for e in runs._read()] == ["done"]


def test_an_id_shared_by_two_workflows_finds_the_newest(tmp_path: Path) -> None:
    runs = index(tmp_path)
    runs.record("r1", "a", "workflows/a/r1", "failed")
    runs.record("r1", "b", "workflows/b/r1", "failed")

    entry = runs.find("r1")

    assert entry is not None and entry.name == "b"
