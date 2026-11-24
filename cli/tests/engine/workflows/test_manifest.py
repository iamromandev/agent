from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.engine.workflows.manifest import file_sha256, load, model_steps_digest, new_manifest, save
from src.engine.workflows.model import PlannedCall, Step, Workflow


def workflow(prompt: str = "p", *, tool_args: dict[str, Any] | None = None) -> Workflow:
    steps = (
        Step("story", "model", prompt=prompt, schema={"type": "object"}),
        Step("video", "tool", tool="assemble_video", args=tool_args or {"path": "x"}),
    )
    return Workflow("reel", "project", Path("/w/reel.json"), "f" * 64, "d", "", "workflows/{name}/{run}", steps)


def call(key: str = "video") -> PlannedCall:
    return PlannedCall(key, "video", None, "assemble_video", {"path": "x"}, "d" * 64)


def test_model_steps_digest_depends_only_on_model_steps() -> None:
    assert model_steps_digest(workflow()) == model_steps_digest(workflow(tool_args={"path": "other"}))
    assert model_steps_digest(workflow("p")) != model_steps_digest(workflow("q"))
    assert len(model_steps_digest(workflow())) == 64


def test_new_manifest_has_the_documented_keys() -> None:
    m = new_manifest(workflow(), "20261120-1000-cats", "workflows/reel/20261120-1000-cats", "cats", {"story": {"a": 1}}, [call()])

    assert m["manifest"] == 1
    assert (m["workflow"], m["source"], m["sha256"]) == ("reel", "project", "f" * 64)
    assert m["model_steps_sha256"] == model_steps_digest(workflow())
    assert (m["input"], m["run"], m["folder"]) == ("cats", "20261120-1000-cats", "workflows/reel/20261120-1000-cats")
    assert m["outputs"] == {"story": {"a": 1}}
    assert m["calls"] == [
        {
            "key": "video",
            "tool": "assemble_video",
            "args": {"path": "x"},
            "digest": "d" * 64,
            "state": "pending",
            "summary": "",
            "output_sha256": None,
        }
    ]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    folder = tmp_path / "run"
    m = new_manifest(workflow(), "r", "f", "t", {}, [call()])

    save(folder, m)

    assert load(folder) == m
    assert sorted(p.name for p in folder.iterdir()) == ["manifest.json"]  # no temp file left


def test_save_replaces_atomically_and_keeps_the_old_file_on_failure(tmp_path: Path) -> None:
    m = new_manifest(workflow(), "r", "f", "t", {}, [call()])
    save(tmp_path, m)

    class Boom:
        pass

    broken = {**m, "outputs": {"x": Boom()}}  # not JSON: the write fails before the replace
    with contextlib.suppress(TypeError):
        save(tmp_path, broken)

    assert load(tmp_path) == m
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_load_missing_or_corrupt_is_none(tmp_path: Path) -> None:
    assert load(tmp_path) is None
    (tmp_path / "manifest.json").write_text("{nope")
    assert load(tmp_path) is None
    (tmp_path / "manifest.json").write_text("[1]")
    assert load(tmp_path) is None
    (tmp_path / "manifest.json").write_text(json.dumps({"manifest": 2}))
    assert load(tmp_path) is None


def test_file_sha256_matches_hashlib_for_a_big_file(tmp_path: Path) -> None:
    data = os.urandom(3 * 1024 * 1024 + 7)
    path = tmp_path / "f.bin"
    path.write_bytes(data)

    assert file_sha256(path) == hashlib.sha256(data).hexdigest()


def test_file_sha256_refuses_missing_symlink_and_folder(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.write_text("x")
    (tmp_path / "link").symlink_to(real)
    (tmp_path / "dir").mkdir()

    assert file_sha256(tmp_path / "missing") is None
    assert file_sha256(tmp_path / "link") is None
    assert file_sha256(tmp_path / "dir") is None
    assert file_sha256(real) == hashlib.sha256(b"x").hexdigest()


def test_a_failed_call_may_carry_an_error_string() -> None:
    from src.engine.workflows.manifest import shape_ok

    m = new_manifest(workflow(), "r", "f", "t", {}, [call()])
    assert shape_ok(m)
    m["calls"][0].update(state="failed", error="scene 2 is 0.8 s")
    assert shape_ok(m)
    m["calls"][0]["error"] = 5
    assert not shape_ok(m)
