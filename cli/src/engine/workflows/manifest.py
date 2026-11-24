"""The manifest of a run: what was asked, what the model answered, and which calls are done. One JSON file."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from src.engine.fileio import parse_json, read_regular
from src.engine.workflows.model import PlannedCall, Workflow

MANIFEST = "manifest.json"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
CHUNK = 1 << 20


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def model_steps_digest(wf: Workflow) -> str:
    """sha256 of the canonical JSON of the model steps: when it changes, the saved answers no longer fit."""
    steps = [
        {"id": s.id, "kind": s.kind, "prompt": s.prompt, "schema": s.schema} for s in wf.steps if s.kind == "model"
    ]
    return hashlib.sha256(canonical(steps).encode("utf-8")).hexdigest()


def new_manifest(
    wf: Workflow, run: str, folder: str, text: str, outputs: dict[str, Any], calls: list[PlannedCall]
) -> dict[str, Any]:
    return {
        "manifest": 1,
        "workflow": wf.name,
        "source": wf.source,
        "sha256": wf.sha256,
        "model_steps_sha256": model_steps_digest(wf),
        "input": text,
        "run": run,
        "folder": folder,  # relative to the workspace
        "outputs": outputs,  # the model steps' answers, by step id
        "calls": [
            {
                "key": c.key,
                "tool": c.tool,
                "args": c.args,
                "digest": c.digest,
                "state": "pending",  # pending, done or failed
                "summary": "",
                "output_sha256": None,  # of the call's `path` after it succeeded
            }
            for c in calls
        ],
    }


def load(folder: Path) -> dict[str, Any] | None:
    """The manifest in `folder`, or None when it is missing, not a regular file, not JSON or not a manifest."""
    try:
        data = parse_json(read_regular(folder / MANIFEST, MAX_MANIFEST_BYTES, follow=False).decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("manifest") != 1 or not isinstance(data.get("calls"), list):
        return None
    return data


def save(folder: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest: a temporary file in the same folder, flushed to disk, then renamed over the old one."""
    # Before any file exists: a bad value leaves nothing. `allow_nan=False`: the file is standard JSON or not written.
    text = json.dumps(manifest, ensure_ascii=False, indent=1, allow_nan=False)
    folder.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=folder, prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, folder / MANIFEST)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


def file_sha256(path: Path) -> str | None:
    """sha256 of a regular file's bytes, read in chunks, never through a symlink; None for anything else."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            digest = hashlib.sha256()
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
            return digest.hexdigest()
    except OSError:
        return None


CALL_STATES = frozenset({"pending", "done", "failed"})


def _call_shape_ok(call: object) -> bool:
    return (
        isinstance(call, dict)
        and all(isinstance(call.get(k), str) for k in ("key", "tool", "digest", "summary"))
        and isinstance(call.get("args"), dict)
        and call.get("state") in CALL_STATES
        and (call.get("output_sha256") is None or isinstance(call.get("output_sha256"), str))
    )


def shape_ok(manifest: dict[str, Any]) -> bool:
    """Whether every field a resume reads has the type it was written with. `load` only checks that it is a manifest;
    a file edited by hand (or damaged) fails here instead of raising later."""
    return (
        all(isinstance(manifest.get(k), str) for k in ("workflow", "model_steps_sha256", "input"))
        and isinstance(manifest.get("outputs"), dict)
        and all(_call_shape_ok(c) for c in manifest["calls"])
    )
