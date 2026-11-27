"""The shared output placement: path checks and atomic placement of bytes or a copied file."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
from core.engine.guardrails.setup import Guardrails
from core.engine.tools.base import ToolContext, ToolResult
from core.engine.tools.outputs import check_output_path, place_bytes, place_file


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def before(self, path: Path) -> None:
        self.calls.append(("before", path))

    def after(self, path: Path) -> None:
        self.calls.append(("after", path))


def journal_of(ctx: ToolContext) -> Journal:
    assert isinstance(ctx.journal, Journal)
    return ctx.journal


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, bash_timeout=5.0, guard=Guardrails.off(workspace), journal=Journal())  # type: ignore[arg-type]


@pytest.fixture
def source(tmp_path: Path) -> Path:
    folder = tmp_path / "private"
    folder.mkdir()
    (folder / "speech.wav").write_bytes(b"RIFF-data" * 1000)
    return folder / "speech.wav"


def test_the_suffix_is_checked_and_named(ctx: ToolContext, workspace: Path) -> None:
    assert check_output_path("a.wav", ctx, ".wav") == workspace / "a.wav"
    refused = check_output_path("a.png", ctx, ".wav")
    assert isinstance(refused, ToolResult)
    assert refused.output == "a.png must end in .wav"
    again = check_output_path("a.wav", ctx)
    assert isinstance(again, ToolResult)
    assert again.output == "a.wav must end in .png"


def test_a_protected_path_is_refused_by_the_check(workspace: Path) -> None:
    class Paths:
        def is_protected(self, path: Path) -> bool:
            return path.name == "s.wav"

    guard = dataclasses.replace(Guardrails.off(workspace), paths=Paths())  # type: ignore[arg-type]
    refused = check_output_path("s.wav", ToolContext(workspace, 5.0, guard=guard), ".wav")
    assert isinstance(refused, ToolResult)
    assert refused.output == "protected path"


def test_place_file_copies_by_rename_and_journals(ctx: ToolContext, workspace: Path, source: Path) -> None:
    target = workspace / "out" / "a.wav"
    assert place_file(target, source, ctx) is None
    assert target.read_bytes() == source.read_bytes()
    assert source.exists()
    assert not [p for p in target.parent.iterdir() if p.name != "a.wav"]
    assert journal_of(ctx).calls == [("before", target), ("after", target)]


def test_place_file_keeps_the_mode_of_the_file_it_replaces(ctx: ToolContext, workspace: Path, source: Path) -> None:
    target = workspace / "a.wav"
    target.write_bytes(b"old")
    target.chmod(0o640)
    assert place_file(target, source, ctx) is None
    assert target.stat().st_mode & 0o777 == 0o640
    assert target.read_bytes() == source.read_bytes()


def test_place_file_refuses_a_symlinked_parent_that_leaves_the_workspace(
    ctx: ToolContext, workspace: Path, source: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside)
    refused = place_file(workspace / "link" / "a.wav", source, ctx)
    assert refused is not None and refused.is_error
    assert list(outside.iterdir()) == []


def test_place_file_refuses_a_protected_target(workspace: Path, source: Path) -> None:
    class Paths:
        def is_protected(self, path: Path) -> bool:
            return path.name == "s.wav"

    guard = dataclasses.replace(Guardrails.off(workspace), paths=Paths())  # type: ignore[arg-type]
    refused = place_file(workspace / "s.wav", source, ToolContext(workspace, 5.0, guard=guard))
    assert refused is not None and refused.is_error
    assert not (workspace / "s.wav").exists()


def test_place_file_never_reads_through_a_symlink(ctx: ToolContext, workspace: Path, source: Path) -> None:
    link = source.parent / "link.wav"
    link.symlink_to(source)
    target = workspace / "a.wav"
    with pytest.raises(OSError):
        place_file(target, link, ctx)
    assert not target.exists()
    assert [p.name for p in workspace.iterdir()] == []


def test_place_bytes_writes_and_leaves_no_temp(ctx: ToolContext, workspace: Path) -> None:
    assert place_bytes(workspace / "a.png", b"data", ctx) is None
    assert (workspace / "a.png").read_bytes() == b"data"
    assert os.listdir(workspace) == ["a.png"]
