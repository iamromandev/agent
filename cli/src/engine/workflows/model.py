"""What a workflow is, what a run plans and what it returns. Plain data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

StepKind = Literal["model", "tool", "each"]
Source = Literal["builtin", "global", "project"]


@dataclass(frozen=True)
class Step:
    id: str
    kind: StepKind
    prompt: str = ""  # model: a template
    schema: dict[str, Any] | None = None  # model: the answer's schema
    tool: str = ""  # tool, each
    args: dict[str, Any] = field(default_factory=dict)  # tool, each: a template
    over: str = ""  # each: "<model step>.<path>" naming a list


@dataclass(frozen=True)
class Workflow:
    name: str
    source: Source
    path: Path
    sha256: str  # of the file's bytes
    description: str
    input_help: str
    folder: str  # a template
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class PlannedCall:
    key: str  # "<step>" or "<step>/<i>" (1-based)
    step: str
    index: int | None  # 1-based position inside an `each` step
    tool: str
    args: dict[str, Any]
    digest: str  # sha256 of the canonical JSON of args


@dataclass(frozen=True)
class PlanSummary:
    workflow: str
    folder: str  # relative to the workspace
    run: str
    calls: tuple[PlannedCall, ...]
    counts: dict[str, int]  # calls per tool
    estimate_s: int


@dataclass(frozen=True)
class RunResult:
    ok: bool
    run: str
    folder: str
    line: str
