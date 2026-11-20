"""Rendering the facts into the `/init` prompt block: the format, the forms per manager, the fitting rule."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pytest
from src.engine.facts import (
    Facts,
    FolderFacts,
    MakeTarget,
    NodeProject,
    PythonProject,
    render_facts,
)

from tests.engine.facts_fixtures import CI_YML, ROOT_MAKEFILE, assert_block_shape

BIG = 10**6
Manager = Literal["bun", "pnpm", "yarn", "npm"]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _repo(tmp_path: Path) -> Path:
    """The shape of this repo: root makefile and workflow, cli/ and api/ with makefile + pyproject, ui/ with package.json."""
    _write(tmp_path / "makefile", ROOT_MAKEFILE)
    _write(tmp_path / ".github" / "workflows" / "ci.yml", CI_YML)
    for name in ("cli", "api"):
        _write(tmp_path / name / "makefile", "check-ci: # Run lint + type checks without fixing anything. For CI.\ntest: # Run the unit test suite\n")
        _write(
            tmp_path / name / "pyproject.toml",
            f'[project]\nname = "{name}-x"\nrequires-python = "==3.15.0"\n[project.scripts]\n{name} = "a:b"\n[tool.pytest.ini_options]\n[tool.ruff]\n[tool.ty]\n[tool.uv]\n',
        )
    _write(tmp_path / "ui" / "package.json", '{"scripts": {"web:test": "bun test", "web:format.check": "p"}}')
    _write(tmp_path / "ui" / "bun.lock", "")
    (tmp_path / "infra").mkdir()
    return tmp_path


def _folder(
    path: str = "",
    *,
    makefile: str | None = None,
    targets: tuple[MakeTarget, ...] = (),
    python: PythonProject | None = None,
    node: NodeProject | None = None,
) -> FolderFacts:
    return FolderFacts(path, makefile, targets, python, node)


def _facts(workspace: FolderFacts | None = None, subs: tuple[FolderFacts, ...] = (), workflows=(), more=0) -> Facts:
    return Facts(workspace or _folder(), subs, tuple(workflows), more)


def _targets(*names: str, comment: str = "") -> tuple[MakeTarget, ...]:
    return tuple(MakeTarget(n, comment) for n in names)


@pytest.mark.parametrize(
    ("manager", "expected"),
    [
        ("npm", "- npm --prefix ui run build"),
    ],
)
def test_a_subfolders_script_uses_the_managers_own_form(manager: Manager, expected: str) -> None:
    ui = _folder("ui", node=NodeProject("w", ("build",), manager))
    lines = render_facts(_facts(subs=(ui,)), BIG).splitlines()
    assert expected in lines
    assert f"- ui/package.json: w; scripts run with {manager}" in lines


def test_a_workspace_script_is_manager_run_script() -> None:
    node = NodeProject(None, ("dev",), "pnpm")
    lines = render_facts(_facts(_folder(node=node)), BIG).splitlines()
    assert "- pnpm run dev" in lines and "- package.json: scripts run with pnpm" in lines


def test_python_line_variants() -> None:
    py = PythonProject(None, None, ("a", "b"), ("uv",))
    lines = render_facts(_facts(_folder(python=py)), BIG).splitlines()
    assert "- pyproject.toml: script `a`, `b`; uv" in lines
    bare = PythonProject(None, None, (), ())
    assert "- pyproject.toml" in render_facts(_facts(_folder(python=bare)), BIG).splitlines()
    two = PythonProject("n", None, (), ("pytest", "ruff"))
    assert "- pyproject.toml: n; pytest and ruff configured" in render_facts(_facts(_folder(python=two)), BIG).splitlines()
    one = PythonProject("n", None, (), ("ty",))
    assert "- pyproject.toml: n; ty configured" in render_facts(_facts(_folder(python=one)), BIG).splitlines()


# fitting (R9, AC8)


def _many(n: int, comment: str = "") -> tuple[MakeTarget, ...]:
    return _targets(*(f"target-{i}" for i in range(n)), comment=comment)


def _counts(text: str) -> dict[str, int]:
    lines = text.splitlines()

    def n(prefix: str) -> int:
        return sum(line.startswith(prefix) and "… and" not in line for line in lines)

    layout = sum(bool(re.match(r"^- d\d/ \(", line)) for line in lines)

    return {
        "layout": layout,
        "ci": n("- ci"),
        "sub_scripts": n("- bun run --cwd d"),
        "sub_targets": n("- make -C d"),
        "project": n("- pyproject.toml") + n("- package.json") + n("- d0/package.json"),
        "ws_scripts": n("- bun run s"),
        "ws_targets": n("- make t"),
    }


# trims for small models (R9)


def test_the_budget_may_lower_the_subfolder_cap_below_ten() -> None:
    ws = _folder(makefile="makefile", targets=_many(3))
    subs = tuple(_folder(f"s{i}", makefile="makefile", targets=_many(30)) for i in range(4))
    facts = _facts(ws, subs)
    full = render_facts(facts, BIG)
    for budget in range(len(full), 150, -11):
        text = render_facts(facts, budget)
        assert len(text) <= budget
        assert_block_shape(text)
    small = render_facts(facts, len(full) - 400)
    assert 0 < sum(line.startswith("- make -C s") for line in small.splitlines()) < 40
