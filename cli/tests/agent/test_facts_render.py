"""Rendering the facts into the `/init` prompt block: the format, the forms per manager, the fitting rule."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pytest
from src.agent.facts import (
    INTRO,
    CiCommand,
    Facts,
    FolderFacts,
    MakeTarget,
    NodeProject,
    PythonProject,
    Workflow,
    render_facts,
    scan,
)

from tests.agent.facts_fixtures import CI_YML, ROOT_MAKEFILE, assert_block_shape

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


def test_this_repos_shape_renders_every_section(tmp_path: Path) -> None:
    text = render_facts(scan(_repo(tmp_path)), BIG)
    lines = text.splitlines()
    assert lines[0] == INTRO and lines[1] == ""
    for expected in (
        "Project",
        "- cli/pyproject.toml: cli-x, Python ==3.15.0, script `cli`; pytest, ruff and ty configured; uv",
        "- ui/package.json: scripts run with bun",
        "Make targets",
        "- make check — Lint + typecheck every stack",
        "- make test — Run the API and CLI test suites",
        "- make -C cli check-ci — Run lint + type checks without fixing anything. For CI.",
        "- make -C cli test — Run the unit test suite",
        "Package scripts",
        "- bun run --cwd ui web:test",
        "- bun run --cwd ui web:format.check",
        "CI (.github/workflows/ci.yml)",
        "- make check-ci (in api)",
        "- bun run --cwd ui web:test",
        "Layout",
        "- api/ (makefile, pyproject.toml)",
        "- cli/ (makefile, pyproject.toml)",
        "- ui/ (package.json)",
        "- infra/",
    ):
        assert expected in lines, expected
    order = [lines.index(h) for h in ("Project", "Make targets", "Package scripts", "CI (.github/workflows/ci.yml)", "Layout")]
    assert order == sorted(order)
    for header in ("Project", "Make targets", "Package scripts", "CI (.github/workflows/ci.yml)", "Layout"):
        assert lines[lines.index(header) - 1] == ""
    assert not text.endswith("\n")


def test_empty_facts_render_nothing() -> None:
    assert render_facts(_facts(), BIG) == ""
    assert render_facts(_facts(subs=(_folder("a"),), more=3), BIG) == ""


def test_a_workspace_makefile_gives_plain_make_lines_and_no_comment_dash_without_comment() -> None:
    facts = _facts(_folder(makefile="makefile", targets=(MakeTarget("a", ""), MakeTarget("b", "Do b"))))
    lines = render_facts(facts, BIG).splitlines()
    assert "- make a" in lines and "- make b — Do b" in lines
    assert "Layout" not in lines  # no subfolders


@pytest.mark.parametrize(
    ("manager", "expected"),
    [
        ("bun", "- bun run --cwd ui build"),
        ("yarn", "- yarn --cwd ui run build"),
        ("pnpm", "- pnpm --dir ui run build"),
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


def test_ci_commands_with_and_without_a_folder_one_section_per_workflow() -> None:
    flows = [
        Workflow(".github/workflows/a.yml", (CiCommand("x", "api"), CiCommand("y", None))),
        Workflow(".github/workflows/b.yml", (CiCommand("z", None),)),
        Workflow(".github/workflows/empty.yml", ()),
    ]
    lines = render_facts(_facts(workflows=flows), BIG).splitlines()
    assert "CI (.github/workflows/a.yml)" in lines and "- x (in api)" in lines and "- y" in lines
    assert "CI (.github/workflows/b.yml)" in lines and "- z" in lines
    assert "CI (.github/workflows/empty.yml)" not in lines


def test_the_layout_ends_with_the_folders_beyond_the_scan_cap() -> None:
    lines = render_facts(_facts(_folder(makefile="makefile", targets=_targets("a")), subs=(_folder("x"),), more=7), BIG).splitlines()
    assert lines[-2:] == ["- x/", "- … and 7 more"]


# fitting (R9, AC8)


def _many(n: int, comment: str = "") -> tuple[MakeTarget, ...]:
    return _targets(*(f"target-{i}" for i in range(n)), comment=comment)


def test_500_targets_at_a_2048_token_window_fit_with_a_more_line_and_the_first_targets() -> None:
    facts = _facts(_folder(makefile="makefile", targets=_many(500, "Does a thing")))
    budget = int(0.2 * 2048) * 4
    text = render_facts(facts, budget)
    assert 0 < len(text) <= budget
    lines = text.splitlines()
    assert "- make target-0 — Does a thing" in lines
    more = [line for line in lines if line.startswith("- … and ")]
    assert more and more[-1].endswith(" more")
    shown = sum(1 for line in lines if line.startswith("- make "))
    assert more[-1] == f"- … and {500 - shown} more"


def test_subfolder_lists_are_cut_before_the_workspaces_own() -> None:
    ws = _folder(makefile="makefile", targets=_many(12))
    sub = _folder("cli", makefile="makefile", targets=_many(40))
    facts = _facts(ws, (sub,))
    full = render_facts(facts, BIG)
    text = render_facts(facts, len(full) - 200)
    lines = text.splitlines()
    assert sum(line.startswith("- make target-") for line in lines) == 12  # every workspace target survived
    assert sum(line.startswith("- make -C cli target-") for line in lines) < 40


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


def test_the_cut_order_is_layout_ci_subfolders_project_then_workspace() -> None:
    ws = _folder(
        makefile="makefile",
        targets=_targets(*(f"t{i}" for i in range(30))),
        node=NodeProject("n", tuple(f"s{i}" for i in range(30)), "bun"),
        python=PythonProject("p", None, (), ()),
    )
    subs = tuple(
        _folder(
            f"d{k}",
            makefile="makefile",
            targets=_targets(*(f"x{i}" for i in range(10))),
            node=NodeProject(None, tuple(f"y{i}" for i in range(10)), "bun"),
        )
        for k in range(3)
    )
    flow = Workflow(".github/workflows/ci.yml", tuple(CiCommand(f"ci{i}", None) for i in range(30)))
    facts = _facts(ws, subs, [flow])
    order = ["layout", "ci", "sub_scripts", "sub_targets", "project", "ws_scripts", "ws_targets"]
    full = _counts(render_facts(facts, BIG))
    assert all(full[name] > 0 for name in order)
    seen_cut = []
    for budget in range(len(render_facts(facts, BIG)), 100, -7):
        text = render_facts(facts, budget)
        assert len(text) <= budget
        counts = _counts(text)
        for index, name in enumerate(order):
            if counts[name] < full[name]:
                # a group is cut only once every group before it in the order is at its floor
                # (the workspace's lists stop at 1; their `… and N more` line is not counted)
                assert all(counts[earlier] == 0 for earlier in order[:index] if earlier not in ("ws_scripts",)), (
                    budget,
                    name,
                    counts,
                )
                seen_cut.append(name)
    assert {"layout", "ci", "sub_scripts", "sub_targets", "project"} <= set(seen_cut)


def test_a_grid_of_budgets_never_overflows_and_a_tiny_one_gives_nothing(tmp_path: Path) -> None:
    facts = scan(_repo(tmp_path))
    for budget in range(0, 3000, 17):
        text = render_facts(facts, budget)
        assert len(text) <= budget
        assert_block_shape(text)
    assert render_facts(facts, 10) == ""


def test_a_budget_that_fits_one_line_of_the_workspace_targets_keeps_them() -> None:
    facts = _facts(_folder(makefile="makefile", targets=_many(100)))
    text = render_facts(facts, len(INTRO) + 60)
    assert text.startswith(INTRO)
    assert "- make target-0" in text.splitlines()


# trims for small models (R9)


def test_a_help_target_without_a_comment_is_not_listed_but_one_with_a_comment_is() -> None:
    ws = _folder(makefile="makefile", targets=(MakeTarget("help", ""), MakeTarget("a", "")))
    sub = _folder("cli", makefile="makefile", targets=(MakeTarget("help", ""), MakeTarget("b", "")))
    lines = render_facts(_facts(ws, (sub,)), BIG).splitlines()
    assert "- make a" in lines and "- make -C cli b" in lines
    assert not any("help" in line for line in lines)
    ws = _folder(makefile="makefile", targets=(MakeTarget("help", "Show"),))
    assert "- make help — Show" in render_facts(_facts(ws), BIG).splitlines()
    assert any(t.name == "help" for t in ws.targets)  # the value keeps it


def test_with_a_workspace_makefile_each_subfolder_shows_ten_targets_then_a_more_line() -> None:
    ws = _folder(makefile="makefile", targets=_many(3))
    subs = (
        _folder("a", makefile="makefile", targets=(*_many(25), MakeTarget("help", ""))),
        _folder("b", makefile="makefile", targets=_many(10)),
        _folder("c", makefile="makefile", targets=_many(11)),
    )
    lines = render_facts(_facts(ws, subs), BIG).splitlines()
    shown = lambda sub: sum(line.startswith(f"- make -C {sub} target-") for line in lines)  # noqa: E731
    assert (shown("a"), shown("b"), shown("c")) == (10, 10, 10)
    mores = [line for line in lines if line.startswith("- … and ") and "more" in line]
    assert mores == ["- … and 15 more", "- … and 1 more"]
    i = lines.index("- make -C a target-9")
    assert lines[i + 1] == "- … and 15 more" and lines[i + 2] == "- make -C b target-0"


def test_without_a_workspace_makefile_subfolder_lists_start_uncapped() -> None:
    subs = (_folder("a", makefile="makefile", targets=_many(25)),)
    lines = render_facts(_facts(_folder(), subs), BIG).splitlines()
    assert sum(line.startswith("- make -C a target-") for line in lines) == 25
    assert not any("more" in line for line in lines)


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
