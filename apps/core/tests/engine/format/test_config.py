"""Reading the `format` key: the switch, the built-in table, the prettier gate, overrides and bad entries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core.engine.format.config import FormatSetup, project_needs_trust, read_format


def found(*names: str):
    def which(name: str) -> str | None:
        return f"/bin/{name}" if name in names else None

    return which


def names(setup: FormatSetup) -> list[str]:
    return [s.name for s in setup.formatters]


def program(folder: Path, name: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def test_no_format_key_is_off(tmp_path: Path) -> None:
    setup = read_format({}, None, tmp_path, which=found("ruff"))
    assert not setup.enabled and setup.formatters == ()


@pytest.mark.parametrize("value", [True, {}])
def test_true_or_an_object_turns_it_on_in_table_order(tmp_path: Path, value: object) -> None:
    setup = read_format({"format": value}, None, tmp_path, which=found("rustfmt", "gofmt", "ruff"))
    assert setup.enabled and names(setup) == ["ruff", "gofmt", "rustfmt"]
    assert setup.formatters[0].argv == ("/bin/ruff", "format", "{file}") and setup.formatters[0].source == "global"
    assert setup.missing == ("prettier",)


def test_prettier_needs_a_config_and_prefers_node_modules(tmp_path: Path) -> None:
    assert names(read_format({"format": True}, None, tmp_path, which=found("prettier"))) == []
    (tmp_path / ".prettierrc").write_text("{}")
    assert read_format({"format": True}, None, tmp_path, which=found("prettier")).formatters[0].argv[0] == "/bin/prettier"
    local = program(tmp_path / "node_modules" / ".bin", "prettier")
    spec = read_format({"format": True}, None, tmp_path, which=found("prettier")).formatters[0]
    assert spec.argv == (local, "--write", "{file}") and spec.source == "project" and spec.fingerprint is not None


def test_a_package_json_prettier_key_counts_as_a_config(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(json.dumps({"prettier": {}}))
    assert names(read_format({"format": True}, None, tmp_path, which=found("prettier"))) == ["prettier"]


def test_false_switches_off_and_a_project_false_wins(tmp_path: Path) -> None:
    setup = read_format({"format": {"servers": {"ruff": False}}}, None, tmp_path, which=found("ruff", "gofmt"))
    assert names(setup) == ["gofmt"]
    setup = read_format({"format": True}, {"format": {"servers": {"gofmt": False}}}, tmp_path, which=found("ruff", "gofmt"))
    assert names(setup) == ["ruff"]


def test_a_custom_entry_adds_a_formatter_and_keeps_its_file_placeholder(tmp_path: Path) -> None:
    tool = program(tmp_path / "tools", "fmt")
    entry = {"command": tool, "args": ["--in-place", "x={file}"], "extensions": [".TXT"], "env": {"A": "1"}}
    setup = read_format({"format": {"servers": {"mine": entry}}}, None, tmp_path, which=found())
    assert names(setup) == ["mine"]
    spec = setup.formatters[0]
    assert spec.argv == (tool, "--in-place", "x={file}") and spec.extensions == {".txt"} and spec.env == {"A": "1"}


def test_bad_entries_are_warnings_and_skipped(tmp_path: Path) -> None:
    servers = {"a": 5, "b": {"command": "nope-not-here", "extensions": [".x"]}, "c": {"command": "x", "extensions": ["x"]}}
    setup = read_format({"format": {"servers": servers}}, None, tmp_path, which=found())
    assert setup.formatters == () and len(setup.warnings) == 3


def test_project_needs_trust_only_for_project_sourced_formatters(tmp_path: Path) -> None:
    assert not project_needs_trust(read_format({"format": True}, None, tmp_path, which=found("ruff")))
    assert project_needs_trust(read_format(None, {"format": True}, tmp_path, which=found("ruff")))


def prettier_spec(workspace: Path):
    return read_format({"format": True}, None, workspace, which=found("prettier")).formatters[0]


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("prettier.config.js", "module.exports = {}"),
        ("prettier.config.mjs", "export default {}"),
        (".prettierrc.js", "module.exports = {}"),
        (".prettierrc.cjs", "module.exports = {}"),
        (".prettierrc.ts", "export default {}"),
        (".prettierrc", '{"plugins": ["./evil.js"]}'),
        (".prettierrc.yaml", "plugins:\n  - ./evil.js\n"),
        ("package.json", '{"prettier": {"plugins": ["./evil.js"]}}'),
        ("package.json", '{"prettier": "./shared-config.js"}'),
    ],
)
def test_a_prettier_config_that_can_run_code_makes_prettier_project_sourced(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text)
    spec = prettier_spec(tmp_path)
    assert spec.source == "project" and spec.evidence
    assert project_needs_trust(read_format({"format": True}, None, tmp_path, which=found("prettier")))


@pytest.mark.parametrize(
    ("name", "text"),
    [
        (".prettierrc", '{"semi": false}'),
        (".prettierrc.json", '{"tabWidth": 4}'),
        (".prettierrc.yaml", "semi: false\n"),
        ("package.json", '{"prettier": {"semi": false}}'),
    ],
)
def test_a_data_only_prettier_config_keeps_the_global_source(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text)
    assert prettier_spec(tmp_path).source == "global"


def test_changing_a_code_config_changes_the_trust_evidence(tmp_path: Path) -> None:
    (tmp_path / "prettier.config.js").write_text("module.exports = {}")
    before = prettier_spec(tmp_path).evidence
    (tmp_path / "prettier.config.js").write_text("require('./evil')")
    assert prettier_spec(tmp_path).evidence != before


def test_a_global_false_stays_off_when_the_project_names_the_formatter(tmp_path: Path) -> None:
    project = {"format": {"servers": {"ruff": {"command": program(tmp_path / "t", "r"), "extensions": [".py"]}}}}
    setup = read_format({"format": {"servers": {"ruff": False}}}, project, tmp_path, which=found("ruff"))
    assert "ruff" not in names(setup)
    setup = read_format({"format": {"servers": {"ruff": False}}}, {"format": True}, tmp_path, which=found("ruff"))
    assert "ruff" not in names(setup)


def test_a_project_false_overrides_a_global_entry(tmp_path: Path) -> None:
    entry = {"command": program(tmp_path / "t", "r"), "extensions": [".py"]}
    setup = read_format({"format": {"servers": {"mine": entry}}}, {"format": {"servers": {"mine": False}}}, tmp_path, which=found())
    assert names(setup) == []
