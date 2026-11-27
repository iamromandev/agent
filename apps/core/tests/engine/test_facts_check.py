"""check_commands, check_notice and init_check (R11-R13, AC10-AC13)."""

from __future__ import annotations

import random
import string
from pathlib import Path

import pytest
from core.engine.facts import CheckResult, check_commands, check_notice, init_check, scan


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return _tree(
        tmp_path,
        {
            "makefile": "test:\ncheck:\ncli-run:\n",
            "cli/makefile": "check-ci:\ntest:\n",
            "ui/package.json": '{"scripts": {"web:test": "x", "build": "y"}}',
            "ui/bun.lock": "",
            "web/makefile": "",
            "web/package.json": '{"scripts": {"dev": "x"}}',
            "empty/placeholder.txt": "",
        },
    )


@pytest.fixture
def ws2(tmp_path: Path) -> Path:
    return _tree(
        tmp_path,
        {
            "makefile": "test:\nbuild:\n",
            "cli/makefile": "check:\nrun:\n",
            "ui/package.json": '{"scripts": {"dev": "x", "web:test": "y"}}',
            "ui/bun.lock": "",
        },
    )


def _check(ws: Path, text: str) -> CheckResult:
    return check_commands(text, scan(ws))


def test_a_makefile_flag_means_not_checked(ws: Path) -> None:
    for form in ("make -f other.mk x", "make --file=o.mk x", "make --makefile o.mk x", "make -fo.mk x"):
        assert _check(ws, f"`{form}`") == CheckResult(0, ()), form


def test_directory_forms(ws: Path) -> None:
    for form in ("make -C cli test", "make -Ccli test", "make --directory cli test", "make --directory=cli test"):
        assert _check(ws, f"`{form}`") == CheckResult(1, ()), form
    assert _check(ws, "`make -C ./cli/ nope`").missing == ("make -C ./cli/ nope",)


def test_run_is_needed(ws: Path) -> None:
    for form in ("npm test", "yarn build", "bun dev", "npm install", "npm run"):
        assert _check(ws, f"`{form}`") == CheckResult(0, ()), form


def test_the_script_skips_option_values(ws: Path) -> None:
    assert _check(ws, "`bun run --silent build --cwd ui`") == CheckResult(1, ())


def test_a_workspace_filter_with_a_value_is_not_checked(ws: Path) -> None:
    assert _check(ws, "`npm run -w x build`") == CheckResult(0, ())


def test_check_commands_never_raises_on_random_text(ws: Path) -> None:
    rng = random.Random(7)
    pieces = ["`", "```", "~~~", "make ", "npm run ", "-C ", "--cwd=", "\n", "&&", ";", "|", '"', "'", "$ ", "\\", "x", "=", " "]
    facts = scan(ws)
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 60)))
        text += "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 40)))
        result = check_commands(text, facts)
        assert result.checked >= len(result.missing)
    for _ in range(50):
        check_commands(bytes(rng.randrange(256) for _ in range(200)).decode("utf-8", "replace"), facts)


# check_notice and init_check


def test_the_warning_lists_ten_and_counts_the_rest() -> None:
    missing = tuple(f"make m{i}" for i in range(14))
    level, text = check_notice(CheckResult(20, missing)) or ("", "")
    assert level == "warn"
    assert text == (
        "/init check: not found by the scan: " + ", ".join(missing[:10]) + " and 4 more. "
        "They may still be right; check before relying on them."
    )


def test_the_warning_without_overflow() -> None:
    assert check_notice(CheckResult(3, ("make lint",))) == (
        "warn",
        "/init check: not found by the scan: make lint. They may still be right; check before relying on them.",
    )


def test_the_info_notices() -> None:
    assert check_notice(CheckResult(3, ())) == ("info", "/init check: all 3 make and script commands were found by the scan")
    assert check_notice(CheckResult(1, ())) == ("info", "/init check: the 1 make or script command was found by the scan")
    assert check_notice(CheckResult(0, ())) is None


def test_init_check_without_a_readable_file_is_none(tmp_path: Path) -> None:
    assert init_check(tmp_path) is None
    (tmp_path / "AGENTS.md").mkdir()
    assert init_check(tmp_path) is None


# Review fixes: shell reading of commands

NONE = CheckResult(0, ())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```\ncd\nmake x\n```", NONE),
        ("```\ncd ~/x\nmake x\n```", NONE),
    ],
)
def test_cd_sets_the_folder_for_the_commands_after_it(ws2: Path, text: str, expected: CheckResult) -> None:
    assert _check(ws2, text) == expected


@pytest.mark.parametrize("form", ["bun run x.sh"])
def test_bun_run_of_a_file_is_not_checked(ws: Path, form: str) -> None:
    assert _check(ws, f"`{form}`") == NONE


@pytest.mark.parametrize(
    "form",
    ["make -o foo test", "make --what-if foo test", "make -j 4 test"],
)
def test_make_options_that_take_a_value(ws: Path, form: str) -> None:
    assert _check(ws, f"`{form}`") == CheckResult(1, ()), form


def test_make_j_without_a_number_keeps_the_next_word_as_a_target(ws: Path) -> None:
    assert _check(ws, "`make -j nope`").missing == ("make nope",)


def test_a_script_after_double_dash_flags_is_not_read(ws: Path) -> None:
    assert _check(ws, "`npm run --cwd ui build -- --filter x nope`") == CheckResult(1, ())
    assert _check(ws, "`npm run --cwd ui nope -- --flag`").missing == ("npm run --cwd ui nope",)


# fences


# trailing punctuation


# Re-review fixes


@pytest.mark.parametrize(
    "text",
    [
        "```\nmake -j$(nproc) test && cd cli\nmake check\n```",
    ],
)
def test_parentheses_scope_a_cd(ws2: Path, text: str) -> None:
    result = _check(ws2, text)
    assert result.missing == (), text
    assert result.checked >= 1


@pytest.mark.parametrize("form", ["pushd"])
def test_pushd_popd_and_dirs_make_the_folder_unknown(ws2: Path, form: str) -> None:
    assert _check(ws2, f"`{form} cli && make check && popd && make test`") == NONE


def test_combined_make_flags(ws2: Path) -> None:
    assert _check(ws2, "`make -sC cli check`") == CheckResult(1, ())
    assert _check(ws2, "`make -kC cli check`") == CheckResult(1, ())
    assert _check(ws2, "`make -sk test`") == CheckResult(1, ())
    assert _check(ws2, "`make -sf other.mk x`") == NONE
    assert _check(ws2, "`make -kC cli nope`").missing == ("make -C cli nope",)


def test_an_unknown_package_option_with_a_value_means_not_checked(ws2: Path) -> None:
    assert _check(ws2, "`bun run --env-file .env dev`") == NONE
    assert _check(ws2, "`bun run --env-file=.env --cwd ui dev`") == CheckResult(1, ())
    assert _check(ws2, "`bun run --cwd ui --foo dev`") == CheckResult(1, ())


def test_a_dash_heredoc_ends_on_a_tab_indented_word(ws2: Path) -> None:
    assert _check(ws2, "```\ncat <<-EOF\nmake nope\n\tEOF\nmake build\n```") == CheckResult(1, ())


@pytest.mark.parametrize("form", ["npm run ..."])
def test_a_word_without_a_letter_or_digit_is_not_checked(ws2: Path, form: str) -> None:
    assert _check(ws2, f"`{form}`") == NONE
