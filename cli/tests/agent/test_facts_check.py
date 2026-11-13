"""check_commands, check_notice and init_check (R11-R13, AC10-AC13)."""

from __future__ import annotations

import random
import string
from pathlib import Path

import pytest
from src.agent.facts import CheckResult, check_commands, check_notice, init_check, scan


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


def test_the_mixed_command_list_flags_exactly_three_in_order(ws: Path) -> None:
    text = (
        "Run `make test`, `make lint`, `make -C cli check-ci`, `make -C cli nope`,\n"
        '`make cli-run ARGS="health check"`, `bun run --cwd ui web:test`, `bun run --cwd ui nope`,\n'
        "`make -C ../x y` and `uv run pytest`. Make sure it works; make sure again.\n"
    )
    result = _check(ws, text)
    assert result.missing == ("make lint", "make -C cli nope", "bun run --cwd ui nope")
    assert result.checked == 7


def test_prose_is_not_checked(ws: Path) -> None:
    assert _check(ws, "make sure it works. npm run nothing here.\n") == CheckResult(0, ())


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_fenced_blocks_are_checked(ws: Path, fence: str) -> None:
    text = f"intro\n{fence}bash\nmake nope\nmake test\n{fence}\nmake outside\n"
    assert _check(ws, text) == CheckResult(2, ("make nope",))


def test_an_unclosed_fence_runs_to_the_end(ws: Path) -> None:
    assert _check(ws, "```\nmake nope\n") == CheckResult(1, ("make nope",))


def test_chains_prompts_and_env_words(ws: Path) -> None:
    text = "```\n$ make test && make lint; FOO=1 make nope | cat\n$ make check || make zip\n```\n"
    assert _check(ws, text).missing == ("make lint", "make nope", "make zip")


def test_a_makefile_flag_means_not_checked(ws: Path) -> None:
    for form in ("make -f other.mk x", "make --file=o.mk x", "make --makefile o.mk x", "make -fo.mk x"):
        assert _check(ws, f"`{form}`") == CheckResult(0, ()), form


def test_options_assignments_numbers_and_placeholders_are_ignored(ws: Path) -> None:
    assert _check(ws, "`make -j4 test`") == CheckResult(1, ())
    assert _check(ws, '`make test ARGS="a b"`') == CheckResult(1, ())
    assert _check(ws, "`make <target>`") == CheckResult(0, ())
    assert _check(ws, "`make -j 4 test`").missing == ()


def test_directory_forms(ws: Path) -> None:
    for form in ("make -C cli test", "make -Ccli test", "make --directory cli test", "make --directory=cli test"):
        assert _check(ws, f"`{form}`") == CheckResult(1, ()), form
    assert _check(ws, "`make -C ./cli/ nope`").missing == ("make -C ./cli/ nope",)


def test_a_folder_with_no_makefile_is_flagged_and_an_unscanned_one_is_not(ws: Path) -> None:
    assert _check(ws, "`make -C empty test`") == CheckResult(1, ("make -C empty test",))
    assert _check(ws, "`make -C nowhere test`") == CheckResult(0, ())
    assert _check(ws, "`make -C ../x y`") == CheckResult(0, ())


def test_no_makefile_at_all_flags_make_build(tmp_path: Path) -> None:
    assert check_commands("`make build`", scan(tmp_path)) == CheckResult(1, ("make build",))


def test_a_hidden_target_in_the_prompt_is_still_found(tmp_path: Path) -> None:
    _tree(tmp_path, {"makefile": "help:\n" + "".join(f"t{i}:\n" for i in range(30))})
    assert check_commands("`make help` `make t29`", scan(tmp_path)) == CheckResult(2, ())


def test_package_manager_forms(ws: Path) -> None:
    ok = (
        "yarn --cwd ui run build",
        "pnpm --dir ui run build",
        "npm --prefix ui run build",
        "bun run --cwd=ui build",
        "bun run --cwd ui web:test",
        "npm -C ui run build",
        "yarn run --cwd ui build",
        "npm run --prefix=ui build",
    )
    for form in ok:
        assert _check(ws, f"`{form}`") == CheckResult(1, ()), form
    assert _check(ws, "`npm --prefix ui run nope`").missing == ("npm --prefix ui run nope",)


def test_run_is_needed(ws: Path) -> None:
    for form in ("npm test", "yarn build", "bun dev", "npm install", "npm run"):
        assert _check(ws, f"`{form}`") == CheckResult(0, ()), form


def test_a_folder_with_no_package_json_flags_the_script(ws: Path) -> None:
    assert _check(ws, "`npm run build`").missing == ("npm run build",)
    assert _check(ws, "`npm --prefix cli run build`").missing == ("npm --prefix cli run build",)
    assert _check(ws, "`npm --prefix nowhere run build`") == CheckResult(0, ())


def test_the_script_skips_option_values(ws: Path) -> None:
    assert _check(ws, "`bun run --silent build --cwd ui`") == CheckResult(1, ())


def test_a_workspace_filter_with_a_value_is_not_checked(ws: Path) -> None:
    assert _check(ws, "`npm run -w x build`") == CheckResult(0, ())


def test_duplicates_count_once(ws: Path) -> None:
    result = _check(ws, "`make nope` `make nope` `make test` `make test`")
    assert result == CheckResult(2, ("make nope",))


def test_unbalanced_quotes_do_not_raise(ws: Path) -> None:
    assert _check(ws, "`make test ARGS=\"a`").checked == 1
    assert _check(ws, "```\nmake nope 'x\n```").missing == ("make nope",)


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


def test_init_check_reads_agents_md_and_scans_again(tmp_path: Path) -> None:
    _tree(tmp_path, {"makefile": "test:\n", "AGENTS.md": "`make test` `make lint`\n"})
    assert init_check(tmp_path) == (
        "warn",
        "/init check: not found by the scan: make lint. They may still be right; check before relying on them.",
    )
    (tmp_path / "makefile").write_text("test:\nlint:\n")  # AC13: a target added later is found
    assert init_check(tmp_path) == ("info", "/init check: all 2 make and script commands were found by the scan")


def test_init_check_without_a_readable_file_is_none(tmp_path: Path) -> None:
    assert init_check(tmp_path) is None
    (tmp_path / "AGENTS.md").mkdir()
    assert init_check(tmp_path) is None


def test_init_check_ignores_a_file_over_the_cap(tmp_path: Path) -> None:
    _tree(tmp_path, {"AGENTS.md": "`make nope`\n" + "x" * (256 * 1024)})
    assert init_check(tmp_path) is None


def test_init_check_does_not_follow_a_link_out_of_the_workspace(tmp_path: Path) -> None:
    outside = tmp_path / "out.md"
    outside.write_text("`make nope`")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "AGENTS.md").symlink_to(outside)
    assert init_check(ws) is None


# Review fixes: shell reading of commands

NONE = CheckResult(0, ())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```sh\nmake test   # run all tests\n```", CheckResult(1, ())),
        ("`make -C cli test  # lint`", CheckResult(1, ())),
        ("```\nmake -C cli test  # lint\n```", CheckResult(1, ())),
        ("```\nmake build  # also: make test\n```", CheckResult(1, ("make build",))),
        ("```\nmake test # a; make zzz\n```", CheckResult(1, ())),
        ("```\nmake test > out.txt\n```", CheckResult(1, ())),
        ("```\nmake test >out.txt 2>&1\n```", CheckResult(1, ())),
        ("```\nmake test >> log nope\n```", CheckResult(1, ())),
        ("```\nmake test < in.txt\n```", CheckResult(1, ())),
        ("```\nmake test 2> err.txt\n```", CheckResult(1, ())),
        ("```\nmake test &> all.txt\n```", CheckResult(1, ())),
        ("```\nmake test 2>&1 | cat\n```", CheckResult(1, ())),
    ],
)
def test_comments_and_redirections_are_not_part_of_the_command(ws: Path, text: str, expected: CheckResult) -> None:
    assert _check(ws, text) == expected


def test_a_comment_with_an_unbalanced_quote_still_drops(ws: Path) -> None:
    assert _check(ws, "```\nmake test  # it's\n```") == CheckResult(1, ())


def test_separators_respect_quotes_and_a_single_ampersand_splits(ws: Path) -> None:
    assert _check(ws, "`echo 'x; make zzz y'`") == NONE
    assert _check(ws, '`echo "a && make zzz"`') == NONE
    assert _check(ws, "`make test & make nope`").missing == ("make nope",)


def test_a_single_ampersand_splits_package_commands(ws: Path) -> None:
    assert _check(ws, "`bun run --cwd web dev & bun run --cwd web nope`").missing == ("bun run --cwd web nope",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("`cd cli && make check`", CheckResult(1, ())),
        ("`cd cli && make test`", CheckResult(1, ("make test",))),
        ("`cd cli && make nope`", CheckResult(1, ("make nope",))),
        ("`make test`", CheckResult(1, ())),
        ("```\ncd cli\nmake check\n```", CheckResult(1, ())),
        ("```\ncd cli\ncd ..\nmake test\n```", CheckResult(1, ())),
        ("```\ncd cli\nmake test\n```", CheckResult(1, ("make test",))),
        ("`cd /tmp && make x`", NONE),
        ("`cd $DIR && make x`", NONE),
        ("```\ncd\nmake x\n```", NONE),
        ("```\ncd -\nmake x\n```", NONE),
        ("```\ncd ~/x\nmake x\n```", NONE),
        ("`cd nowhere && make x`", NONE),
        ("`cd cli && make -C .. build`", CheckResult(1, ())),
        ("`cd .. && make x`", NONE),
        ("```\n(cd cli && make check)\nmake test\n```", CheckResult(2, ())),
        ("`(cd cli && make check)`", CheckResult(1, ())),
        ("`make -C cli -C sub t`", NONE),
        ("`cd ui && make -C ../cli check`", CheckResult(1, ())),
    ],
)
def test_cd_sets_the_folder_for_the_commands_after_it(ws2: Path, text: str, expected: CheckResult) -> None:
    assert _check(ws2, text) == expected


def test_two_spans_do_not_share_a_cd(ws2: Path) -> None:
    assert _check(ws2, "`cd cli` then `make test`") == CheckResult(1, ())
    assert _check(ws2, "`cd cli` then `make check`") == CheckResult(1, ("make check",))


def test_a_cd_applies_to_package_commands_too(ws2: Path) -> None:
    assert _check(ws2, "`cd ui && bun run web:test`") == CheckResult(1, ())
    assert _check(ws2, "`cd ui && bun run nope`") == CheckResult(1, ("bun run nope",))
    assert _check(ws2, "`cd cli && bun run --cwd ../ui web:test`") == CheckResult(1, ())


@pytest.mark.parametrize("form", ["bun run src/index.ts", "bun run ./x.ts", "bun run build.ts", "bun run a.mjs", "bun run x.sh"])
def test_bun_run_of_a_file_is_not_checked(ws: Path, form: str) -> None:
    assert _check(ws, f"`{form}`") == NONE


def test_bun_run_of_a_script_is_still_checked(ws: Path) -> None:
    assert _check(ws, "`bun run --cwd ui web:test`") == CheckResult(1, ())
    assert _check(ws, "`npm run build.ts`").checked == 1  # only bun treats files specially


@pytest.mark.parametrize(
    "form",
    [
        "pnpm run --filter x nope",
        "pnpm --filter=x run nope",
        "pnpm -F x run nope",
        "npm run -w ui nope",
        "npm run --workspace=ui nope",
        "npm run --workspace ui nope",
        "npm run --workspaces nope",
    ],
)
def test_a_workspace_filter_means_not_checked(ws: Path, form: str) -> None:
    assert _check(ws, f"`{form}`") == NONE


@pytest.mark.parametrize(
    "form",
    ["make -I inc test", "make -o foo test", "make -W foo test", "make -l 2.5 test", "make --include-dir inc test",
     "make --old-file foo test", "make --what-if foo test", "make --load-average 2.5 test", "make -j 4 test",
     "make --jobs 4 test", "make -Iinc test", "make --include-dir=inc test", "make -j4 test", "make --jobs=4 test"],
)
def test_make_options_that_take_a_value(ws: Path, form: str) -> None:
    assert _check(ws, f"`{form}`") == CheckResult(1, ()), form


def test_make_j_without_a_number_keeps_the_next_word_as_a_target(ws: Path) -> None:
    assert _check(ws, "`make -j nope`").missing == ("make nope",)


def test_a_script_after_double_dash_flags_is_not_read(ws: Path) -> None:
    assert _check(ws, "`npm run --cwd ui build -- --filter x nope`") == CheckResult(1, ())
    assert _check(ws, "`npm run --cwd ui nope -- --flag`").missing == ("npm run --cwd ui nope",)


def test_only_exact_program_names_count(ws: Path) -> None:
    text = "`make.sh deploy` `makefile-lint x` `cmake build` `remake y` `npmx run z` `bunx run q` `npx run r`"
    assert _check(ws, text) == CheckResult(0, ())


# fences


def test_an_inline_triple_backtick_span_is_not_a_fence(ws: Path) -> None:
    assert _check(ws, "```foo``` and then\nmake sure it works\nmake nope\n") == NONE


def test_a_backtick_in_the_info_string_is_not_a_fence(ws: Path) -> None:
    assert _check(ws, "``` a`b\nmake nope\n") == NONE


def test_a_longer_fence_needs_a_closer_as_long(ws: Path) -> None:
    text = "````md\n```\nmake nope\n```\nmake zip\n````\nmake outside\n"
    assert _check(ws, text) == CheckResult(2, ("make nope", "make zip"))


def test_a_fence_closes_only_on_its_own_character_and_bare(ws: Path) -> None:
    text = "~~~\n```\nmake nope\n~~~ x\nmake zip\n~~~\nmake outside\n"
    assert _check(ws, text) == CheckResult(2, ("make nope", "make zip"))


def test_an_indented_fence_opens_and_four_spaces_do_not(ws: Path) -> None:
    assert _check(ws, "   ```\nmake nope\n   ```\n") == CheckResult(1, ("make nope",))
    assert _check(ws, "    ```\nmake nope\n") == NONE


# trailing punctuation


def test_trailing_punctuation_in_a_span_is_dropped(ws: Path) -> None:
    assert _check(ws, "`make test.` `make test,` `make check:`") == CheckResult(2, ())
    assert _check(ws, "`make foo.o`").missing == ("make foo.o",)
    assert _check(ws, "`cd cli && cd .. && make test`") == CheckResult(1, ())


def test_trailing_punctuation_in_a_fence_is_kept(ws: Path) -> None:
    assert _check(ws, "```\nmake foo.\n```") == CheckResult(1, ("make foo.",))


def test_check_commands_never_raises_and_is_fast_on_shell_tokens(ws: Path) -> None:
    import time

    rng = random.Random(11)
    tokens = [
        "make", "-C", "cli", "cd", "&&", ";", "#", ">", "`", "```", "'", '"', "test", "nope", "\n", " ", " ",
        "(", ")", "$(", "pushd", "<<EOF", "EOF",
    ]  # fmt: skip
    facts = scan(ws)
    for _ in range(300):
        text = "".join(rng.choice(tokens) + rng.choice(["", " "]) for _ in range(rng.randint(0, 40)))
        start = time.perf_counter()
        result = check_commands(text, facts)
        assert time.perf_counter() - start < 0.05, text
        assert result.checked >= len(result.missing)


# Re-review fixes


@pytest.mark.parametrize(
    "text",
    [
        "`(cd cli && make check) && make test`",
        "`(cd cli && make check); make test`",
        "```\n(\n  cd cli\n  make check\n)\nmake test\n```",
        "```\nmake -j$(nproc) test && cd cli\nmake check\n```",
        "`$(cd cli && make check) && make test`",
        "`(cd cli && (cd .. && make test) && make check)`",
        '`echo "(" && cd cli && make check`',
    ],
)
def test_parentheses_scope_a_cd(ws2: Path, text: str) -> None:
    result = _check(ws2, text)
    assert result.missing == (), text
    assert result.checked >= 1


def test_parentheses_scope_flags_only_the_real_miss(ws2: Path) -> None:
    assert _check(ws2, "`(cd cli && make nope) && make test`") == CheckResult(2, ("make nope",))


def test_an_unbalanced_close_is_ignored_and_an_unclosed_open_ends_with_the_piece(ws2: Path) -> None:
    assert _check(ws2, "`) && make test`") == CheckResult(1, ())
    assert _check(ws2, "```\n(cd cli\nmake check\n```\n`make test`") == CheckResult(2, ())


@pytest.mark.parametrize("form", ["pushd", "popd", "dirs"])
def test_pushd_popd_and_dirs_make_the_folder_unknown(ws2: Path, form: str) -> None:
    assert _check(ws2, f"`{form} cli && make check && popd && make test`") == NONE


def test_a_cd_after_pushd_stays_unknown(ws2: Path) -> None:
    assert _check(ws2, "```\npushd cli\ncd ..\nmake nope\n```") == NONE


@pytest.mark.parametrize(
    "text",
    [
        "```\n" + "cd ..\n" * 40000 + "make nope\n```",
        "```\n" + "cd cli\n" * 37000 + "make nope\n```",
        "```\n" + "cd;" * 85000 + "\n```",
        "```\n" + "x" * 256000 + "\n```",
    ],
)
def test_huge_inputs_finish_in_bounded_time(ws2: Path, text: str) -> None:
    import time

    start = time.perf_counter()
    result = _check(ws2, text)
    assert time.perf_counter() - start < 1.0
    assert result.missing == ()


def test_cd_below_a_scanned_folder_is_conservative(ws2: Path) -> None:
    assert _check(ws2, "`cd cli/sub && cd ../.. && make nope`") == NONE
    assert _check(ws2, "`cd cli && cd .. && make test`") == CheckResult(1, ())


def test_a_line_over_two_thousand_characters_is_skipped(ws2: Path) -> None:
    assert _check(ws2, "```\nmake nope " + "x" * 2000 + "\nmake nope\n```") == CheckResult(1, ("make nope",))


@pytest.mark.parametrize(
    "form",
    [
        "pnpm -r run test",
        "pnpm run -r web:test",
        "pnpm --recursive run x",
        "npm run web:test -ws",
        "npm run dev --include-workspace-root",
    ],
)
def test_recursive_and_workspace_flags_are_not_checked(ws2: Path, form: str) -> None:
    assert _check(ws2, f"`{form}`") == NONE


def test_bun_dash_r_takes_a_value(ws2: Path) -> None:
    assert _check(ws2, "`bun run -r ./pre.ts nope`") == NONE
    assert _check(ws2, "`bun -r ./pre.ts run nope`") == NONE


def test_combined_make_flags(ws2: Path) -> None:
    assert _check(ws2, "`make -sC cli check`") == CheckResult(1, ())
    assert _check(ws2, "`make -kC cli check`") == CheckResult(1, ())
    assert _check(ws2, "`make -sk test`") == CheckResult(1, ())
    assert _check(ws2, "`make -sf other.mk x`") == NONE
    assert _check(ws2, "`make -kC cli nope`").missing == ("make -C cli nope",)


@pytest.mark.parametrize("form", ["make -j4 test", "make -Ccli check", "make -fother.mk x", "make -sj4 test"])
def test_attached_make_flag_values_keep_working(ws2: Path, form: str) -> None:
    expected = NONE if "-f" in form else CheckResult(1, ())
    assert _check(ws2, f"`{form}`") == expected


def test_a_cluster_ending_in_a_value_flag_takes_the_next_word(ws2: Path) -> None:
    assert _check(ws2, "`make -sI inc test`") == CheckResult(1, ())


def test_an_unknown_package_option_with_a_value_means_not_checked(ws2: Path) -> None:
    assert _check(ws2, "`bun run --env-file .env dev`") == NONE
    assert _check(ws2, "`bun run --env-file=.env --cwd ui dev`") == CheckResult(1, ())
    assert _check(ws2, "`bun run --cwd ui --foo dev`") == CheckResult(1, ())


def test_known_valueless_package_options_are_skipped(ws2: Path) -> None:
    assert _check(ws2, "`bun run --silent --cwd ui dev`") == CheckResult(1, ())
    assert _check(ws2, "`bun run --watch --cwd ui web:test`") == CheckResult(1, ())
    assert _check(ws2, "`npm run --if-present nope`").missing == ("npm run --if-present nope",)


def test_a_heredoc_body_is_not_read(ws2: Path) -> None:
    for opener in ("<<EOF", "<<-EOF", "<<'EOF'", '<<"EOF"'):
        text = f"```\nmake test {opener}\nmake nope\nEOF\nmake build\n```"
        assert _check(ws2, text) == CheckResult(2, ()), opener


def test_an_unterminated_heredoc_skips_to_the_end_of_the_block(ws2: Path) -> None:
    assert _check(ws2, "```\nmake test <<EOF\nmake nope\n```\n`make build`") == CheckResult(2, ())


def test_a_dash_heredoc_ends_on_a_tab_indented_word(ws2: Path) -> None:
    assert _check(ws2, "```\ncat <<-EOF\nmake nope\n\tEOF\nmake build\n```") == CheckResult(1, ())


def test_only_prompt_lines_are_commands_when_a_block_has_one(ws2: Path) -> None:
    assert _check(ws2, "```\n$ make test\nmake nope passed\n```") == CheckResult(1, ())
    assert _check(ws2, "```\n  $ make test\nmake nope\n```") == CheckResult(1, ())
    assert _check(ws2, "```\nmake nope\n```") == CheckResult(1, ("make nope",))


@pytest.mark.parametrize("form", ["make ...", "make --", "make .", "npm run ...", "npm run ."])
def test_a_word_without_a_letter_or_digit_is_not_checked(ws2: Path, form: str) -> None:
    assert _check(ws2, f"`{form}`") == NONE


def test_a_hash_inside_a_word_is_not_a_comment(ws2: Path) -> None:
    assert _check(ws2, "`make test#1`") == NONE
    assert _check(ws2, "`make test #1`") == CheckResult(1, ())
    assert _check(ws2, "`make test # x`") == CheckResult(1, ())
