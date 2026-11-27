"""The image eval: options, planning, skips, and the whole run through real worker processes with fake commands."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import evals.images as images_module
import pytest
from core.testing.image_fakes import dry_run, fake_compare, fake_composite, fake_hf, fake_mflux
from evals.images import CASES, Tools, main, options_from_env, plan_cases

CLI = Path(__file__).resolve().parents[2]
GEN, EDIT = "mflux-generate-z-image-turbo", "mflux-generate-flux2-edit"


def _tools(**over: Any) -> Tools:
    base: dict[str, Any] = {
        "generate": "/bin/gen", "edit": "/bin/edit", "hf": "/bin/hf", "helper": "/h.py", "compare": "/c.py",
        "generate_model": "x/gen", "edit_model": "x/edit", "timeout": 60.0,
    }  # fmt: skip
    return Tools(**{**base, **over})


def _cache(hub: Path, *models: str) -> None:
    for model in models:
        snap = hub / f"models--{model.replace('/', '--')}" / "snapshots" / "abc"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "weights").write_bytes(b"x")


# options


def test_options_default_runs_every_case() -> None:
    opts = options_from_env({})
    assert not isinstance(opts, str)
    assert opts.cases == CASES and not opts.download and not opts.keep


def test_options_filter_keeps_canonical_order() -> None:
    opts = options_from_env({"IMAGE_EVAL_CASES": "masked, generate", "IMAGE_EVAL_DOWNLOAD": "1", "IMAGE_EVAL_KEEP": "1"})
    assert not isinstance(opts, str)
    assert opts.cases == ("generate", "masked") and opts.download and opts.keep


def test_options_unknown_case_lists_valid_names() -> None:
    message = options_from_env({"IMAGE_EVAL_CASES": "generate,nope"})
    assert isinstance(message, str) and "nope" in message
    assert all(name in message for name in CASES)


# plan_cases


def test_plan_runs_everything_when_cached() -> None:
    opts = options_from_env({})
    assert not isinstance(opts, str)
    assert plan_cases(opts, _tools(), lambda m: True) == [(name, None) for name in CASES]


def test_plan_skips_uncached_edit_model_and_pull() -> None:
    opts = options_from_env({})
    assert not isinstance(opts, str)
    plan = dict(plan_cases(opts, _tools(), lambda m: m == "x/gen"))
    assert plan == {
        "pull": "model not downloaded",
        "generate": None,
        "edit": "model not downloaded",
        "masked": "model not downloaded",
    }


def test_plan_download_runs_everything() -> None:
    opts = options_from_env({"IMAGE_EVAL_DOWNLOAD": "1"})
    assert not isinstance(opts, str)
    assert all(reason is None for _, reason in plan_cases(opts, _tools(), lambda m: False))


def test_plan_missing_command_skips_its_cases() -> None:
    opts = options_from_env({})
    assert not isinstance(opts, str)
    plan = dict(plan_cases(opts, _tools(edit=None), lambda m: True))
    assert plan["edit"] == plan["masked"] == "mflux command not found"
    assert plan["generate"] is None and plan["pull"] is None


# main with a skip


def test_main_skip_prints_reason_and_exits_zero(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main([], discover=lambda s: "no mflux", settings_factory=lambda: None, results_dir=tmp_path)
    assert code == 0
    assert "image-eval skipped: no mflux" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_main_unknown_case_exits_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = main([], env={"IMAGE_EVAL_CASES": "nope"}, discover=lambda s: _tools(), settings_factory=lambda: None, results_dir=tmp_path)
    assert code == 2
    assert "pull" in capsys.readouterr().err


# full runs with fake commands


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    folder = tmp_path / "bin"
    folder.mkdir()
    hub = tmp_path / "hub"
    _cache(hub, "x/gen", "x/edit")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    for name in ("FAKE_MFLUX_MODE", "FAKE_COMPARE_EXIT", "IMAGE_EVAL_CASES", "IMAGE_EVAL_KEEP", "IMAGE_EVAL_DOWNLOAD"):
        monkeypatch.delenv(name, raising=False)
    tools = _tools(
        generate=str(fake_mflux(folder, GEN)),
        edit=str(fake_mflux(folder, EDIT)),
        hf=str(fake_hf(folder, dry_run([]))),
        helper=str(fake_composite(folder)),
        compare=str(fake_compare(folder)),
    )
    return {"tools": tools, "hub": hub, "results": tmp_path / "results", "folder": folder}


def _run(rig: dict[str, Any], **kwargs: Any) -> tuple[int, dict]:
    results: Path = rig["results"]
    code = main([], discover=lambda s: rig["tools"], settings_factory=lambda: None, results_dir=results, **kwargs)
    files = sorted(results.glob("images-*.json")) if results.exists() else []
    return code, (json.loads(files[-1].read_text()) if files else {})


def test_all_cases_pass_in_order_and_json_has_the_keys(rig: dict[str, Any]) -> None:
    code, data = _run(rig)
    assert code == 0
    assert [c["name"] for c in data["cases"]] == list(CASES)
    assert [c["result"] for c in data["cases"]] == ["pass"] * 4
    assert set(data) == {"started", "machine", "mflux", "commit", "models", "rss_note", "cases"}
    assert "excludes" in data["rss_note"]
    assert data["models"] == {"generate": "x/gen", "edit": "x/edit"}
    for case in data["cases"]:
        assert set(case) == {"name", "result", "reason", "seconds", "peak_rss_bytes"}
        assert case["peak_rss_bytes"] is None or (isinstance(case["peak_rss_bytes"], int) and case["peak_rss_bytes"] > 0)


def test_run_prints_a_table_and_the_path(rig: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    _run(rig)
    out = capsys.readouterr().out
    assert "case" in out and "generate" in out and "pass" in out and "images-" in out and ".json" in out
    assert "peak RSS (excl. GPU)" in out


def test_wrong_size_output_fails_generate_and_later_cases_use_the_fallback(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "wrongsize")
    code, data = _run(rig)
    by = {c["name"]: c for c in data["cases"]}
    assert code == 1
    assert by["generate"]["result"] == "fail" and "128" in by["generate"]["reason"]
    assert by["edit"]["result"] == "fail" and by["masked"]["result"] == "fail"  # they ran (wrong size again)
    assert "worker" not in (by["edit"]["reason"] or "")


def test_edit_runs_on_the_fallback_when_generate_is_skipped(rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_EVAL_CASES", "edit,masked")
    code, data = _run(rig)
    assert code == 0
    assert [c["result"] for c in data["cases"]] == ["pass", "pass"]


def _reasons(rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, cases: str, only: str, code: int = 1) -> dict[str, Any]:
    monkeypatch.setenv("IMAGE_EVAL_CASES", cases)
    monkeypatch.setenv("FAKE_COMPARE_EXIT", str(code))
    monkeypatch.setenv("FAKE_COMPARE_ONLY", only)
    run_code, data = _run(rig)
    assert run_code == 1
    return {c["name"]: c for c in data["cases"]}


@pytest.mark.parametrize(
    ("cases", "only", "name", "reason"),
    [
        ("masked", "same", "masked", "pixels outside the region changed"),
        ("masked", "changed", "masked", "the edit changed nothing inside the region"),
        ("edit", "changed", "edit", "the edit changed nothing"),
        ("generate", "varied", "generate", "the generated image is uniform"),
    ],
)
def test_a_failed_compare_check_has_its_own_reason(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, cases: str, only: str, name: str, reason: str
) -> None:
    by = _reasons(rig, monkeypatch, cases, only)
    assert by[name]["result"] == "fail" and by[name]["reason"] == reason


@pytest.mark.parametrize("code", [2, 3])
def test_a_broken_compare_is_not_reported_as_a_failed_check(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    by = _reasons(rig, monkeypatch, "masked", "same", code)
    assert by["masked"]["reason"].startswith("pixel compare failed")
    assert "compare broke" in by["masked"]["reason"] and "outside the region" not in by["masked"]["reason"]


def test_the_compare_gets_the_modes_and_the_region_columns(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dump = tmp_path / "compare.jsonl"
    monkeypatch.setenv("FAKE_COMPARE_DUMP", str(dump))
    monkeypatch.setenv("IMAGE_EVAL_CASES", "generate,edit,masked")
    code, _ = _run(rig)
    assert code == 0
    calls = [json.loads(line) for line in dump.read_text().splitlines()]
    assert [c[0] for c in calls] == ["varied", "changed", "same", "changed"]
    assert calls[0][1].endswith("gen.png")
    assert calls[1][1].endswith("gen.png") and calls[1][2].endswith("edit.png") and calls[1][3] == "256"
    assert calls[2][1].endswith("gen.png") and calls[2][2].endswith("masked.png") and calls[2][3] == "136"
    assert calls[3][3] == "120" and calls[3][4] == str(int(120 * 256 * 0.05))
    assert int(calls[1][4]) > 0


def test_the_real_compare_flags_a_uniform_generated_image(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    rig["tools"] = _tools(**{**rig["tools"].__dict__, "compare": str(CLI / "evals" / "pixel_compare.py")})
    monkeypatch.setenv("IMAGE_EVAL_CASES", "generate")
    _, data = _run(rig)
    (case,) = data["cases"]
    assert case["result"] == "fail" and case["reason"] == "the generated image is uniform"


def test_a_raising_case_is_a_failure_and_later_cases_still_run(rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    rig["tools"] = _tools(**{**rig["tools"].__dict__, "generate": "/does/not/exist"})
    code, data = _run(rig)
    by = {c["name"]: c for c in data["cases"]}
    assert code == 1
    assert by["generate"]["result"] == "fail"
    assert by["edit"]["result"] == "pass"


def test_missing_compare_script_raises_inside_the_case_and_is_reported(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rig["tools"] = _tools(**{**rig["tools"].__dict__, "compare": "/does/not/exist.py"})
    monkeypatch.setenv("IMAGE_EVAL_CASES", "masked")
    code, data = _run(rig)
    (case,) = data["cases"]
    assert code == 1 and case["result"] == "fail" and case["reason"]


def test_reason_is_cut_to_200_characters(rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MFLUX_MODE", "fail")
    monkeypatch.setenv("FAKE_MFLUX_LINES", "50")
    monkeypatch.setenv("IMAGE_EVAL_CASES", "generate")
    _, data = _run(rig)
    (case,) = data["cases"]
    assert case["result"] == "fail" and 0 < len(case["reason"]) <= 200


def test_uncached_edit_model_skips_edit_masked_and_pull(rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    for entry in (rig["hub"] / "models--x--edit").rglob("*"):
        if entry.is_file():
            entry.unlink()
    code, data = _run(rig)
    by = {c["name"]: c for c in data["cases"]}
    assert code == 0
    assert by["generate"]["result"] == "pass"
    for name in ("pull", "edit", "masked"):
        assert by[name]["result"] == "skipped" and by[name]["reason"] == "model not downloaded"


def test_case_filter_runs_only_that_case(rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_EVAL_CASES", "generate")
    _, data = _run(rig)
    assert [c["name"] for c in data["cases"]] == ["generate"]


def test_workspace_is_removed_unless_kept(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("IMAGE_EVAL_CASES", "generate")
    _run(rig)
    assert "workspace kept" not in capsys.readouterr().out
    monkeypatch.setenv("IMAGE_EVAL_KEEP", "1")
    _run(rig)
    line = next(row for row in capsys.readouterr().out.splitlines() if "workspace kept" in row)
    kept = Path(line.split("workspace kept:", 1)[1].strip())
    try:
        assert (kept / "gen.png").is_file()
    finally:
        import shutil

        shutil.rmtree(kept, ignore_errors=True)


DRIVER = """
import json, sys
from evals.images import Tools, main
tools = Tools(**json.load(open(sys.argv[1])))
code = main([], discover=lambda s: tools, settings_factory=lambda: None, results_dir=__import__("pathlib").Path(sys.argv[2]))
print("PIL" in sys.modules or "numpy" in sys.modules, file=sys.stderr)
sys.exit(code)
"""


def _driver(rig: dict[str, Any], tmp_path: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    state = tmp_path / "tools.json"
    state.write_text(json.dumps(rig["tools"].__dict__))
    return subprocess.Popen(
        [sys.executable, "-c", DRIVER, str(state), str(rig["results"])],
        cwd=CLI, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip


def test_no_eval_process_imports_pillow_or_numpy(rig: dict[str, Any], tmp_path: Path) -> None:
    proc = _driver(rig, tmp_path, {})
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, out + err
    assert err.strip().splitlines()[-1] == "False"
    assert [p.name for p in Path(str(rig["results"])).iterdir()] == [p.name for p in Path(str(rig["results"])).glob("images-*.json")]


def test_sigint_kills_the_worker_and_its_mflux_group_and_exits_130(rig: dict[str, Any], tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    proc = _driver(rig, tmp_path, {"FAKE_MFLUX_MODE": "sleep", "FAKE_MFLUX_PIDS": str(pids), "IMAGE_EVAL_CASES": "generate"})
    try:
        for _ in range(300):
            if pids.exists() and pids.read_text().strip().count(" ") == 1:
                break
            time.sleep(0.1)
        else:
            pytest.fail("the fake mflux never started")
        mflux_pid, child_pid = _pids(pids)
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 130
    _assert_dead(mflux_pid, child_pid)
    (written,) = Path(str(rig["results"])).glob("images-*.json")
    (case,) = json.loads(written.read_text())["cases"]
    assert case["name"] == "generate" and case["result"] == "fail" and case["reason"] == "interrupted"


def _assert_dead(*pids: int) -> None:
    for pid in pids:
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
            pytest.fail(f"process {pid} survived")


def _pids(path: Path) -> tuple[int, int]:
    mflux_pid, child_pid = (int(v) for v in path.read_text().split())
    return mflux_pid, child_pid


def test_an_early_ctrl_c_exits_130_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def discover(settings: Any) -> Tools:
        raise KeyboardInterrupt

    code = main([], discover=discover, settings_factory=lambda: None, results_dir=tmp_path)
    assert code == 130
    assert "interrupted" in capsys.readouterr().err


def test_the_case_timeout_stops_a_stuck_case_and_sweeps_what_ignores_signals(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pids = tmp_path / "pids"
    monkeypatch.setattr(images_module, "INTERRUPT_WAIT", 0.0)  # the worker is SIGKILLed at once, its mflux is orphaned
    monkeypatch.setenv("FAKE_MFLUX_MODE", "stubborn")
    monkeypatch.setenv("FAKE_MFLUX_PIDS", str(pids))
    monkeypatch.setenv("IMAGE_EVAL_CASES", "generate")
    monkeypatch.setenv("IMAGE_EVAL_CASE_TIMEOUT", "4")
    started = time.monotonic()
    code, data = _run(rig)
    assert time.monotonic() - started < 60
    (case,) = data["cases"]
    assert code == 1 and case["result"] == "fail" and case["reason"] == "timed out after 4 s"
    _assert_dead(*_pids(pids))


def test_a_bad_case_timeout_is_a_usage_error() -> None:
    for value in ("abc", "0", "-5"):
        message = options_from_env({"IMAGE_EVAL_CASE_TIMEOUT": value})
        assert isinstance(message, str) and "IMAGE_EVAL_CASE_TIMEOUT" in message


def test_sweep_kills_the_recorded_groups_and_ignores_dead_ones(tmp_path: Path) -> None:
    survivor = subprocess.Popen(["sleep", "60"], start_new_session=True)
    file = tmp_path / "groups"
    file.write_text(f"{survivor.pid}\n999999\nnot a number\n")
    try:
        images_module._sweep(file)
        assert survivor.wait(timeout=10) == -signal.SIGKILL
    finally:
        if survivor.poll() is None:
            survivor.kill()
    images_module._sweep(tmp_path / "missing")


def test_the_worker_gets_a_private_tmpdir_inside_the_eval_area(
    rig: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dump = tmp_path / "dump.json"
    monkeypatch.setenv("FAKE_MFLUX_DUMP", str(dump))
    monkeypatch.setenv("IMAGE_EVAL_CASES", "generate")
    _run(rig)
    env = json.loads(dump.read_text())["env"]
    assert "agent-image-eval-" in env["TMPDIR"] and Path(json.loads(dump.read_text())["cwd"]).parent == Path(env["TMPDIR"]).resolve()
