from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.engine.workflows.model import PlannedCall, Step, Workflow
from src.engine.workflows.retry import feedback, merge_calls, model_deps, redo_targets


def wf(*steps: Step) -> Workflow:
    return Workflow("w", "project", Path("/w/w.json"), "f" * 64, "d", "", "workflows/{name}/{run}", steps)


def chain() -> Workflow:
    return wf(
        Step("a", "model", prompt="about {input}", schema={}),
        Step("b", "model", prompt="more of {a.title}", schema={}),
        Step("pic", "each", tool="t", args={"path": "{folder}/p{index}", "prompt": "{item}"}, over="a.items"),
        Step("vid", "tool", tool="t", args={"path": "{folder}/v", "x": {"$map": "b.items", "to": {"i": "{item}"}}}),
        Step("const", "tool", tool="t", args={"path": "{folder}/c"}),
    )


def test_deps_come_from_args_and_over() -> None:
    w = chain()
    assert model_deps(w, "pic") == {"a"}
    assert model_deps(w, "vid") == {"b"}


def test_a_constant_step_has_no_deps() -> None:
    assert model_deps(chain(), "const") == set()


def test_a_model_step_depends_on_the_models_its_prompt_uses() -> None:
    assert model_deps(chain(), "b") == {"a"}
    assert model_deps(chain(), "a") == set()


def test_redo_of_a_includes_b_which_uses_it() -> None:
    assert redo_targets(chain(), {"a"}) == ["a", "b"]


def test_redo_of_b_is_only_b() -> None:
    assert redo_targets(chain(), {"b"}) == ["b"]


def test_feedback_without_error_says_the_user_asked() -> None:
    text = feedback("assemble_video", "video", None)
    assert text.startswith("The user asked for a new answer.")


def test_feedback_names_tool_key_and_error() -> None:
    text = feedback("assemble_video", "video", "scene 2 is 0.8 s")
    assert "The tool assemble_video refused the call video: scene 2 is 0.8 s" in text
    assert "fix the problem" in text


def test_feedback_for_a_dependent_step_says_an_earlier_answer_was_corrected_and_repeats_the_error() -> None:
    text = feedback("assemble_video", "video", "scene 2 is 0.8 s", dependent=True)
    assert "earlier answer" in text and "corrected" in text
    assert "scene 2 is 0.8 s" in text and "assemble_video" in text
    assert "fix the problem" not in text
    assert text != feedback("assemble_video", "video", "scene 2 is 0.8 s")


def test_feedback_for_a_dependent_step_after_a_user_redo_says_the_earlier_answer_changed() -> None:
    text = feedback("", "", None, dependent=True)
    assert "earlier answer" in text and "asked again" in text
    assert text != feedback("", "", None)


def pc(key: str, digest: str = "d") -> PlannedCall:
    return PlannedCall(key, key.split("/")[0], None, "t", {"path": f"{key}.png"}, digest)


def entry(key: str, digest: str = "d", state: str = "done") -> dict[str, Any]:
    return {"key": key, "tool": "t", "args": {}, "digest": digest, "state": state, "summary": "ok", "output_sha256": "x"}


def merge(
    calls: list[PlannedCall],
    saved: list[dict[str, Any]],
    unchanged: Callable[[PlannedCall, dict[str, Any]], bool] = lambda c, e: True,
    fed: Callable[[PlannedCall, set[Path]], bool] = lambda c, r: False,
):
    return merge_calls(calls, saved, unchanged, fed, lambda c: {Path(c.key)})


def test_unchanged_done_calls_are_reused() -> None:
    result = merge([pc("a"), pc("b")], [entry("a"), entry("b")])
    assert result.pending == [] and result.reused == 2
    assert [e["state"] for e in result.entries] == ["done", "done"]
    assert result.entries[0]["output_sha256"] == "x"


def test_a_changed_digest_is_pending_and_new_keys_are_added() -> None:
    result = merge([pc("a", "new"), pc("b"), pc("c")], [entry("a"), entry("b")])
    assert result.pending == [0, 2] and result.reused == 1
    assert [e["key"] for e in result.entries] == ["a", "b", "c"]
    assert result.entries[0]["state"] == "pending" and result.entries[0]["output_sha256"] is None
    assert result.entries[2]["state"] == "pending"


def test_dropped_keys_are_not_kept() -> None:
    result = merge([pc("a")], [entry("a"), entry("b")])
    assert [e["key"] for e in result.entries] == ["a"]


def test_a_failed_saved_call_is_pending() -> None:
    result = merge([pc("a")], [entry("a", state="failed")])
    assert result.pending == [0]
    assert "error" not in result.entries[0]


def test_a_changed_file_is_pending() -> None:
    result = merge([pc("a")], [entry("a")], unchanged=lambda c, e: False)
    assert result.pending == [0]


def test_a_call_fed_by_a_rewritten_path_is_pending() -> None:
    result = merge(
        [pc("a", "new"), pc("b")], [entry("a"), entry("b")], fed=lambda c, rewritten: Path("a") in rewritten and c.key == "b"
    )
    assert result.pending == [0, 1]
