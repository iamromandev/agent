from __future__ import annotations

import pytest
from src.engine.workflows.template import Scope, TemplateError, placeholders, render

SCENES = {"scenes": [{"narration": "one", "image_prompt": "a"}, {"narration": "two", "image_prompt": "b"}]}


def scope(**values: object) -> Scope:
    return Scope({"folder": "workflows/reel/r1", "run": "r1", "name": "reel", "input": "cats", **values})


def test_plain_text_is_unchanged() -> None:
    assert render("no braces", scope()) == "no braces"


def test_placeholder_inside_text_is_substituted() -> None:
    assert render("{folder}/a.png", scope()) == "workflows/reel/r1/a.png"


def test_double_braces_are_literal_braces() -> None:
    assert render("{{x}}", scope()) == "{x}"
    assert render("a {{ b }} c", scope()) == "a { b } c"


def test_exact_placeholder_keeps_the_type() -> None:
    s = scope(plan={"n": 5, "ok": True, "list": [1, 2]})
    assert render("{plan.n}", s) == 5
    assert render("{plan.ok}", s) is True
    assert render("{plan.list}", s) == [1, 2]


def test_non_string_inside_text_is_json() -> None:
    assert render("n={plan.n}", scope(plan={"n": 5})) == "n=5"
    assert render("l={plan.l}", scope(plan={"l": ["a", 1]})) == 'l=["a", 1]'


def test_paths_use_keys_and_indexes() -> None:
    s = scope(story=SCENES)
    assert render("{story.scenes.1.narration}", s) == "two"


def test_item_and_index_in_scope() -> None:
    s = scope(item={"narration": "hi"}, index=3)
    assert render("scene-{index}-{item.narration}", s) == "scene-3-hi"
    assert render("{index}", s) == 3


def test_missing_path_names_the_placeholder() -> None:
    with pytest.raises(TemplateError, match=r"\{story\.nope\}"):
        render("{story.nope}", scope(story=SCENES))


def test_unknown_root_is_an_error() -> None:
    with pytest.raises(TemplateError, match=r"\{ghost\}"):
        render("{ghost}", scope())


def test_index_out_of_range_is_an_error() -> None:
    with pytest.raises(TemplateError, match=r"\{story\.scenes\.9\}"):
        render("{story.scenes.9}", scope(story=SCENES))


def test_stray_braces_are_errors() -> None:
    with pytest.raises(TemplateError):
        render("a { b", scope())
    with pytest.raises(TemplateError):
        render("a } b", scope())
    with pytest.raises(TemplateError):
        render("{a b}", scope())


def test_substituted_text_is_never_expanded_again() -> None:
    assert render("{input}", scope(input="{input}")) == "{input}"
    assert render("x {input} y", scope(input="{folder}")) == "x {folder} y"
    assert render("{story.t}", scope(story={"t": "{{x}}"})) == "{{x}}"


def test_big_value_is_substituted_whole() -> None:
    big = "y" * 10_000
    assert render("a{input}b", scope(input=big)) == "a" + big + "b"


def test_dicts_and_lists_are_rendered_recursively() -> None:
    out = render({"a": ["{run}", 1, {"b": "{name}"}], "n": None}, scope())
    assert out == {"a": ["r1", 1, {"b": "reel"}], "n": None}


def test_map_builds_the_list() -> None:
    s = scope(story=SCENES)
    template = {"$map": "story.scenes", "to": {"image": "{folder}/s{index}.png", "text": "{item.narration}"}}
    assert render(template, s) == [
        {"image": "workflows/reel/r1/s1.png", "text": "one"},
        {"image": "workflows/reel/r1/s2.png", "text": "two"},
    ]


def test_map_inside_args() -> None:
    s = scope(story=SCENES)
    out = render({"scenes": {"$map": "story.scenes", "to": "{item.image_prompt}"}, "fps": 30}, s)
    assert out == {"scenes": ["a", "b"], "fps": 30}


def test_map_over_a_non_list_is_an_error() -> None:
    with pytest.raises(TemplateError, match=r"story\.title"):
        render({"$map": "story.title", "to": "x"}, scope(story={"title": "t"}))


def test_map_needs_exactly_map_and_to() -> None:
    with pytest.raises(TemplateError):
        render({"$map": "story.scenes", "to": "x", "extra": 1}, scope(story=SCENES))
    with pytest.raises(TemplateError):
        render({"$map": "story.scenes"}, scope(story=SCENES))


def test_placeholders_lists_root_names() -> None:
    assert placeholders("{folder}/{story.scenes.0.x} {{lit}}") == {"folder", "story"}
    assert placeholders({"a": ["{input}", 3]}) == {"input"}


def test_placeholders_reports_item_and_index() -> None:
    assert placeholders("{item.x}-{index}") == {"item", "index"}


def test_placeholders_of_map_bind_item_and_index_inside_to() -> None:
    template = {"$map": "story.scenes", "to": "{item.x}{index}{folder}"}
    assert placeholders(template) == {"story", "folder"}


def test_placeholders_of_a_bad_template_is_an_error() -> None:
    with pytest.raises(TemplateError):
        placeholders("a { b")


def test_render_text_uses_json_for_a_structured_single_placeholder() -> None:
    from src.engine.workflows.template import render_text

    scope = Scope({"story": {"scenes": [{"a": "x"}], "title": "T"}})

    assert render_text("{story.scenes}", scope) == '[{"a": "x"}]'
    assert render_text("{story.title}", scope) == "T"
    assert render_text("pre {story}", scope) == 'pre {"scenes": [{"a": "x"}], "title": "T"}'
