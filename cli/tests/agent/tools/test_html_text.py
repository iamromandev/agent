from __future__ import annotations

import time

import pytest
from src.agent.tools.html_text import html_to_text, meta_charset

SAMPLE = (
    "<html><head><title>T</title><style>x{}</style></head>"
    "<body><h1>A</h1><script>evil()</script><p>B &amp; C</p></body></html>"
)


def test_spec_sample_converts_to_title_heading_paragraph() -> None:
    assert html_to_text(SAMPLE) == "T\nA\nB & C"


def test_script_style_and_noscript_text_is_dropped() -> None:
    text = html_to_text("<p>a</p><noscript>no</noscript><script>s()</script><style>b{}</style><p>z</p>")
    assert text == "a\nz"


def test_pre_keeps_its_whitespace() -> None:
    assert html_to_text("<pre>a  b\n  c</pre>") == "a  b\n  c"


def test_whitespace_inside_a_line_is_collapsed() -> None:
    assert html_to_text("<p>a   \n  b\t c</p>") == "a b c"


def test_entities_are_decoded() -> None:
    assert html_to_text("<p>&lt;x&gt; &#65; &copy;</p>") == "<x> A ©"


def test_nested_blocks_and_breaks_add_no_blank_lines() -> None:
    assert html_to_text("<div><div><p>a</p></div></div><br><br><br><p>b</p>") == "a\nb"


def test_blank_runs_inside_pre_collapse_to_one() -> None:
    assert html_to_text("<pre>a\n\n\n\nb</pre>") == "a\n\nb"


def test_list_items_are_on_separate_lines() -> None:
    assert html_to_text("<ul><li>one</li><li>two</li></ul>") == "one\ntwo"


def test_empty_html_gives_empty_text() -> None:
    assert html_to_text("<script>x</script>") == ""


def test_meta_charset_short_form() -> None:
    assert meta_charset(b'<html><meta charset="iso-8859-1">') == "iso-8859-1"


def test_meta_charset_http_equiv_form() -> None:
    raw = b'<meta http-equiv="Content-Type" content="text/html; charset=windows-1251">'
    assert meta_charset(raw) == "windows-1251"


def test_meta_charset_absent() -> None:
    assert meta_charset(b"<html><body>hi</body></html>") is None


def test_two_mb_converts_quickly() -> None:
    page = "<html><body>" + "<p>some <b>bold</b> text &amp; more</p><div>x</div>" * 40_000
    assert len(page) > 1_800_000
    start = time.perf_counter()
    html_to_text(page)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize("page", ["<" * 2_000_000, "<b>x</b>" * 300_000], ids=["lt-2M", "bold-300k"])
def test_pathological_input_converts_in_bounded_time(page: str) -> None:
    # "<" * 2M takes about 0.8s here; the tool runs it off the event loop, so this bound only catches blow-ups.
    start = time.perf_counter()
    html_to_text(page)
    assert time.perf_counter() - start < 5.0
