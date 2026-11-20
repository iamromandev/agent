from __future__ import annotations

from src.engine.tools.html_text import html_to_text

SAMPLE = (
    "<html><head><title>T</title><style>x{}</style></head>"
    "<body><h1>A</h1><script>evil()</script><p>B &amp; C</p></body></html>"
)


def test_spec_sample_converts_to_title_heading_paragraph() -> None:
    assert html_to_text(SAMPLE) == "T\nA\nB & C"


def test_blank_runs_inside_pre_collapse_to_one() -> None:
    assert html_to_text("<pre>a\n\n\n\nb</pre>") == "a\n\nb"
