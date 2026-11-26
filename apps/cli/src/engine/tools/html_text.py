"""HTML to readable text with the standard library only (no new dependency)."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_SKIP = {"script", "style", "noscript", "template", "svg"}
_BLOCK = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section",
    "article", "header", "footer", "nav", "pre", "blockquote", "dt", "dd", "hr",
}  # fmt: skip
_SPACE = re.compile(r"\s+")
_META = re.compile(rb"<meta[^>]+?charset\s*=\s*[\"']?\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE)


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[tuple[str, bool]] = []  # (text, preformatted)
        self._cur: list[str] = []
        self._cur_pre = False
        self._skip = 0
        self._pre = 0
        self._in_title = False
        self._title: list[str] = []

    def _newline(self, force: bool = False) -> None:
        if not force and not "".join(self._cur).strip():
            self._cur = []
            self._cur_pre = self._pre > 0
            return
        self.lines.append(("".join(self._cur), self._cur_pre))
        self._cur = []
        self._cur_pre = self._pre > 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK:
            self._newline()
            if tag == "pre":
                self._pre += 1
                self._cur_pre = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK:
            self._newline()
        # a self-closing script/style/svg has no body, so nothing to skip

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK:
            if tag == "pre":
                self._pre = max(0, self._pre - 1)
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self._title.append(data)
        elif self._pre:
            parts = data.split("\n")
            for i, part in enumerate(parts):
                if i:
                    self._newline(force=True)
                    self._cur_pre = True
                self._cur.append(part)
        else:
            self._cur.append(_SPACE.sub(" ", data))

    def text(self) -> str:
        self._newline()
        out: list[str] = []
        title = _SPACE.sub(" ", "".join(self._title)).strip()
        if title:
            out.append(title)
        for text, pre in self.lines:
            line = text.rstrip() if pre else text.strip()
            if not line and (not out or not out[-1]):
                continue  # no leading blank lines, and runs of blank lines collapse to one
            out.append(line)
        while out and not out[-1]:
            out.pop()
        return "\n".join(out)


def html_to_text(html: str) -> str:
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def meta_charset(raw: bytes) -> str | None:
    """The charset a `<meta charset=...>` or `http-equiv` tag names in the first 2 KB, if any."""
    match = _META.search(raw[:2048])
    return match.group(1).decode("ascii", "replace") if match else None
