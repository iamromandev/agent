"""The scrolling conversation: user messages, streamed replies, tool calls and notices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from rich.syntax import Syntax
from rich.text import Text
from textual.containers import VerticalScroll
from textual.content import Content
from textual.widgets import Collapsible, Markdown, Static
from textual.widgets.markdown import MarkdownStream

from src.agent.session import TranscriptItem
from src.agent.tools.base import Preview, ToolResult

REPLAY_BATCH = 500  # entries mounted at once when a saved conversation is redrawn
ICONS = {"info": "ℹ", "warn": "⚠", "error": "✗"}  # noqa: RUF001 - the info sign is the intended glyph


class UserEntry(Static):
    DEFAULT_CSS = "UserEntry { margin: 1 0 0 0; padding: 0 1; background: $boost; }"

    def __init__(self, text: str) -> None:
        super().__init__(Text(f"> {text}"))
        self.text = text


class AssistantEntry(Markdown):
    DEFAULT_CSS = "AssistantEntry { margin: 0 1; }"

    def __init__(self, text: str = "") -> None:
        super().__init__(text, open_links=False)  # model text must not open the browser by itself
        self.text = text
        self._stream: MarkdownStream | None = None

    async def write(self, delta: str) -> None:
        if self._stream is None:
            self._stream = Markdown.get_stream(self)
        self.text += delta
        await self._stream.write(delta)

    async def finish(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None


class NoticeEntry(Static):
    DEFAULT_CSS = "NoticeEntry { padding: 0 1; color: $text-muted; }"

    def __init__(self, level: Literal["info", "warn", "error"], text: str, icon: str | None = None) -> None:
        shown = f"{icon or ICONS[level]} {text}"
        super().__init__(Text(shown))
        self.text = shown


def _literal(text: str) -> str:
    """`text` as a title Textual shows as written. Collapsible parses a `str` title as markup, and
    `markup.escape` is not safe for every input (`[$var]`, backslashes); a `Content` is never parsed.
    Collapsible accepts it at runtime although its hint says `str`."""
    return cast("str", Content(text))


class ToolEntry(Collapsible):
    """One tool call: `label · summary`, with the output (and the applied diff) inside."""

    def __init__(self, label: str, summary: str | None = None) -> None:
        """`summary` given: a finished call that is only shown (a replayed one), with nothing inside."""
        shown = label if not summary else f"{label} · {summary}"
        super().__init__(title=_literal(shown), collapsed=True)
        self.label = label
        self.finished = summary is not None
        self.detail = ""

    async def finish(self, result: ToolResult | None, preview: Preview | None, applied: bool, note: str = "") -> None:
        self.finished = True
        summary = note or (result.summary if result else "")
        self.title = _literal(f"{self.label} · {summary}" if summary else self.label)
        parts: list[Text | Syntax] = []
        details: list[str] = []
        if result is not None and applied and preview is not None and preview.kind in ("diff", "content"):
            parts.append(Syntax(preview.body, "diff" if preview.kind == "diff" else "text", word_wrap=True))
            details.append(preview.body)
            self.collapsed = False
        if result is not None and result.output:
            parts.append(Text(result.output))
            details.append(result.output)
        self.detail = "\n".join(details)
        contents = self.query_one(Collapsible.Contents)
        for part in parts:
            await contents.mount(Static(part))


class Transcript(VerticalScroll):
    DEFAULT_CSS = "Transcript { height: 1fr; }"

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._tools: dict[str, ToolEntry] = {}
        self._assistant: AssistantEntry | None = None

    async def add_user(self, text: str) -> None:
        await self._add(UserEntry(text))

    async def write_assistant(self, delta: str) -> None:
        if self._assistant is None:
            self._assistant = AssistantEntry()
            await self._add(self._assistant)
        await self._assistant.write(delta)
        self.scroll_end(animate=False)

    async def end_assistant(self) -> None:
        if self._assistant is not None:
            entry, self._assistant = self._assistant, None
            await entry.finish()

    async def add_tool(self, call_id: str, label: str) -> ToolEntry:
        await self.end_assistant()
        entry = ToolEntry(label)
        self._tools[call_id] = entry
        await self._add(entry)
        return entry

    def tool(self, call_id: str) -> ToolEntry | None:
        return self._tools.get(call_id)

    async def close_open_tools(self, note: str) -> None:
        for entry in self._tools.values():
            if not entry.finished:
                await entry.finish(None, None, False, note)

    async def add_notice(
        self, level: Literal["info", "warn", "error"], text: str, icon: str | None = None
    ) -> None:
        await self.end_assistant()
        await self._add(NoticeEntry(level, text, icon))

    async def replay(self, items: Sequence[TranscriptItem]) -> None:
        """Redraw a saved conversation. Tool entries come back collapsed, with their summary and no diff.

        The entries are mounted in batches: one at a time, a session of a few thousand messages takes
        far too long to draw.
        """
        await self.end_assistant()
        entries: list[Static | Markdown | Collapsible] = []
        for item in items:
            match item.kind:
                case "user":
                    entries.append(UserEntry(item.text))
                case "assistant":
                    entries.append(AssistantEntry(item.text))
                case "tool":
                    entries.append(ToolEntry(item.text, item.summary))
                case "notice":
                    entries.append(NoticeEntry("info", item.text))
        for start in range(0, len(entries), REPLAY_BATCH):
            await self.mount_all(entries[start : start + REPLAY_BATCH])
        self.scroll_end(animate=False)

    async def clear_all(self) -> None:
        await self.end_assistant()
        self._tools.clear()
        await self.remove_children()

    async def _add(self, widget: Static | Markdown | Collapsible) -> None:
        await self.mount(widget)
        self.scroll_end(animate=False)
