"""The scrolling conversation: user messages, streamed replies, tool calls and notices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from core.engine.session import TranscriptItem
from core.engine.text import escape
from core.engine.tools.base import Preview, ToolResult
from rich.cells import cell_len
from rich.color import Color as RichColor
from rich.color import ColorType
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.terminal_theme import TerminalTheme
from rich.text import Text
from textual.color import Color
from textual.containers import VerticalScroll
from textual.content import Content
from textual.geometry import Region
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Collapsible, Markdown, Static
from textual.widgets.markdown import MarkdownStream

from src.tui.styles import (
    ACCENT,
    ICONS,
    MARK_USER,
    NOTICE,
    Outcome,
    ansi_output,
    diff_theme,
    notice_text,
    tool_title,
)

REPLAY_BATCH = 500  # entries mounted at once when a saved conversation is redrawn


def _readable(text: str) -> str:
    """`escape(text)` per line and per tab, so line breaks and tabs stay as they are."""
    return "\n".join("\t".join(escape(part) for part in line.split("\t")) for line in text.split("\n"))


def _themed(style: Style | None, theme: TerminalTheme) -> Style | None:
    """`style` with a standard ANSI colour written as the theme's own colour, the way the text beside it is drawn."""
    if style is None or style.color is None or style.color.type != ColorType.STANDARD:
        return style
    number = style.color.number
    return style + Style(color=RichColor.from_triplet(theme.ansi_colors[number])) if number is not None else style


class GutterMark:
    """Mixin: writes `gutter_mark` over the first cells of the widget's left padding, on row `gutter_row`.

    Listed before the widget class in the bases. The padding keeps wrapped rows indented, so the mark
    needs no extra widget. Skipped when the widget is narrower than the mark or the crop does not start at 0."""

    gutter_mark: Text | None = None
    gutter_row: int = 0

    def set_gutter(self, mark: Text | None, row: int | None = None) -> None:
        self.gutter_mark = mark
        if row is not None:
            self.gutter_row = row
        cast("Widget", self).refresh()

    def render_lines(self, crop: Region) -> list[Strip]:
        strips = cast("Any", super()).render_lines(crop)
        mark = self.gutter_mark
        if mark is None or crop.x != 0 or not crop.y <= self.gutter_row < crop.bottom:
            return strips
        i = self.gutter_row - crop.y
        if i >= len(strips):
            return strips
        strip: Strip = strips[i]
        padding = cast("Widget", self).styles.padding.left
        if strip.cell_length < min(cell_len(mark.plain), padding):
            return strips
        base = next(iter(strip), None)  # keep the padding's background under the mark
        theme = cast("Widget", self).app.ansi_theme
        shown = [Segment(seg.text, _themed(seg.style, theme)) for seg in Content.from_rich_text(mark).render_segments()]
        if base is not None and base.style is not None:
            shown = [Segment(seg.text, base.style + seg.style if seg.style else base.style) for seg in shown]
        width = min(sum(seg.cell_length for seg in shown), padding)  # never over the text
        drawn = Strip(shown).crop(0, width)
        strips[i] = Strip([*drawn, *strip.crop(width, strip.cell_length)], strip.cell_length)
        return strips


class UserEntry(GutterMark, Static):
    DEFAULT_CSS = """
    UserEntry { margin: 1 0 0 0; padding: 0 0 0 2; background: transparent; }
    UserEntry:first-child { margin-top: 0; }
    """

    def __init__(self, text: str, label: str | None = None) -> None:
        body = Text()
        if label:
            body.append(f"{label}\n", "dim")
            body.append(_readable(text))  # a custom command's prompt comes from a file: no escape sequences
        else:
            body.append(text)
        super().__init__(body)
        self.gutter_mark = Text(MARK_USER)
        self.gutter_row = 1 if label else 0
        self.text = text
        self.label = label
        self.body = body
        self.turn: int | None = None  # set by Transcript.add_user


class AssistantEntry(GutterMark, Markdown):
    DEFAULT_CSS = """
    AssistantEntry { padding: 0 0 0 2; margin: 0; }
    AssistantEntry > .newest { margin-bottom: 0; }
    """

    def __init__(self, text: str = "") -> None:
        # model text must not open the browser by itself
        super().__init__(text, open_links=False)
        self.gutter_mark = Text("● ", ACCENT)
        self.text = text
        self._stream: MarkdownStream | None = None

    def mount(self, *widgets: Widget, before: Any = None, after: Any = None) -> Any:
        """The newest block has no bottom margin, so a reply leaves no blank row after it.

        Every mount goes through here (`mount_all` calls it, and so does a streamed list or quote that
        replaces its block), so the last block always carries the class. A class set here, not a
        `:last-child` rule: that selector costs about 40% more when many replies mount."""
        if widgets and before is None and after is None:
            if self.children:
                self.children[-1].remove_class("newest")  # no longer last: the default gap again
            widgets[-1].add_class("newest")
        return super().mount(*widgets, before=before, after=after)

    @property
    def markdown(self) -> AssistantEntry:
        return self

    def on_mount(self) -> None:
        accent = self.app.current_theme.accent  # a plain hex colour, not `auto`
        if accent is not None:
            self.set_gutter(Text("● ", Style(color=Color.parse(accent).rich_color)))

    async def write(self, delta: str) -> None:
        if self._stream is None:
            self._stream = Markdown.get_stream(self)
        self.text += delta
        await self._stream.write(delta)

    async def finish(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None


class NoticeEntry(GutterMark, Static):
    DEFAULT_CSS = "NoticeEntry { margin: 0 0 0 2; padding: 0 0 0 2; }"

    def __init__(self, level: Literal["info", "warn", "error"], text: str, icon: str | None = None) -> None:
        rendered = notice_text(level, text, icon)
        style = NOTICE[level]
        super().__init__(Text(text, style))
        self.gutter_mark = Text(f"{icon or ICONS[level]} ", style)
        self.text = rendered.plain


def _title(text: Text) -> str:
    """A title Textual shows as written. Collapsible parses a `str` title as markup, and
    `markup.escape` is not safe for every input (`[$var]`, backslashes); a `Content` is never parsed.
    Collapsible accepts it at runtime although its hint says `str`."""
    return cast("str", Content.from_rich_text(text))


def _child_line(label: str, summary: str, outcome: Outcome) -> Text:
    return tool_title(label, summary, outcome)


class ToolEntry(Collapsible):
    """One tool call: `label · summary`, with the output (and the applied diff) inside."""

    DEFAULT_CSS = """
    ToolEntry { background: transparent; border-top: none; padding: 0; }
    ToolEntry > CollapsibleTitle { padding: 0 0 0 1; }
    ToolEntry > Contents { margin: 0 0 0 4; padding: 0 0 0 1; border-left: solid $foreground 30%; }
    """

    def __init__(self, label: str, summary: str | None = None, task: bool = False) -> None:
        """`summary` given: a finished call that is only shown (a replayed one), with nothing inside.

        `task`: a running `task` call, open while it runs and folded when it ends."""
        running_task = task and summary is None
        super().__init__(
            title=_title(tool_title(label, summary or "", "replayed" if summary is not None else None)),
            collapsed=not running_task,
            collapsed_symbol="",
            expanded_symbol="",
        )
        self.label = label
        self.finished = summary is not None
        self.is_task = running_task
        self.detail = ""
        self._children: dict[str, tuple[Static, str]] = {}  # a task's child calls: line and label, by child call id

    async def add_child(self, call_id: str, label: str) -> None:
        """A tool call of a task's child, as a line inside the block."""
        if (earlier := self._children.get(call_id)) is not None and earlier[1]:
            earlier[0].update(_child_line(earlier[1], "ended", "ended"))  # the id was reused before it finished
        line = Static(_child_line(label, "", None))
        self._children[call_id] = (line, label)
        await self.query_one(Collapsible.Contents).mount(line)

    async def finish_child(self, call_id: str, result: ToolResult) -> None:
        if (child := self._children.get(call_id)) is not None:
            line, label = child
            line.update(_child_line(label, result.summary, "error" if result.is_error else "ok"))
            self._children[call_id] = (line, "")  # finished: the interrupt pass leaves it alone

    async def finish(self, result: ToolResult | None, preview: Preview | None, applied: bool, note: str = "") -> None:
        self.finished = True
        summary = note or (result.summary if result else "")
        outcome: Outcome = "ended" if note or result is None else ("error" if result.is_error else "ok")
        self.title = _title(tool_title(self.label, summary, outcome))
        for line, child_label in self._children.values():
            if child_label:  # a child call that never finished (the task was interrupted)
                line.update(_child_line(child_label, note or "ended", "ended"))
        parts: list[Text | Syntax] = []
        details: list[str] = []
        if result is not None and applied and preview is not None and preview.kind in ("diff", "content"):
            theme = diff_theme(self.app.current_theme.dark)
            parts.append(Syntax(preview.body, "diff" if preview.kind == "diff" else "text", theme=theme, word_wrap=True))
            details.append(preview.body)
            self.collapsed = False
        if result is not None and result.output:
            parts.append(ansi_output(result.display) if result.display is not None else Text(result.output))
            details.append(result.output)
        self.detail = "\n".join(details)
        contents = self.query_one(Collapsible.Contents)
        for part in parts:
            await contents.mount(Static(part))
        if self.is_task:
            self.collapsed = True  # a finished task folds; its lines and report stay inside


class Transcript(VerticalScroll):
    DEFAULT_CSS = "Transcript { height: 1fr; }"

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._tools: dict[str, ToolEntry] = {}
        self._assistant: AssistantEntry | None = None
        self._turns = 0
        self.latest_turn = 0  # token of the newest user entry, set before it is mounted
        self._last: dict[int, Widget] = {}  # turn token -> the last widget it ended with
        self._ended: list[int] = []  # tokens of the turns that ended, oldest first

    async def add_user(self, text: str, label: str | None = None) -> int:
        """Show a user message (with a dim `label` line above it, e.g. the custom command); returns the turn's token."""
        self._turns += 1
        self.latest_turn = self._turns
        entry = UserEntry(text, label)
        entry.turn = self._turns
        await self._add(entry)
        return self._turns

    def end_turn(self, token: int) -> None:
        """The turn's widgets are all drawn: what is added from here on does not belong to it."""
        if last := next(iter(reversed(self.children)), None):
            self._last[token] = last
            self._ended.append(token)

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

    async def add_tool(self, call_id: str, label: str, task: bool = False) -> ToolEntry:
        await self.end_assistant()
        entry = ToolEntry(label, task=task)
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
        entries: list[Widget] = []
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

    async def remove_turn(self, token: int) -> bool:
        """Remove one turn: its user entry through the last widget it ended with, not what came after.

        A turn that never ended (its message never reached the agent) loses only its user entry.
        False when the turn is not on screen.
        """
        await self.end_assistant()
        children = list(self.children)
        start = next((i for i, c in enumerate(children) if isinstance(c, UserEntry) and c.turn == token), None)
        if start is None:
            return False
        last = self._last.pop(token, None)
        stop = children.index(last) + 1 if last in children else start + 1
        removed = children[start : max(stop, start + 1)]
        self._tools = {call_id: tool for call_id, tool in self._tools.items() if tool not in removed}
        if token in self._ended:
            self._ended.remove(token)
        await self.remove_children(removed)
        return True

    async def remove_last_turn(self) -> bool:
        """Remove the newest turn that ended. False when there is none."""
        return bool(self._ended) and await self.remove_turn(self._ended[-1])

    async def clear_all(self) -> None:
        await self.end_assistant()
        self._tools.clear()
        self._last.clear()
        self._ended.clear()
        await self.remove_children()

    async def _add(self, widget: Widget) -> None:
        await self.mount(widget)
        self.scroll_end(animate=False)
