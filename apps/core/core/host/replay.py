"""A snapshot's message records as transcript entries: what a remote client draws for a session it attaches to (the
same entries `SessionStore.load` gives the local app; image bytes never cross the wire, so their notice lines are
built from the metadata)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from core.engine.provider.images import size_text
from core.engine.session.records import RecordError, message_from_obj
from core.engine.session.store import TranscriptItem, transcript_item
from core.engine.text import escape_display


def snapshot_items(records: Iterable[dict[str, Any]]) -> list[TranscriptItem]:
    """The transcript of the records, in order; a record that cannot be read is left out."""
    items: list[TranscriptItem] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        images = record.get("images") or []
        try:
            message, meta = message_from_obj({key: value for key, value in record.items() if key != "images"})
        except (RecordError, ValueError, TypeError):
            continue
        own = transcript_item(message, meta)
        if own is not None:
            items.append(own)
        for image in images if isinstance(images, list) else ():
            if not isinstance(image, dict):
                continue
            size = image.get("size")
            shown = f"{str(image.get('format', '')).upper()}, {size_text(size if isinstance(size, int) else 0)}"
            items.append(TranscriptItem("notice", f"image {escape_display(str(image.get('name', '')))} ({shown})"))
    return items
