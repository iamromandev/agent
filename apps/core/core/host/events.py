"""What the host publishes besides the engine's own events: the steps of a turn that a client draws, and its end."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

JobKind = Literal["turn", "compact", "workflow"]


@dataclass(frozen=True)
class ProviderState:
    """The provider was asked whether it is ready: `problem` is None when it is (the text is shown as an error)."""

    problem: str | None


@dataclass(frozen=True)
class UserEntry:
    """The user's message goes into the conversation: draw it (with the custom command's label, if any)."""

    text: str
    label: str | None = None
    client: str = ""  # the client that sent it


@dataclass(frozen=True)
class Delivering:
    """The entry and its notices are out; the agent gets the message once they are drawn."""


@dataclass(frozen=True)
class JobEventsDone:
    """The job's agent events are all published: close the reply and any tool block still open (as `ended`)."""


@dataclass(frozen=True)
class EntryEnd:
    """The turn's user entry is complete (`discard` False) or must go: the agent never got the message."""

    discard: bool


@dataclass(frozen=True)
class TurnOutcome:
    """How a turn, a compaction or a workflow ended. `continued`: the queue was sent as the next turn (busy stays on).
    Otherwise `restore_text` (the message that was not sent) and `queue_restore` (the queued messages' typed text)
    go back to the input box, in that order. `delivered` False: the agent never got the message, so the images the
    client handed over for turn `turn` are its again. `ack` (not 0): the host holds the claim until a client calls
    `outcome_applied(ack)` once it has applied this outcome (or a short timeout passes), so nothing new starts before
    the box got its text back. With several clients, `restores` says whose each text is: `restore_for(client)` is what
    goes back into that client's box (`restore` is all of it, for a lone client). `client` sent the turn and
    `interrupted_by` pressed Esc (empty: nobody)."""

    kind: JobKind
    status: str | None
    continued: bool = False
    restore_text: str | None = None
    queue_restore: str | None = None
    delivered: bool = False
    turn: int = 0
    init_check: tuple[str, str] | None = None  # the `/init` file check's notice (level, text), also published
    ack: int = 0
    client: str = ""
    interrupted_by: str = ""
    restores: tuple[tuple[str, str], ...] = ()  # (client, typed text): the draft's parts, then the queue's

    @property
    def restore(self) -> str | None:
        back = [text for text in (self.restore_text, self.queue_restore) if text]
        return "\n".join(back) if back else None

    def restore_for(self, client: str) -> str | None:
        """The text that goes back into `client`'s input box, in order; None when nothing of it is that client's."""
        back = [text for owner, text in self.restores if owner == client and text]
        return "\n".join(back) if back else None
