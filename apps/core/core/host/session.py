"""One live session without a UI: the agent, the claims on it (a turn, a switch, an undo), the message queue, the event
bus and the asks, and the turn pipeline that used to live in the Textual app.

Everything a client draws comes back as host events on the bus (`core.host.events`) or as return values; the host
waits for its subscribers (`EventBus.drained`) where the app used to wait for its widgets, so an Esc lands at the same
points as before."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Generator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from core.engine import Agent, AgentBusy, AgentEvent, EventSink, Notice, TodosChanged, TurnEnded
from core.engine.approval import Approver
from core.engine.compaction import CHARS_PER_TOKEN
from core.engine.facts import FACTS_SHARE, init_check, render_facts, scan
from core.engine.guardrails.setup import Guardrails
from core.engine.hooks.setup import Hooks, hook_failed
from core.engine.instructions import fingerprint, levels
from core.engine.loop import NOT_SUMMARIZED, close_recorder
from core.engine.mentions import MentionNote, apply_vision, expand_mentions, note_text
from core.engine.permissions.setup import Rules, RuleSavedNotTrusted
from core.engine.prompt import build_init_prompt
from core.engine.provider.base import Image, SwitchableProvider
from core.engine.provider.images import describe
from core.engine.provider.models import (
    SHOW_TIMEOUT,
    CatalogError,
    ModelCatalog,
    ModelInfo,
    installed_name,
    resolve_window,
    shown_name,
)
from core.engine.provider.vision import VisionCheck
from core.engine.session import (
    FileRecorder,
    LoadedSession,
    NullRecorder,
    Recorder,
    SessionError,
    SessionInfo,
    SessionStore,
    SessionWriteError,
)
from core.engine.session.records import model_notice
from core.engine.text import escape, escape_display
from core.engine.turn_diff import DiffItem, FileDiff, TurnDiff
from core.host.asks import AskBroker, HostApprover
from core.host.bus import EventBus
from core.host.events import (
    Delivering,
    EntryEnd,
    JobEventsDone,
    JobKind,
    ModeChanged,
    ProviderState,
    QueueChanged,
    TurnOutcome,
    TurnUndone,
    UserEntry,
)
from core.host.queue import MessageQueue, Queued, merge

WORKFLOW_RUNNING = "a workflow is running (Esc or /workflow cancel stops it)"
# The daemon runs one job that runs tools at a time across its sessions (they share one checkpoint store).
OTHER_SESSION = "busy — another session is running a turn; try again when it ends"
OUTCOME_ACK_TIMEOUT = 2.0  # seconds the claim waits for a client to apply a turn's outcome (one that never does)
LOCAL_CLIENT = "local"  # the client id of the in-process TUI (a remote one sends its own in `X-Agent-Client`)
HOST = "host"  # the owner of a switch claim the host takes for itself (closing a replaced recorder)

Level = Literal["info", "warn", "error"]
SessionSource = Literal["startup", "resume", "new", "switch"]
# What a client hands over with a turn: the pasted images read now, and the warnings for those it could not read.
Attachments = Callable[[int], Awaitable[tuple[list[Image], list[str]]]]


async def uninterrupted(work: Awaitable[Any]) -> None:
    """Run `work` to the end even if this task is cancelled meanwhile; the cancel is re-raised after."""
    task = asyncio.ensure_future(work)
    cancel: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancel = cancel or exc  # a repeated cancel must not abandon the work
        except Exception:
            break  # the task holds it; read below
    task.result()
    if cancel is not None:
        raise cancel


@dataclass(frozen=True)
class SendResult:
    status: Literal["started", "queued", "ignored", "refused"]
    reason: str = ""  # why it was refused (a warning to show)


@dataclass(frozen=True)
class UndoResult:
    prompt: str | None  # the undone turn's prompt (for the input box); None when nothing was undone
    events: list[AgentEvent]


@dataclass(frozen=True)
class ModelSwitch:
    notices: list[tuple[Level, str]]
    model: str | None = None  # set when the provider was switched
    window: int | None = None


@dataclass(frozen=True)
class OpenedSession:
    loaded: LoadedSession | None  # None: not opened (the current session stays)
    notices: list[tuple[Level, str]]
    error: Exception | None = None  # why it was not opened (an `AgentBusy` when a turn or switch was in the way)


class SessionHost:
    def __init__(
        self,
        agent_factory: Callable[[Approver], Agent],
        provider: SwitchableProvider,
        workspace: Path,
        *,
        window_cap: int,
        store: SessionStore | None = None,
        recorder_factory: Callable[[], Recorder] | None = None,
        catalog: ModelCatalog | None = None,
        vision: VisionCheck | None = None,
        guard: Guardrails | None = None,
        permissions: Rules | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self.bus = EventBus()
        self.asks = AskBroker(self.bus)
        self.provider = provider
        self.workspace = workspace
        self.window_cap = window_cap  # AGENT_NUM_CTX: caps every window
        self.store = store
        self.recorder_factory: Callable[[], Recorder] = recorder_factory or NullRecorder
        self.catalog = catalog
        self.vision = vision if vision is not None else VisionCheck(catalog)  # one answer per model per run
        self.guard = guard if guard is not None else Guardrails.off(workspace)
        self.permissions = permissions if permissions is not None else Rules.empty()
        self.hooks = hooks if hooks is not None else Hooks.empty()
        self.agent = agent_factory(HostApprover(self.asks))
        self.queue = MessageQueue()  # messages sent while a turn runs, sent together when it ends `done`
        self.turn_active = False  # held by a turn, a compaction, an undo or a workflow
        self.turn_entered = False  # the job's task has run its first line
        # The switch claim (a session is being loaded or switched to: no turn may start meanwhile) and its client.
        self._switch_owner: str | None = None
        self.undoing = False  # /undo is running: an interrupt leaves it alone
        self.workflow_active = False  # a workflow (run by the client) holds the turn claim
        self.interrupted = False  # an interrupt landed while the job was active: the queue is not sent
        self.interrupted_by = ""  # the client whose interrupt that was
        self.job_status: str | None = None  # how the running turn or compaction ended, once known
        self.not_ready = False  # the last provider check failed: the next turn asks again first
        self.closing = False  # the clients are going away: nothing more is sent
        self._attachments: dict[str, Attachments] = {}  # where each client's pasted images come from
        # Whether a client is attached now (the daemon: it has an event stream open); None: every client is (local).
        self.connected: Callable[[str], bool] | None = None
        # Whether another session of the same daemon holds a job (a turn, a compaction, an undo); None: no other.
        self.elsewhere: Callable[[], bool] | None = None
        self.on_job_end: Callable[[], None] | None = None  # every turn, compaction or workflow end (the daemon reindexes)
        self._draft: str | None = None  # the message until the turn really starts sending it
        self._draft_parts: tuple[tuple[str, str], ...] = ()  # (client, typed text) of that message
        self._origin = LOCAL_CLIENT  # the client that sent the running turn
        self._ending_client = ""  # the client whose outcome holds the claim
        self._job: JobKind = "turn"
        self._turn_no = 0
        self._task: asyncio.Task[None] | None = None  # the running turn or compaction
        self._hook_task: asyncio.Task[None] | None = None  # the running SessionStart hooks
        self._outcomes = 0  # outcomes published with an ack
        self._ending = 0  # the ended job's outcome no client has applied yet (it holds the claim); 0: none
        self._ending_timer: asyncio.TimerHandle | None = None

    # state

    @property
    def session_id(self) -> str:
        recorder_id = getattr(self.agent.recorder, "session_id", "")
        return recorder_id if isinstance(recorder_id, str) else ""

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def switching(self) -> bool:
        """A session is being loaded or switched to (by some client): no turn may start meanwhile."""
        return self._switch_owner is not None

    def register_client(self, client: str, *, attachments: Attachments | None = None) -> None:
        """A client's own sources: where its turns take their pasted images from (a local client only)."""
        if attachments is None:
            self._attachments.pop(client, None)
        else:
            self._attachments[client] = attachments

    def client_gone(self, client: str) -> None:
        """`client` disconnected: whatever it held is released (a switch claim, an outcome it was to apply), so it
        cannot hold the session forever. Its queued messages stay: they are part of the conversation's queue."""
        self._attachments.pop(client, None)
        if self._switch_owner == client:
            self._switch_owner = None
        if self._ending and self._ending_client == client:
            self.outcome_applied(self._ending, client)

    def _can_apply(self, client: str) -> bool:
        """`client` is there to apply an outcome (always, unless the daemon says it has no stream open)."""
        return self.connected is None or self.connected(client)

    def begin_close(self) -> None:
        """The clients are going away: nothing more is sent (a queued message stays unsent)."""
        self.closing = True

    def job_entered(self) -> None:
        """The client's own job (a workflow) has run its first line: an interrupt from now on reaches its `finally`."""
        self.turn_entered = True

    @property
    def queueing(self) -> bool:
        """A message sent now waits for the running turn, `/init` or `/compact` (not for an undo or a session load)."""
        return self.turn_active and not self.undoing and not self.switching and not self.workflow_active

    @property
    def session_starting(self) -> bool:
        return self._hook_task is not None and not self._hook_task.done()

    def busy_text(self, command: str) -> str:
        hint = "a session is loading" if self.switching and not self.turn_active else f"press Esc to interrupt, then {command}"
        return f"busy — {hint}"

    def other_session_busy(self) -> bool:
        """Another session of this daemon runs a job now: this one must not start one (see `OTHER_SESSION`)."""
        return self.elsewhere is not None and self.elsewhere()

    def refusal(self, command: str) -> str | None:
        """The busy line for `command` while a turn runs or a session loads (here, or a job in another session of the
        daemon), else None."""
        if self.turn_active or self.switching:
            return self.busy_text(command)
        return OTHER_SESSION if self.other_session_busy() else None

    def _publish(self, event: object) -> None:
        self.bus.publish(event)

    def _emit(self, event: AgentEvent) -> None:
        if isinstance(event, TurnEnded):
            self.job_status = event.status
        self._publish(event)

    # turns

    def send(
        self,
        text: str,
        *,
        label: str | None = None,
        init: bool = False,
        draft: str | None = None,
        pasted: Sequence[Image] | None = None,
        client: str = LOCAL_CLIENT,
    ) -> SendResult:
        """Start a turn for `text`, queue it behind the running one, or refuse it. `draft` is what the input box gets
        back if it never sends (the typed text; `text` when None); `pasted` overrides the client's attachments.
        `client` sent it: a text that goes back goes back to that client."""
        typed = text if draft is None else draft
        if init:
            if (reason := self.refusal("/init")) is not None:
                return SendResult("refused", reason)
        elif self.queueing:
            if not text.strip():
                return SendResult("ignored")
            self.queue.add(Queued(typed, text, label, client))
            self._publish(QueueChanged(self.queue.entries))
            return SendResult("queued")
        elif (reason := self.message_refusal()) is not None:
            return SendResult("refused", reason)
        self._begin(text, typed, init=init, label=label, pasted=pasted, parts=((client, typed),))
        return SendResult("started")

    def message_refusal(self) -> str | None:
        """Why a message cannot start now (and is not queued either), or None."""
        if not (self.turn_active or self.switching):
            return OTHER_SESSION if self.other_session_busy() else None
        if self.workflow_active:
            return WORKFLOW_RUNNING
        return f"busy — {'press Esc to interrupt' if self.turn_active else 'a session is loading'}"

    def _claim(self, kind: JobKind) -> None:
        self.turn_active = True
        self.job_status = None
        self.interrupted = False
        self.interrupted_by = ""
        self.turn_entered = False
        self._job = kind

    def _begin(
        self,
        text: str,
        draft: str,
        *,
        init: bool = False,
        label: str | None = None,
        pasted: Sequence[Image] | None = None,
        parts: tuple[tuple[str, str], ...],
    ) -> None:
        self._claim("turn")
        self._draft = draft
        self._draft_parts = parts
        self._origin = parts[0][0] if parts else LOCAL_CLIENT
        self._turn_no += 1
        self._task = asyncio.get_running_loop().create_task(self._turn(text, init, draft, label, pasted, self._turn_no))

    async def _take_attachments(self, pasted: Sequence[Image] | None, turn: int) -> tuple[list[Image], list[str]]:
        if pasted is not None:
            return list(pasted), []
        source = self._attachments.get(self._origin)  # the images pasted in the client that sent the turn
        if source is None:
            return [], []
        return await source(turn)

    async def _turn(
        self, text: str, init: bool, draft: str | None, label: str | None, given: Sequence[Image] | None, turn: int
    ) -> None:
        self.turn_entered = True
        agents_md = self.workspace / "AGENTS.md"
        before: tuple[int, int] | None = None
        sent = False
        shown = False  # the user entry was published (and is being drawn)
        started = False  # the agent was handed the message
        hook_events: list[AgentEvent] = []  # what the UserPromptSubmit hooks reported, shown once the prompt is
        context: list[str] = []  # and what they added for the model
        sent_text = text  # what the hooks and the model get: `text` plus the files its @mentions attach
        notes: tuple[MentionNote, ...] = ()
        images: tuple[Image, ...] = ()
        image_warnings: list[str] = []  # images that were not attached, and why
        pasted: list[Image] = []
        init_notice: tuple[str, str] | None = None

        async def work(emit: EventSink) -> None:
            nonlocal started
            started = True
            for hook_event in hook_events:
                emit(hook_event)
            await self.agent.run_turn(sent_text, emit, draft=draft, context=context, images=images, pasted=tuple(pasted))

        try:
            if self.not_ready and not await self.check():  # re-check before giving up on the model
                return
            if not init:  # /init's prompt is the app's own, not something the user sent
                expansion = await asyncio.to_thread(expand_mentions, text, self.workspace, self.guard)
                pasted, pasted_warnings = await self._take_attachments(given, turn)
                expansion = replace(expansion, images=(*expansion.images, *pasted))
                expansion, image_warnings = await apply_vision(expansion, self.vision, self.provider.model)
                image_warnings = [*pasted_warnings, *image_warnings]
                sent_text, notes, images = expansion.text, expansion.notes, expansion.images
                decision = await self.agent.prompt_hooks(sent_text, hook_events.append)
                if decision.blocked:  # the prompt is dropped: the end puts it back in the box
                    for warning in image_warnings:  # still true when it is sent again, so say it now
                        self._publish(Notice("warn", warning))
                    for hook_event in hook_events:
                        if isinstance(hook_event, Notice):
                            self._publish(hook_event)
                    await self.bus.drained()
                    return
                context = decision.context
            self._draft = None  # from here the message is part of the conversation
            sent = True
            before = fingerprint(agents_md)
            shown = True
            self._publish(UserEntry(text, label, self._origin))
            for note in notes:
                self._publish(Notice("info" if note.attached else "warn", note_text(note)))
            for number, image in enumerate(pasted, 1):  # the numbers edit_image's `clipboard:N` uses
                if any(image is i for i in images):
                    self._publish(Notice("info", f"image clipboard:{number} ({describe(image)})"))
            for warning in image_warnings:
                self._publish(Notice("warn", warning))
            self._publish(Delivering())
            await self.bus.drained()  # the entry is drawn: an interrupt until here keeps the message out
            await self._run_job(work)
        finally:
            try:
                if shown and not self.closing:
                    if not started:  # interrupted before the agent got the message: the text is kept
                        self._draft = draft if draft is not None else text
                    self._publish(EntryEnd(discard=not started))
                if init and sent and started and not self.closing:
                    await uninterrupted(self.bus.drained())  # the entry is closed before the check's notice
                    if fingerprint(agents_md) == before:
                        init_notice = ("info", "/init finished without changing AGENTS.md")
                        self._publish(Notice("info", init_notice[1]))
                    else:
                        init_notice = await self._check_init()
            finally:
                try:
                    await uninterrupted(self.bus.drained())
                finally:
                    self._end_turn(delivered=started, turn=turn, init_notice=init_notice)

    async def _check_init(self) -> tuple[str, str] | None:
        """After an `/init` turn that changed AGENTS.md: the commands the scan did not find (published too)."""
        try:
            notice: tuple[str, str] | None = await asyncio.to_thread(init_check, self.workspace)
        except Exception as error:
            notice = ("error", f"/init check failed: {error}")
        if notice is not None:
            level, text = notice
            self._publish(Notice(level, text))
        return notice

    async def _run_job(self, work: Callable[[EventSink], Awaitable[None]]) -> bool:
        """Run an agent job (a turn, a compaction) with its events on the bus; returns once they are drawn.
        False when the agent was busy and the job did not run."""
        ran = True
        try:
            await work(self._emit)
        except AgentBusy:
            ran = False
            self._publish(Notice("warn", "busy — a turn is already running"))
        finally:
            # Whatever ended the job, its events still reach the clients, and a repeated interrupt must not cut that
            # short.
            self._publish(JobEventsDone())
            await uninterrupted(self.bus.drained())
        return ran

    def compact(self) -> SendResult:
        """`/compact` as a job in the turn claim: an interrupt cancels it like a turn."""
        if (reason := self.refusal("/compact")) is not None:
            return SendResult("refused", reason)
        self._claim("compact")
        self._task = asyncio.get_running_loop().create_task(self._compact())
        return SendResult("started")

    async def _compact(self) -> None:
        self.turn_entered = True
        try:
            failed = False

            async def work(emit: EventSink) -> None:
                def watch(event: AgentEvent) -> None:
                    nonlocal failed
                    failed = failed or event == Notice("warn", NOT_SUMMARIZED)
                    emit(event)

                await self.agent.compact(watch)

            if await self._run_job(work) and not failed:  # a failed summary is not a finished compaction
                self.job_status = "done"
        finally:
            self._end_turn()

    def begin_workflow(self) -> None:
        """A workflow run by the client takes the turn claim; it ends with `end_turn`."""
        self._claim("workflow")
        self.workflow_active = True

    def end_turn(self) -> None:
        self._end_turn()

    def outcome_applied(self, ack: int, client: str | None = None) -> bool:
        """The client that sent the turn applied the outcome `ack` (its text is back in the box): the claim is
        released, so "turn ended" is one step for the client. A message queued meanwhile is sent now; True when it
        started a turn. Called again, for another outcome, or by another client, it does nothing. `client` None is
        the host itself (the timeout, closing)."""
        if ack == 0 or ack != self._ending:
            return False
        if client is not None and client != self._ending_client:
            return False
        self._ending = 0
        self._ending_client = ""
        if self._ending_timer is not None:
            self._ending_timer.cancel()
            self._ending_timer = None
        kind, interrupted, by = self._job, self.interrupted, self.interrupted_by
        self.turn_active = False
        self.workflow_active = False
        self.interrupted = False
        self.interrupted_by = ""
        self.job_status = None
        queued = self._take_queue()  # sent after the outcome was decided, while the claim was still held
        if not queued or self.closing:
            return False
        merged = merge(queued)
        if interrupted:  # Esc came after them: they go back to the box, as for an interrupted turn
            self._publish(
                TurnOutcome(kind, "interrupted", queue_restore=merged.draft, interrupted_by=by, restores=merged.parts)
            )
            return False
        self._begin(merged.prompt, merged.draft, label=merged.label, parts=merged.parts)
        return True

    def _take_queue(self) -> tuple[Queued, ...]:
        """Everything queued, taken out (sent or given back); the clients are told when there was anything."""
        queued = self.queue.take()
        if queued:
            self._publish(QueueChanged(()))
        return queued

    def _hold_until_applied(self, client: str) -> int:
        """Keep the claim until `client` applies the outcome about to be published, at most OUTCOME_ACK_TIMEOUT (a
        client that never answers must not hold the session forever). Returns the outcome's ack."""
        self._outcomes += 1
        ack = self._ending = self._outcomes
        self._ending_client = client
        if self._ending_timer is not None:
            self._ending_timer.cancel()
        self._ending_timer = asyncio.get_running_loop().call_later(OUTCOME_ACK_TIMEOUT, self.outcome_applied, ack)
        return ack

    def _end_turn(self, *, delivered: bool = False, turn: int = 0, init_notice: tuple[str, str] | None = None) -> None:
        """Every job ends here: a `done` end sends the queue as one turn, any other gives it back to the client (and
        the claim stays until the client applied that, see `outcome_applied`)."""
        if self.on_job_end is not None:
            with contextlib.suppress(Exception):  # its owner's bug must not keep the turn from ending
                self.on_job_end()
        kind = self._job
        status = self.job_status
        origin = self._origin if kind == "turn" else LOCAL_CLIENT  # a compaction and a workflow are the local client's
        by = self.interrupted_by
        done = status == "done" and not self.interrupted
        self.job_status = None
        self.interrupted = False
        self.interrupted_by = ""
        queued = self._take_queue()
        draft, self._draft = self._draft, None  # set when the turn ended before it sent the message
        parts = self._draft_parts if draft is not None else ()
        self._draft_parts = ()
        if queued and done and draft is None and not self.closing:
            merged = merge(queued)
            self.workflow_active = False
            self._begin(merged.prompt, merged.draft, label=merged.label, parts=merged.parts)  # the claim stays on
            self._publish(
                TurnOutcome(
                    kind, status, continued=True, delivered=delivered, turn=turn, init_check=init_notice, client=origin
                )
            )
            return
        merged_back = merge(queued) if queued and not self.closing else None
        if self.closing or not self._can_apply(origin):  # no client will apply it
            self.turn_active = False
            self.workflow_active = False
            ack = 0
        else:
            ack = self._hold_until_applied(origin)
        self._publish(
            TurnOutcome(
                kind,
                status,
                restore_text=draft,
                queue_restore=None if merged_back is None else merged_back.draft,
                delivered=delivered,
                turn=turn,
                init_check=init_notice,
                ack=ack,
                client=origin,
                interrupted_by=by,
                restores=(*parts, *(() if merged_back is None else merged_back.parts)),
            )
        )

    def interrupt(self, client: str = LOCAL_CLIENT) -> None:
        """Esc in `client`: cancel the running turn or compaction and the SessionStart hooks. The queue is not
        sent."""
        if self.undoing:  # an undo is not a turn: there is nothing to cancel, and its guard must stay up
            return
        for task in (self._task, self._hook_task):
            if task is not None and not task.done():
                task.cancel()
        if self._ending:  # the job already ended; only its outcome is not applied yet: what was queued since goes back
            self.interrupted = True
            self.interrupted_by = client
            return
        if self.turn_active:
            self.interrupted = True  # also after `done`, while the job still finishes (`/init`'s file check)
            self.interrupted_by = client
        if self.turn_active and not self.turn_entered:
            # Cancelled before its first line ran: its `finally` never will.
            self._end_turn()

    # the provider

    async def check(self) -> bool:
        """Ask the provider whether it is ready (published as `ProviderState`); True when it is."""
        try:
            problem = await self.provider.check()
        except Exception as exc:  # the contract says it returns a message; a bug must not kill the session
            problem = f"Provider check failed: {exc}"
        self.not_ready = problem is not None
        self._publish(ProviderState(problem))
        return problem is None

    # commands

    def init_prompt(self) -> str:
        """The `/init` message: the file's state and the parent files already loaded (reads the disk)."""
        own = levels(self.workspace)[-1][0]
        parents = [f.label for f in self.agent.instructions().files if f.label != own and f.state != "omitted"]
        block = render_facts(scan(self.workspace), int(FACTS_SHARE * self.provider.context_window) * CHARS_PER_TOKEN)
        return build_init_prompt(self.workspace, (self.workspace / "AGENTS.md").is_file(), parents, facts=block)

    async def undo(self, client: str = LOCAL_CLIENT) -> UndoResult:
        """Undo the last turn, holding the turn claim meanwhile (nothing to interrupt). The other clients of the session
        are told (`TurnUndone`, then the todo list as it is now) so they drop the turn too."""
        if (reason := self.refusal("/undo")) is not None:
            return UndoResult(None, [Notice("warn", reason)])
        self.turn_active = True
        self.undoing = True
        events: list[AgentEvent] = []
        prompt: str | None = None
        try:
            prompt = await self.agent.undo(events.append)
        except AgentBusy:
            events.append(Notice("warn", self.busy_text("/undo")))
        except Exception as error:
            events.append(Notice("error", f"Undo failed: {error}"))
        finally:
            self.turn_active = False
            self.undoing = False
        if prompt is not None:
            self._publish(TurnUndone(client))
            self._publish(TodosChanged(self.agent.todos))
        return UndoResult(prompt, events)

    def set_mode(self, mode: Any) -> None:
        self.agent.set_mode(mode)
        self._publish(ModeChanged(self.agent.mode))

    # sessions

    def begin_switch(self, client: str = LOCAL_CLIENT) -> bool:
        """Claim the right to switch sessions or models for `client`: False while a turn runs or another switch is
        under way. It is released by `end_switch` from the same client, or when that client disconnects."""
        if self.turn_active or self.switching:
            return False
        self._switch_owner = client  # no message is accepted until the switch ends
        return True

    def end_switch(self, client: str = LOCAL_CLIENT) -> None:
        """Release `client`'s switch claim; another client's stays."""
        if self._switch_owner == client:
            self._switch_owner = None

    @contextlib.contextmanager
    def _switch_held(self, client: str, command: str) -> Generator[str | None]:
        """For a switch: `client`'s own claim when it holds it, else the claim taken now and released at the end.
        Yields the busy line (and holds nothing) when another client holds it or a turn runs."""
        if self._switch_owner == client:
            yield None
            return
        if not self.begin_switch(client):
            yield self.busy_text(command)
            return
        try:
            yield None
        finally:
            self.end_switch(client)

    @contextlib.contextmanager
    def hold_claim_for_test(self, claim: Literal["turn", "switch"]) -> Generator[None]:
        """Tests only: hold the turn or the switch claim with no job behind it, as a running one would; released on
        exit."""
        if claim == "turn":
            self.turn_active = True
        else:
            self._switch_owner = HOST
        try:
            yield
        finally:
            if claim == "turn":
                self.turn_active = False
            else:
                self._switch_owner = None

    async def close_old(self, old: Recorder | None) -> None:
        """Close the replaced recorder in a thread (a final write must not block), holding the switch so no message
        is accepted meanwhile. The new recorder is already in place: nothing more is written to the old session."""
        if old is None:
            return
        held = not self.switching
        if held:
            self._switch_owner = HOST
        try:
            await uninterrupted(asyncio.to_thread(close_recorder, old))
        finally:
            if held and self._switch_owner == HOST:
                self._switch_owner = None

    async def new_session(self, command: str = "/new", client: str = LOCAL_CLIENT) -> str | None:
        """`/new` and `/clear`: a fresh session; the old file stays. Returns the busy line when refused. On success the
        switch claim is `client`'s (no message until it has redrawn); `end_switch(client)` releases it."""
        try:
            # Refused while a turn runs or a session loads: the history belongs to them until they end.
            if self.turn_active or self.switching:
                raise AgentBusy
            old = self.agent.reset(recorder=self.recorder_factory())
        except AgentBusy:
            return self.busy_text(command)
        self._switch_owner = client
        try:
            await self.close_old(old)
        except BaseException:
            self.end_switch(client)
            raise
        return None

    async def open_session(self, info: SessionInfo, client: str = LOCAL_CLIENT) -> OpenedSession:
        """Load `info` and make it the live session. `client`'s switch claim is used when it holds it, else one is
        taken for the call (refused while another client's switch or a turn runs). `loaded` None: not opened."""
        store = self.store
        if store is None:
            return OpenedSession(None, [])
        with self._switch_held(client, "/sessions") as refused:
            if refused is not None:
                return OpenedSession(None, [("warn", refused)], AgentBusy())
            try:
                # In a thread: reading the file, and the session lock (which may wait out another process's probe).
                loaded, recorder = await asyncio.to_thread(self._load_and_lock, store, info.id)
            except (SessionError, SessionWriteError) as exc:  # stay on the session we have
                return OpenedSession(None, [("error", str(exc))], exc)
            try:
                if self.turn_active:  # a turn started while the file was read
                    raise AgentBusy
                old = self.agent.reset(
                    recorder=recorder,
                    history=loaded.history,
                    tool_meta=loaded.tool_meta,
                    todos=loaded.todos,
                    usage=loaded.usage,
                    unrecorded=loaded.unrecorded,
                )
            except AgentBusy as exc:
                close_recorder(recorder)  # releases its session lock: the session stays free
                return OpenedSession(None, [("warn", self.busy_text("/sessions"))], exc)
            await self.close_old(old)
            return OpenedSession(loaded, [])

    def _load_and_lock(self, store: SessionStore, session_id: str) -> tuple[LoadedSession, FileRecorder]:
        loaded = store.load(session_id)
        return loaded, FileRecorder.resume(loaded.info.path, self.workspace)

    async def list_sessions(self) -> list[SessionInfo]:
        if self.store is None:
            return []
        return await asyncio.to_thread(self.store.list)

    async def session_started(self, source: SessionSource) -> list[AgentEvent]:
        """Run the SessionStart hooks as the host's own task, which `interrupt` cancels (the runner then kills the
        hook's process group). Returns their notices; never raises but for the caller's own cancel."""
        if not self.hooks.has("SessionStart"):
            return []  # nothing to run, so nothing to interrupt
        events: list[AgentEvent] = []

        async def run() -> None:
            try:
                await self.agent.session_started(source, events.append)
            except Exception as exc:  # the agent does not raise; a bug here must not take the session down
                events.append(hook_failed("SessionStart", exc))

        task = asyncio.get_running_loop().create_task(run())
        self._hook_task = task
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            if self._hook_task is task:
                self._hook_task = None
        if task.cancelled():
            events.append(Notice("info", "hook SessionStart interrupted"))
        return events

    # models

    async def list_models(self) -> list[ModelInfo]:
        """The installed models; raises CatalogError (also when there is no catalog)."""
        if self.catalog is None:
            raise CatalogError("No model catalog is available")
        return await self.catalog.list()

    async def model_length(self, name: str) -> int | None:
        if self.catalog is None:
            raise CatalogError("No model catalog is available")
        return await self.catalog.context_length(name)

    async def switch_model(
        self, name: str, looked: tuple[int | None, str | None] | None = None, client: str = LOCAL_CLIENT
    ) -> ModelSwitch:
        """Make `name` (an installed model) the one every later request uses, with a window that fits it, and
        remember it for the project. Ollama failing leaves the old model and window in place. `client`'s switch
        claim is used when it holds it, else one is taken for the call (refused while another's is held)."""
        with self._switch_held(client, "/model") as refused:
            if refused is not None:
                return ModelSwitch([("warn", refused)])
            return await self._switch_model(name, looked)

    async def _switch_model(self, name: str, looked: tuple[int | None, str | None] | None) -> ModelSwitch:
        notices: list[tuple[Level, str]] = []
        if self.catalog is None:
            return ModelSwitch([("error", "No model catalog is available")])
        try:
            installed = [info.name for info in await self.catalog.list()]
        except CatalogError as error:
            return ModelSwitch([("error", f"Could not list the installed models: {error}")])
        target = installed_name(name, installed)
        if target is None:
            listed = ", ".join(escape_display(n) for n in installed) or "none"
            return ModelSwitch([("error", f"Model {shown_name(name)} is not installed. Installed: {listed}")])
        if installed_name(self.provider.model, installed) == target:
            return ModelSwitch([("info", f"Already using {escape_display(target)}.")])
        length: int | None = None
        failed: str | None = None
        if looked is not None:  # the picker already asked
            length, failed = looked
        else:
            try:
                async with asyncio.timeout(SHOW_TIMEOUT):
                    length = await self.catalog.context_length(target)
            except CatalogError as error:
                failed = str(error)
            except TimeoutError:
                failed = f"Ollama did not answer within {SHOW_TIMEOUT:g} s"
        window = resolve_window(length, self.window_cap)
        self.provider.switch(target, window)
        notices.append(("info", model_notice(target, window)))
        if failed is not None:
            notices.append(
                (
                    "warn",
                    f"Could not read the context length of {escape_display(target)}; using AGENT_NUM_CTX ({self.window_cap}): {failed}",
                )
            )
        recorded: list[AgentEvent] = []
        self.agent.model_switched(target, window, recorded.append)
        notices.extend((event.level, event.text) for event in recorded if isinstance(event, Notice))
        try:
            self.permissions.save_project_key("model", target)
        except RuleSavedNotTrusted as error:
            notices.append(("warn", escape(str(error))))
        except (OSError, ValueError) as error:
            notices.append(("warn", f"Model saved for the run only: {escape(str(error))}"))
        return ModelSwitch(notices, target, window)

    # the last turn's diff

    async def diff_list(self) -> TurnDiff | None:
        return await self.agent.last_turn_diff()

    async def diff_file(self, diff: TurnDiff, item: DiffItem) -> FileDiff:
        return await self.agent.load_diff_file(diff, item)

    # life cycle

    def busy_tasks(self) -> list[asyncio.Task[None]]:
        """The job and hook tasks still running."""
        return [t for t in (self._task, self._hook_task) if t is not None and not t.done()]

    async def idle(self) -> None:
        """Wait until no job or hook task runs and every subscriber has drawn what it was given."""
        while True:
            if tasks := self.busy_tasks():
                await asyncio.wait(tasks)
                continue
            await self.bus.drained()
            if not self.busy_tasks():
                return

    def close(self) -> None:
        """The client is going away: cancel the job and the hooks and withdraw every ask (as the app's workers are
        cancelled when it quits; their cleanup runs on its own). Never raises."""
        self.closing = True
        self.outcome_applied(self._ending)  # no client will apply it now: release the claim (nothing is sent)
        self.asks.cancel_all("closing")
        for task in (self._task, self._hook_task):
            if task is not None and not task.done():
                task.cancel()

    async def aclose(self) -> None:
        """`close`, then wait until the cancelled job and hooks have run their cleanup (the daemon awaits this before it
        closes the runtime). Never raises but for the caller's own cancel."""
        self.close()
        tasks = self.busy_tasks()
        if tasks:
            await asyncio.wait(tasks)
