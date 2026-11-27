"""The wiring of a run: settings, clients, rules, guardrails, hooks, tools and MCP, shared by the app and headless mode.

UI-free (no Textual): `src.tui.run` builds the app from a `Runtime`, `src.headless` drives the agent with it."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import os
import sys
import tempfile
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from core.client import OllamaClient, SearxClient, WebClient
    from core.client.clipboard import PasteFolder
    from core.config.settings import Settings
    from core.engine import Agent
    from core.engine.agents.loader import Agents
    from core.engine.approval import Approver
    from core.engine.checkpoints.store import CheckpointStore
    from core.engine.commands.loader import Commands
    from core.engine.configguard import ConfigGuard, HeldGlobal
    from core.engine.configwatch import ConfigWatch
    from core.engine.guardrails.setup import Guardrails
    from core.engine.hooks.runner import ProcessRunner
    from core.engine.hooks.setup import Hooks
    from core.engine.image_pull import ImagePuller
    from core.engine.imagegen import MfluxGenerator
    from core.engine.mcp.contract import McpConnector
    from core.engine.mcp.hub import McpHub
    from core.engine.mcp.setup import McpSetup
    from core.engine.permissions.setup import PermissionSetup
    from core.engine.provider.base import SwitchableProvider
    from core.engine.provider.models import ModelCatalog, ModelInfo
    from core.engine.provider.vision import VisionCheck
    from core.engine.session import Recorder, SessionStore
    from core.engine.skills.loader import Skills
    from core.engine.terminal.manager import TerminalManager
    from core.engine.tools.media import AssembleVideoTool, SpeakTool
    from core.engine.workflows.loader import Workflows


@dataclass(frozen=True)
class StartChoice:
    """How a run begins: a new session, the latest one, the picker, or a session by id prefix."""

    kind: Literal["new", "continue", "pick", "resume"]
    session_id: str | None = None


NEW = StartChoice("new")


def _last_check(watch: ConfigWatch) -> None:
    """A check with the background cause that never raises: the end of the build, and the quit checks after every
    child process group was stopped."""
    from core.engine.configwatch import Cause

    with contextlib.suppress(Exception):
        watch.check(Cause.background())


def _speaker(found: Path | None, settings: Settings, guard: Guardrails) -> SpeakTool | None:
    """The speak tool for the main agent; None when the speech engine is not installed."""
    from core.engine.tools.media import SpeakTool

    return None if found is None else SpeakTool(settings.speech_model, settings.speech_timeout, guard)


def _video(found: Path | None, settings: Settings, guard: Guardrails) -> AssembleVideoTool | None:
    """The assemble_video tool for the main agent; None when ffmpeg or the media helper is not there."""
    from core.engine import media
    from core.engine.tools.media import AssembleVideoTool

    helper = media.MEDIA_HELPER
    if found is None or helper is None or not helper.is_file():
        return None
    return AssembleVideoTool(found, settings.video_timeout, guard)


def _image_generators(
    commands: tuple[Path | None, Path | None, Path | None], settings: Settings, guard: Guardrails
) -> tuple[MfluxGenerator | None, MfluxGenerator | None, MfluxGenerator | None]:
    """The mflux generators (generate_image, edit_image, upscale_image) for the main agent; each is None when mflux
    was not asked for or its own command was not found."""
    from core.engine.imagegen import EDIT_EXTRA, UPSCALE_EXTRA, MfluxGenerator

    generate, edit, upscale = commands
    imager = None if generate is None else MfluxGenerator(generate, settings.image_model, settings.image_timeout, guard)
    editor = (
        None
        if edit is None
        else MfluxGenerator(edit, settings.image_edit_model, settings.image_timeout, guard, extra=EDIT_EXTRA)
    )
    upscaler = (
        None
        if upscale is None
        else MfluxGenerator(
            upscale, settings.image_upscale_model, settings.image_timeout, guard, extra=UPSCALE_EXTRA, kind="upscale"
        )
    )
    return imager, editor, upscaler


@dataclass
class Runtime:
    """Everything a launch builds once. `close` (async, inside the loop that owns the connections) and `kill_all`
    (sync fallback after that loop is gone) stop what the run started; `abort` is for a launch that fails before
    anything could call `close`."""

    settings: Settings
    root: Path
    client: OllamaClient
    web: WebClient
    search: SearxClient | None
    provider: SwitchableProvider
    store: SessionStore
    recorder_factory: Callable[[], Recorder]
    agent_factory: Callable[[Approver], Agent]
    mcp: McpHub
    setup: McpSetup
    permissions: PermissionSetup
    hooks: Hooks
    hook_runner: ProcessRunner
    guard: Guardrails
    checkpoints: CheckpointStore | None
    terminals: TerminalManager | None
    commands: Commands
    skills: Skills
    agents: Agents
    workflows: Workflows
    catalog: ModelCatalog
    vision: VisionCheck  # shared by the app, headless mode and the agents: one answer per model
    config_watch: ConfigWatch  # the agent's own config and trust files, checked around every command
    config_guard: ConfigGuard  # what happens on a change; its `listeners` hear the warning text
    global_held: bool  # the global marker was set at launch: global allow rules, hooks and servers wait for trust
    held: HeldGlobal  # and what else it holds until the next launch: global keepEnv, commands, skills, agents, workflows
    model_notices: list[str]
    warnings: tuple[str, ...]
    close: Callable[[], Awaitable[None]]
    paste: PasteFolder
    images: ImagePuller | None = None  # the app only: `/images` and the tools' refusal while a warm-up runs

    def kill_all(self) -> None:
        """Synchronous fallback once the loop is gone: whatever `close` did not stop is killed."""
        if self.images is not None:
            self.images.kill_all()  # a warm-up's mflux (a no-op once `close` ran)
        self.hook_runner.kill_all()  # nor a hook (a no-op once `close` ran)
        self.paste.remove()  # pasted images (idempotent)
        if self.checkpoints is not None:
            self.checkpoints.close()  # idempotent: the shadow folder is gone after the first call
        self.mcp.kill_all()  # a server still running means `close` never ran, and its loop is gone
        if self.terminals is not None and self.terminals.names():  # none when `close` ran
            self.terminals.kill_all()
        _last_check(self.config_watch)  # what they wrote before they died

    def abort(self) -> None:
        """Close the web and search clients of a launch that failed before it could be closed (no loop is running)."""
        _abort(self.web, self.search)


class _ListOnce:
    """A catalog whose `list()` asks once per launch (the launch model and the agents' models share the answer;
    a `CatalogError` is kept too). A call that is cancelled or times out is not kept."""

    def __init__(self, inner: ModelCatalog) -> None:
        self._inner = inner
        self._models: builtins.list[ModelInfo] | None = None
        self._error: Exception | None = None

    async def list(self) -> builtins.list[ModelInfo]:
        if self._error is not None:
            raise self._error
        if self._models is None:
            try:
                self._models = await self._inner.list()
            except Exception as error:
                self._error = error
                raise
        return builtins.list(self._models)

    async def context_length(self, name: str) -> int | None:
        return await self._inner.context_length(name)

    async def capabilities(self, name: str) -> tuple[str, ...]:
        return await self._inner.capabilities(name)


def _abort(web: WebClient, search: SearxClient | None) -> None:
    try:
        asyncio.run(web.aclose())
    finally:
        if search is not None:
            asyncio.run(search.aclose())


def build_runtime(
    workspace: Path,
    builtins: Collection[str],
    *,
    model: str | None = None,
    terminals: bool = True,
    provider: SwitchableProvider | None = None,
    connector: McpConnector | None = None,
    trust_project_mcp: bool = False,
    checkpoints: bool = False,
    catalog: ModelCatalog | None = None,
    images: bool = False,
) -> Runtime:
    """Read the settings files once and wire the run for `workspace` (a folder).

    `model` (`--model`) is the model for this launch only: it beats `AGENT_MODEL` and the project's `model` key.
    `builtins` are the slash commands a custom command may not shadow. With `terminals=False` there is no
    terminal manager and no `term_*` tool. `provider` replaces the Ollama one and skips the launch model lookup
    (for tests), and so does `connector` for the MCP servers. `trust_project_mcp` treats the project's MCP file as
    trusted for this run (nothing is remembered). `images=True` (the app) offers `generate_image` to the main agent when
    mflux is found. `checkpoints=True` (the app) snapshots the workspace around each turn for `/undo`."""
    from core.client import OllamaClient, SearxClient, WebClient
    from core.client.clipboard import PasteFolder
    from core.client.clipboard import prune_stale as prune_pastes
    from core.client.mcp import SdkConnector
    from core.client.ollama import OllamaCatalog, OllamaUnreachable
    from core.config.settings import get_settings
    from core.engine import Agent
    from core.engine.agents.loader import load_agents
    from core.engine.checkpoints.setup import prepare_checkpoints
    from core.engine.commands.loader import load_commands
    from core.engine.configguard import ConfigGuard, GlobalMarker, HeldGlobal, marker_path
    from core.engine.configwatch import Cause, ConfigWatch, WatchedSet
    from core.engine.guardrails.setup import Guardrails
    from core.engine.guardrails.setup import prepare as prepare_guardrails
    from core.engine.hooks.runner import ProcessRunner
    from core.engine.hooks.setup import prepare as prepare_hooks
    from core.engine.image_pull import HF_COMMAND, ImagePuller
    from core.engine.imagegen import EDIT_COMMAND, UPSCALE_COMMAND, find_mflux
    from core.engine.mcp.config import global_config_path
    from core.engine.mcp.hub import McpHub
    from core.engine.mcp.setup import prepare
    from core.engine.mcp.trust import TrustStore, trust_store_path
    from core.engine.media import find_ffmpeg, find_speech
    from core.engine.permissions import setup as permission_setup
    from core.engine.permissions.config import global_settings_path, project_settings_path
    from core.engine.provider.models import SHOW_TIMEOUT, SLOW_AFTER, CatalogError, launch_model, shown_name
    from core.engine.provider.ollama import OllamaProvider
    from core.engine.provider.vision import VisionCheck
    from core.engine.session import FileRecorder, SessionStore, sessions_root, workspace_dir
    from core.engine.skills.loader import builtin_skills_dir, load_skills
    from core.engine.terminal.manager import TerminalManager
    from core.engine.tools import default_tools, read_only_tools
    from core.engine.tools.base import ToolContext
    from core.engine.workflows.loader import load_workflows

    root = workspace.resolve()
    # The config watch's baseline comes first, before any loader reads a config file: a change between this and the
    # end of the build is reported by the first check. Its reaction object is bound once the rules and the hub exist.
    config_watch = ConfigWatch(WatchedSet.for_run(root, os.environ, Path.home()))
    settings = get_settings()
    client = OllamaClient(settings.ollama_url)
    sessions = sessions_root(settings.sessions_dir)
    store = SessionStore(sessions, root)
    web = WebClient(settings.fetch_timeout, settings.fetch_max_bytes)
    search: SearxClient | None = None
    try:
        search = SearxClient(settings.search_url, settings.fetch_timeout) if settings.search_url else None
        mcp_store = TrustStore(trust_store_path())
        permission_store = TrustStore(trust_store_path(name="permission-trust.json"))
        # An earlier run saw a command change the global config (the marker): until the user trusts it, its allow
        # rules, hooks and MCP servers are off, its keepEnv is ignored and its commands, skills, agents and workflows
        # are not loaded (the app asks; headless mode only warns). Read before any loader. Deny rules always apply.
        marker = GlobalMarker(marker_path())
        global_held = marker.active

        skills = load_skills(root, builtin=builtin_skills_dir(), hold_global=global_held)  # read once, like the commands
        # None: no tool, no warning (not macOS arm64, or no extra); each command is found on its own
        image_commands = (
            (find_mflux(), find_mflux(command=EDIT_COMMAND), find_mflux(command=UPSCALE_COMMAND))
            if images
            else (None, None, None)
        )
        speech_found = find_speech() if images else None  # same for the speak tool (the `media` extra)
        ffmpeg_found = find_ffmpeg() if images else None  # and for assemble_video (the same extra)
        agents = load_agents(root, hold_global=global_held)  # same; `model:` names are checked at launch in Phase 3
        workflows = load_workflows(root, hold_global=global_held)  # same: read once (the app's `/workflow` only)
        probe_imager, probe_editor, probe_upscaler = _image_generators(image_commands, settings, Guardrails.off(root))
        # The permission rules: global ones are active; the project's allow rules wait for the user's trust. "Always
        # here" and the trust writes are the agent's own (`own_write`).
        permissions = permission_setup.prepare(
            root,
            global_settings_path(),
            permission_store,
            # to warn about rules that never apply; the manager is only asked for its tools' names, it starts nothing
            permission_setup.tool_keys(
                default_tools(
                    TerminalManager(root),
                    fetcher=web,
                    searcher=search,
                    skills=skills,
                    imager=probe_imager,
                    editor=probe_editor,
                    upscaler=probe_upscaler,
                    speaker=_speaker(speech_found, settings, Guardrails.off(root)),
                    video=_video(ffmpeg_found, settings, Guardrails.off(root)),
                )
            ),
            own_write=config_watch.own_write,
        )
        rules = permissions.rules
        # Both MCP files are read once, here; the project's servers stay untrusted until the user says yes, and the
        # trust write is refused once the rules were revoked.
        setup = prepare(root, global_config_path(), mcp_store, config_watch.own_write, revoked=lambda: rules.revoked)
        # The model: --model, then AGENT_MODEL, then the project's `model` key (read without trust: it only picks an
        # installed model), then the default. A throwaway client, because this loop is closed before the app's starts.
        model_notices: list[str] = []

        def say_slow(name: str) -> None:  # before anything is drawn: the user is waiting on Ollama
            print(f"Asking Ollama about {shown_name(name)}…", file=sys.stderr, flush=True)

        async def pick_model() -> tuple[str, int, list[str], Agents]:
            probe = OllamaClient(settings.ollama_url)
            try:
                found = _ListOnce(catalog if catalog is not None else OllamaCatalog(probe))
                chosen_model, chosen_window, notes = await launch_model(
                    settings,
                    model,
                    permissions.project_data,
                    found,
                    notices=model_notices,
                    on_slow=say_slow,
                    slow_after=SLOW_AFTER,
                )
                installed: list[str] | None = None
                if agents.needs_models():  # one listing, only when an agent file names a model
                    try:
                        installed = [m.name for m in await asyncio.wait_for(found.list(), SHOW_TIMEOUT)]
                    except (CatalogError, TimeoutError, OllamaUnreachable):
                        installed = None  # unknown: the names are taken as written
                return chosen_model, chosen_window, notes, agents.with_models(installed)
            finally:
                await probe.aclose()

        model_warnings: list[str] = []
        if provider is None:
            chosen, window, model_warnings, agents = asyncio.run(pick_model())
            provider = OllamaProvider(client, chosen, window)
        else:
            agents = agents.with_models(None)  # an injected provider: no catalog call
        run_provider = provider
        # Guardrails come from the same two settings files, already parsed: protected paths and the child environment.
        guard = prepare_guardrails(
            root,
            permissions.global_data,
            permissions.project_data,
            os.environ,
            global_path=global_settings_path(),
            project_path=project_settings_path(root),
            sessions=sessions,  # accept-edits never writes it
            hold_keep_env=global_held,
        )
        if global_held:
            permissions.rules.disable_global(until_trusted=True)
        # Hooks too: both files already parsed; project hooks stay off until the same trust as the allow rules.
        hook_runner = ProcessRunner(root, guard, config_watch)
        hooks = prepare_hooks(permissions, root, hook_runner)
        shadow, shadow_warnings = prepare_checkpoints(
            root, guard, guard.env.filter(os.environ), enabled=checkpoints
        )
        run_catalog = catalog if catalog is not None else OllamaCatalog(client)
        vision = VisionCheck(run_catalog)  # the uncached catalog: the check keeps its own answers
        ctx = ToolContext(root, settings.bash_timeout, guard=guard)
        # lives for the whole run; /new and session switches do not touch it
        manager = TerminalManager(root, guard=guard) if terminals else None
        commands = load_commands(root, builtins, hold_global=global_held)  # read once, like the settings files
        held = HeldGlobal(guard.held_keep_env, commands.held, skills.held, agents.held, workflows.held)
        mcp = McpHub(  # nothing starts yet
            connector if connector is not None else SdkConnector(guard=guard),
            root,
            setup.servers,
            settings.mcp_timeout,
            setup.warnings,
            start_timeout=settings.mcp_start_timeout,
            untrusted=(frozenset() if trust_project_mcp else setup.untrusted)
            | (frozenset(s.name for s in setup.servers if s.source == "global") if global_held else frozenset()),
        )
        config_guard = ConfigGuard(root, permissions.rules, mcp, [permission_store, mcp_store], marker)
        config_guard.bind(config_watch)  # from here a change is reacted to (one seen before is handed over)

        puller: ImagePuller | None = None
        if images:
            pull_gen, pull_edit, pull_upscale = _image_generators(image_commands, settings, guard)
            puller = ImagePuller(
                {"generate": pull_gen, "edit": pull_edit, "upscale": pull_upscale}, find_mflux(command=HF_COMMAND), guard
            )

        def recorder_factory() -> Recorder:
            return FileRecorder(workspace_dir(sessions, root), root)  # lazy: no file until the first message

        def agent_factory(approver: Approver) -> Agent:
            imager, editor, upscaler = _image_generators(image_commands, settings, guard)
            return Agent(
                run_provider,
                default_tools(
                    manager,
                    fetcher=web,
                    searcher=search,
                    skills=skills,
                    imager=imager,
                    editor=editor,
                    upscaler=upscaler,
                    pull=puller,
                    speaker=_speaker(speech_found, settings, guard),
                    video=_video(ffmpeg_found, settings, guard),
                ),
                approver,
                ctx,
                settings.max_steps,
                recorder=recorder_factory(),
                compact_at=settings.compact_at,
                compact_keep_turns=settings.compact_keep_turns,
                instructions_share=settings.instructions_share,
                extra_tools=mcp.tools,
                task_tools=read_only_tools(web, search, skills),
                agents=agents,
                task_max_steps=settings.task_max_steps,
                rules=permissions.rules,
                hooks=hooks,
                checkpoints=shadow,
                vision=vision,
                watch=config_watch,
            )

        prune_pastes(Path(tempfile.gettempdir()), Path.home() / ".cache" / "agent")  # folders of dead runs
        paste = PasteFolder(root)
        # Every loader has read its files: a change made during the build is reacted to now, before anything runs
        # (the first listener hears it).
        _last_check(config_watch)

        async def close() -> None:
            with contextlib.suppress(Exception):  # a failing check must not keep the cleanup below from running
                config_watch.check(Cause.background())  # first: a process that outlived the last turn, before anything is stopped
                await config_guard.settle()  # the servers the reaction stops
            if puller is not None:
                await puller.cancel()  # first: it may be downloading, and its mflux must not outlive the run
            paste.remove()
            hook_runner.kill_all()  # a hook still running is killed with its process group, and no more start
            if shadow is not None:
                shadow.close()  # the run's shadow repository is deleted
            try:
                await mcp.close_all()  # no server process outlives the agent
            finally:
                try:
                    if manager is not None:
                        await manager.close_all()  # nor any terminal process
                finally:
                    try:
                        await web.aclose()
                    finally:
                        try:
                            if search is not None:
                                await search.aclose()
                        finally:
                            try:
                                await client.aclose()
                            finally:
                                _last_check(config_watch)  # the hooks, servers and terminals are gone now

        return Runtime(
            settings=settings,
            root=root,
            client=client,
            web=web,
            search=search,
            provider=run_provider,
            store=store,
            recorder_factory=recorder_factory,
            agent_factory=agent_factory,
            mcp=mcp,
            setup=setup,
            permissions=permissions,
            hooks=hooks,
            hook_runner=hook_runner,
            guard=guard,
            checkpoints=shadow,
            terminals=manager,
            commands=commands,
            skills=skills,
            agents=agents,
            workflows=workflows,
            catalog=run_catalog,
            vision=vision,
            config_watch=config_watch,
            config_guard=config_guard,
            global_held=global_held,
            held=held,
            model_notices=model_notices,
            warnings=(*permissions.warnings, *agents.warnings, *workflows.warnings, *model_warnings, *guard.warnings, *hooks.warnings, *shadow_warnings),
            close=close,
            paste=paste,
            images=puller,
        )
    except BaseException:
        _abort(web, search)  # setup failed before anyone could close them (no loop is running here)
        raise
