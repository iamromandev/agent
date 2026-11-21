"""The wiring of a run: settings, clients, rules, guardrails, hooks, tools and MCP, shared by the app and headless mode.

UI-free (no Textual): `src.tui.run` builds the app from a `Runtime`, `src.headless` drives the agent with it."""

from __future__ import annotations

import asyncio
import builtins
import os
import platform
import sys
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.client import OllamaClient, SearxClient, WebClient
    from src.config.settings import Settings
    from src.engine import Agent
    from src.engine.agents.loader import Agents
    from src.engine.approval import Approver
    from src.engine.checkpoints.store import CheckpointStore
    from src.engine.commands.loader import Commands
    from src.engine.guardrails.setup import Guardrails
    from src.engine.hooks.runner import ProcessRunner
    from src.engine.hooks.setup import Hooks
    from src.engine.mcp.contract import McpConnector
    from src.engine.mcp.hub import McpHub
    from src.engine.mcp.setup import McpSetup
    from src.engine.permissions.setup import PermissionSetup
    from src.engine.provider.base import SwitchableProvider
    from src.engine.provider.models import ModelCatalog, ModelInfo
    from src.engine.provider.vision import VisionCheck
    from src.engine.session import Recorder, SessionStore
    from src.engine.skills.loader import Skills
    from src.engine.terminal.manager import TerminalManager


@dataclass(frozen=True)
class StartChoice:
    """How a run begins: a new session, the latest one, the picker, or a session by id prefix."""

    kind: Literal["new", "continue", "pick", "resume"]
    session_id: str | None = None


NEW = StartChoice("new")


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
    catalog: ModelCatalog
    vision: VisionCheck  # shared by the app, headless mode and the agents: one answer per model
    model_notices: list[str]
    warnings: tuple[str, ...]
    close: Callable[[], Awaitable[None]]

    def kill_all(self) -> None:
        """Synchronous fallback once the loop is gone: whatever `close` did not stop is killed."""
        self.hook_runner.kill_all()  # nor a hook (a no-op once `close` ran)
        if self.checkpoints is not None:
            self.checkpoints.close()  # idempotent: the shadow folder is gone after the first call
        self.mcp.kill_all()  # a server still running means `close` never ran, and its loop is gone
        if self.terminals is not None and self.terminals.names():  # none when `close` ran
            self.terminals.kill_all()

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
) -> Runtime:
    """Read the settings files once and wire the run for `workspace` (a folder).

    `model` (`--model`) is the model for this launch only: it beats `AGENT_MODEL` and the project's `model` key.
    `builtins` are the slash commands a custom command may not shadow. With `terminals=False` there is no
    terminal manager and no `term_*` tool. `provider` replaces the Ollama one and skips the launch model lookup
    (for tests), and so does `connector` for the MCP servers. `trust_project_mcp` treats the project's MCP file as
    trusted for this run (nothing is remembered). `checkpoints=True` (the app) snapshots the workspace around each turn for `/undo`."""
    from src.client import OllamaClient, SearxClient, WebClient
    from src.client.mcp import SdkConnector
    from src.client.ollama import OllamaCatalog, OllamaUnreachable
    from src.config.settings import get_settings
    from src.engine import Agent
    from src.engine.agents.loader import load_agents
    from src.engine.checkpoints.setup import prepare_checkpoints
    from src.engine.commands.loader import load_commands
    from src.engine.guardrails.setup import prepare as prepare_guardrails
    from src.engine.hooks.runner import ProcessRunner
    from src.engine.hooks.setup import prepare as prepare_hooks
    from src.engine.mcp.config import global_config_path
    from src.engine.mcp.hub import McpHub
    from src.engine.mcp.setup import prepare
    from src.engine.mcp.trust import TrustStore, trust_store_path
    from src.engine.permissions import setup as permission_setup
    from src.engine.permissions.config import global_settings_path, project_settings_path
    from src.engine.provider.models import SHOW_TIMEOUT, SLOW_AFTER, CatalogError, launch_model, shown_name
    from src.engine.provider.ollama import OllamaProvider
    from src.engine.provider.vision import VisionCheck
    from src.engine.session import FileRecorder, SessionStore, sessions_root, workspace_dir
    from src.engine.skills.loader import load_skills
    from src.engine.terminal.manager import TerminalManager
    from src.engine.tools import default_tools, read_only_tools
    from src.engine.tools.base import ToolContext

    settings = get_settings()
    root = workspace.resolve()
    client = OllamaClient(settings.ollama_url)
    sessions = sessions_root(settings.sessions_dir)
    store = SessionStore(sessions, root)
    web = WebClient(settings.fetch_timeout, settings.fetch_max_bytes)
    search: SearxClient | None = None
    try:
        search = SearxClient(settings.search_url, settings.fetch_timeout) if settings.search_url else None
        # Both config files are read once, here; the project's servers stay untrusted until the user says yes.
        setup = prepare(root, global_config_path(), TrustStore(trust_store_path()))
        # Same for the permission rules: global ones are active; the project's allow rules wait for the user's trust.
        skills = load_skills(root)  # read once, like the commands
        agents = load_agents(root)  # same; `model:` names are checked at launch in Phase 3
        permissions = permission_setup.prepare(
            root,
            global_settings_path(),
            TrustStore(trust_store_path(name="permission-trust.json")),
            # to warn about rules that never apply; the manager is only asked for its tools' names, it starts nothing
            permission_setup.tool_keys(default_tools(TerminalManager(root), fetcher=web, searcher=search, skills=skills)),
        )
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
        # Guardrails come from the same two settings files, already parsed: protected paths, the child environment, the sandbox.
        guard = prepare_guardrails(
            root,
            permissions.global_data,
            permissions.project_data,
            os.environ,
            platform.system(),
            global_path=global_settings_path(),
            project_path=project_settings_path(root),
            sessions=sessions,  # the sandbox never lets a command write it
        )
        # Hooks too: both files already parsed; project hooks stay off until the same trust as the allow rules.
        hook_runner = ProcessRunner(root, guard)
        hooks = prepare_hooks(permissions, root, hook_runner)
        shadow, shadow_warnings = prepare_checkpoints(
            root, guard, guard.env.filter(os.environ), enabled=checkpoints
        )
        run_catalog = catalog if catalog is not None else OllamaCatalog(client)
        vision = VisionCheck(run_catalog)  # the uncached catalog: the check keeps its own answers
        ctx = ToolContext(root, settings.bash_timeout, guard=guard)
        # lives for the whole run; /new and session switches do not touch it
        manager = TerminalManager(root, guard=guard) if terminals else None
        commands = load_commands(root, builtins)  # read once, like the settings files
        mcp = McpHub(  # nothing starts yet
            connector if connector is not None else SdkConnector(guard=guard),
            root,
            setup.servers,
            settings.mcp_timeout,
            setup.warnings,
            start_timeout=settings.mcp_start_timeout,
            untrusted=frozenset() if trust_project_mcp else setup.untrusted,
        )

        def recorder_factory() -> Recorder:
            return FileRecorder(workspace_dir(sessions, root), root)  # lazy: no file until the first message

        def agent_factory(approver: Approver) -> Agent:
            return Agent(
                run_provider,
                default_tools(manager, fetcher=web, searcher=search, skills=skills),
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
            )

        async def close() -> None:
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
                            await client.aclose()

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
            catalog=run_catalog,
            vision=vision,
            model_notices=model_notices,
            warnings=(*permissions.warnings, *agents.warnings, *model_warnings, *guard.warnings, *hooks.warnings, *shadow_warnings),
            close=close,
        )
    except BaseException:
        _abort(web, search)  # setup failed before anyone could close them (no loop is running here)
        raise
