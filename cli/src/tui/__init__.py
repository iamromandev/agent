"""The full-screen terminal UI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class StartChoice:
    """How the app begins: a new session, the latest one, the picker, or a session by id prefix."""

    kind: Literal["new", "continue", "pick", "resume"]
    session_id: str | None = None


NEW = StartChoice("new")


def run(workspace: Path, start: StartChoice = NEW) -> int:
    """Open the coding agent for `workspace` (a folder) and block until the user quits; returns the exit code."""
    from src.client import OllamaClient, SearxClient, WebClient
    from src.client.mcp import SdkConnector
    from src.config.settings import get_settings
    from src.engine import Agent
    from src.engine.approval import Approver
    from src.engine.commands.loader import load_commands
    from src.engine.mcp.config import global_config_path
    from src.engine.mcp.hub import McpHub
    from src.engine.mcp.setup import prepare
    from src.engine.mcp.trust import TrustStore, trust_store_path
    from src.engine.permissions import setup as permission_setup
    from src.engine.permissions.config import global_settings_path
    from src.engine.provider.ollama import OllamaProvider
    from src.engine.session import FileRecorder, Recorder, SessionStore, sessions_root, workspace_dir
    from src.engine.skills.loader import load_skills
    from src.engine.terminal.manager import TerminalManager
    from src.engine.tools import default_tools, read_only_tools
    from src.engine.tools.base import ToolContext
    from src.tui.app import BUILTINS, AgentApp

    settings = get_settings()
    root = workspace.resolve()
    client = OllamaClient(settings.ollama_url)
    provider = OllamaProvider(client, settings.model, settings.num_ctx)
    ctx = ToolContext(root, settings.bash_timeout)
    sessions = sessions_root(settings.sessions_dir)
    store = SessionStore(sessions, root)
    web = WebClient(settings.fetch_timeout, settings.fetch_max_bytes)
    search: SearxClient | None = None
    try:
        search = SearxClient(settings.search_url, settings.fetch_timeout) if settings.search_url else None
        terminals = TerminalManager(root)  # lives for the whole run; /new and session switches do not touch it
        # Both config files are read once, here; the project's servers stay untrusted until the user says yes.
        setup = prepare(root, global_config_path(), TrustStore(trust_store_path()))
        # Same for the permission rules: global ones are active; the project's allow rules wait for the user's trust.
        skills = load_skills(root)  # read once, like the commands
        permissions = permission_setup.prepare(
            root,
            global_settings_path(),
            TrustStore(trust_store_path(name="permission-trust.json")),
            permission_setup.tool_keys(default_tools(terminals, fetcher=web, searcher=search, skills=skills)),  # to warn about rules that never apply
        )
        commands = load_commands(root, BUILTINS)  # read once, like the settings files
        mcp = McpHub(  # nothing starts yet
            SdkConnector(),
            root,
            setup.servers,
            settings.mcp_timeout,
            setup.warnings,
            start_timeout=settings.mcp_start_timeout,
            untrusted=setup.untrusted,
        )

        def recorder_factory() -> Recorder:
            return FileRecorder(workspace_dir(sessions, root), root)  # lazy: no file until the first message

        def agent_factory(approver: Approver) -> Agent:
            return Agent(
                provider,
                default_tools(terminals, fetcher=web, searcher=search, skills=skills),
                approver,
                ctx,
                settings.max_steps,
                recorder=recorder_factory(),
                compact_at=settings.compact_at,
                compact_keep_turns=settings.compact_keep_turns,
                instructions_share=settings.instructions_share,
                extra_tools=mcp.tools,
                task_tools=read_only_tools(web, search, skills),
                task_max_steps=settings.task_max_steps,
                rules=permissions.rules,
            )

        async def close() -> None:
            try:
                await mcp.close_all()  # no server process outlives the agent
            finally:
                try:
                    await terminals.close_all()  # nor any terminal process
                finally:
                    try:
                        await web.aclose()
                    finally:
                        try:
                            if search is not None:
                                await search.aclose()
                        finally:
                            await client.aclose()

        # Closed by the app on unmount, inside the loop that owns the connections and the terminals' readers.
        app = AgentApp(
            agent_factory,
            provider,
            root,
            store=store,
            start=start,
            recorder_factory=recorder_factory,
            on_close=close,
            terminals=terminals,
            mcp=mcp,
            mcp_trust=setup.remember,
            permissions=permissions.rules,
            permission_warnings=permissions.warnings,
            commands=commands,
            skills=skills,
        )
    except BaseException:
        try:  # setup failed before the app could close them (no loop is running here)
            asyncio.run(web.aclose())
        finally:
            if search is not None:
                asyncio.run(search.aclose())
        raise
    try:
        app.run()
    finally:
        mcp.kill_all()  # a server still running means the app failed before it could close it, and its loop is gone
        if terminals.names():  # none when the app closed them; else it failed before it could, and its loop is gone
            terminals.kill_all()
    return app.return_code or 0
