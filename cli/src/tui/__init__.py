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
    from src.agent import Agent
    from src.agent.approval import Approver
    from src.agent.provider.ollama import OllamaProvider
    from src.agent.session import FileRecorder, Recorder, SessionStore, sessions_root, workspace_dir
    from src.agent.terminal.manager import TerminalManager
    from src.agent.tools import default_tools
    from src.agent.tools.base import ToolContext
    from src.client import OllamaClient, WebClient
    from src.config.settings import get_settings
    from src.tui.app import AgentApp

    settings = get_settings()
    root = workspace.resolve()
    client = OllamaClient(settings.ollama_url)
    provider = OllamaProvider(client, settings.model, settings.num_ctx)
    ctx = ToolContext(root, settings.bash_timeout)
    sessions = sessions_root(settings.sessions_dir)
    store = SessionStore(sessions, root)
    web = WebClient(settings.fetch_timeout, settings.fetch_max_bytes)
    try:
        terminals = TerminalManager(root)  # lives for the whole run; /new and session switches do not touch it

        def recorder_factory() -> Recorder:
            return FileRecorder(workspace_dir(sessions, root), root)  # lazy: no file until the first message

        def agent_factory(approver: Approver) -> Agent:
            return Agent(
                provider,
                default_tools(terminals, fetcher=web),
                approver,
                ctx,
                settings.max_steps,
                recorder=recorder_factory(),
                compact_at=settings.compact_at,
                compact_keep_turns=settings.compact_keep_turns,
                instructions_share=settings.instructions_share,
            )

        async def close() -> None:
            try:
                await terminals.close_all()  # no terminal process outlives the agent
            finally:
                try:
                    await web.aclose()
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
        )
    except BaseException:
        asyncio.run(web.aclose())  # setup failed before the app could close it (no loop is running here)
        raise
    try:
        app.run()
    finally:
        if terminals.names():  # none when the app closed them; else it failed before it could, and its loop is gone
            terminals.kill_all()
    return app.return_code or 0
