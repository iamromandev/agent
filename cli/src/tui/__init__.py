"""The full-screen terminal UI."""

from __future__ import annotations

from pathlib import Path


def run(workspace: Path) -> None:
    """Open the coding agent for `workspace` (a folder) and block until the user quits."""
    from src.agent import Agent
    from src.agent.approval import Approver
    from src.agent.provider.ollama import OllamaProvider
    from src.agent.tools import default_tools
    from src.agent.tools.base import ToolContext
    from src.client import OllamaClient
    from src.config.settings import get_settings
    from src.tui.app import AgentApp

    settings = get_settings()
    root = workspace.resolve()
    client = OllamaClient(settings.ollama_url)
    provider = OllamaProvider(client, settings.model, settings.num_ctx)
    ctx = ToolContext(root, settings.bash_timeout)

    def agent_factory(approver: Approver) -> Agent:
        return Agent(provider, default_tools(), approver, ctx, settings.max_steps)

    # The client is closed by the app on unmount, inside the loop that owns its connections.
    AgentApp(agent_factory, provider, root, on_close=client.aclose).run()
