"""The full-screen terminal UI."""

from __future__ import annotations

import sys
from pathlib import Path

from src.runtime import NEW, StartChoice, build_runtime

__all__ = ["NEW", "StartChoice", "run"]


def run(workspace: Path, start: StartChoice = NEW, model: str | None = None) -> int:
    """Open the coding agent for `workspace` (a folder) and block until the user quits; returns the exit code.

    `model` (`--model`) is the model for this launch only: it beats `AGENT_MODEL` and the project's `model` key."""
    from src.tui.app import BUILTINS, AgentApp

    rt = build_runtime(workspace, BUILTINS, model=model, checkpoints=True, images=True)
    try:
        # Closed by the app on unmount, inside the loop that owns the connections and the terminals' readers.
        app = AgentApp(
            rt.agent_factory,
            rt.provider,
            rt.root,
            store=rt.store,
            start=start,
            recorder_factory=rt.recorder_factory,
            on_close=rt.close,
            terminals=rt.terminals,
            mcp=rt.mcp,
            mcp_trust=rt.setup.remember,
            permissions=rt.permissions.rules,
            permission_warnings=rt.warnings,
            catalog=rt.catalog,
            vision=rt.vision,
            window_cap=rt.settings.num_ctx,
            launch_notices=rt.model_notices,
            commands=rt.commands,
            skills=rt.skills,
            agents=rt.agents,
            workflows=rt.workflows,
            guard=rt.guard,
            hooks=rt.hooks,
            paste=rt.paste,
            images=rt.images,
            config_guard=rt.config_guard,
            held=rt.held,
        )
    except BaseException:
        rt.abort()  # the app could not be built, so it can never close them
        raise
    try:
        app.run()
    finally:
        try:
            rt.kill_all()
        finally:
            # Config watch warnings the quit checks saw after the transcript was gone (already escaped).
            for text in rt.config_guard.undelivered():
                print(f"warning: {text}", file=sys.stderr)
    return app.return_code or 0
