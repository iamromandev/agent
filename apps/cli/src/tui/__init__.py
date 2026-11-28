"""The full-screen terminal UI."""

from __future__ import annotations

import sys
from pathlib import Path

from core.runtime import NEW, StartChoice, build_runtime

__all__ = ["NEW", "StartChoice", "run"]


def run(workspace: Path, start: StartChoice = NEW, model: str | None = None, shared: bool = False) -> int:
    """Open the coding agent for `workspace` (a folder) and block until the user quits; returns the exit code.

    `model` (`--model`) is the model for this launch only: it beats `AGENT_MODEL` and the project's `model` key.
    `shared` (`--shared`): the session lives in the folder's daemon (started when none runs), shared with every other
    `--shared` client of the folder; plain `agent` never touches the daemon."""
    if shared:
        return run_shared(workspace, start, model)
    from core.engine.mentions_index import FileIndex
    from core.host.port import LocalPort
    from core.host.session import SessionHost

    from src.tui.app import BUILTINS, AgentApp

    rt = build_runtime(workspace, BUILTINS, model=model, checkpoints=True, images=True)
    try:
        # The live session; the app drives it through the in-process port.
        host = SessionHost(
            rt.agent_factory,
            rt.provider,
            rt.root,
            window_cap=rt.settings.num_ctx,
            store=rt.store,
            recorder_factory=rt.recorder_factory,
            catalog=rt.catalog,
            vision=rt.vision,
            guard=rt.guard,
            permissions=rt.permissions.rules,
            hooks=rt.hooks,
        )
        # Closed by the app on unmount, inside the loop that owns the connections and the terminals' readers.
        app = AgentApp(
            LocalPort(host, FileIndex(rt.root, rt.guard)),
            start=start,
            on_close=rt.close,
            terminals=rt.terminals,
            mcp=rt.mcp,
            mcp_trust=rt.setup.remember,
            permission_warnings=rt.warnings,
            launch_notices=rt.model_notices,
            commands=rt.commands,
            skills=rt.skills,
            agents=rt.agents,
            workflows=rt.workflows,
            paste=rt.paste,
            images=rt.images,
            config_guard=rt.config_guard,
            held=rt.held,
            lsp=rt.lsp,
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


def run_shared(workspace: Path, start: StartChoice = NEW, model: str | None = None) -> int:
    """`agent --shared`: make sure the folder's daemon runs, then open the app on a `RemotePort` to it. No runtime is
    built here: terminals, workflows, image tools and pasted images are refused in this mode."""
    from core.server.paths import RunDirError
    from core.server.serve import ensure_daemon

    from src.tui.app import AgentApp
    from src.tui.remote import RemotePort

    ws = workspace.resolve()
    try:
        socket = ensure_daemon(ws, model)
    except (RunDirError, OSError, TimeoutError) as exc:
        print(f"agent --shared: {exc}", file=sys.stderr)
        return 1
    app = AgentApp(RemotePort(socket, ws), start=start, model=model)
    app.run()
    return app.return_code or 0
