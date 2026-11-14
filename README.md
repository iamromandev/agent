# Agent

Placeholder project: purpose to be defined. The skeleton follows the same architecture as `exateks/auth`:
a FastAPI API, a Python CLI, a React SPA and OpenTofu infra, each self-contained, with a root `makefile`
that only delegates.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.15.0 | API and CLI runtime |
| uv | latest | Python package manager |
| Docker | latest | PostgreSQL database, API container, OpenTofu |
| Bun | latest | UI package manager & runtime |

## Project Structure

```
agent/
├── api/                # Python FastAPI backend
│   ├── src/
│   │   ├── core/       # base model/repo/schema/service, error + success envelopes, enums
│   │   ├── config/     # Settings (pydantic-settings), logging
│   │   ├── route/      # HTTP routers            ─┐
│   │   ├── service/    # business services        │ route → service → data/repo
│   │   ├── data/       # db, model, repo, schema, type, seed
│   │   ├── lib/        # domain logic
│   │   ├── deps/       # FastAPI dependencies
│   │   ├── client/     # outbound clients
│   │   ├── middleware/
│   │   └── main.py     # src.main:app
│   ├── scripts/        # migrate, lint
│   ├── tests/          # mirrors src/
│   ├── dockerfile
│   ├── docker-compose.yml
│   ├── makefile
│   └── pyproject.toml
├── cli/                # Python Typer CLI (`agent`) that talks to the API
│   ├── src/            # main.py, command/, client/, config/
│   ├── tests/
│   ├── makefile
│   └── pyproject.toml
├── ui/                 # TypeScript React frontend (Bun workspace)
│   └── apps/web/       # Vite + React Router SPA
├── infra/              # OpenTofu (AWS ECS + RDS) in Docker
└── docs/               # Design documents
```

## API

```bash
cd api
cp .env.example .env
make up        # Postgres + API on http://localhost:8000; migrations run on boot
make logs
```

Postgres is published on host port **5410**. `make run` serves on the host instead (set `DB_HOST=localhost`
and `DB_PORT=5410` in `.env`). Check it with:

```bash
curl http://localhost:8000/health/check
```

`make check` (ruff + ty, fixes), `make check-ci` (no fixes), `make test` (pytest).

## CLI

```bash
cd cli
cp .env.example .env   # optional: AGENT_API_URL defaults to http://localhost:8000
uv sync
uv run agent health check
uv run agent           # coding agent TUI; its workspace is the current directory, here `cli/` (needs a local Ollama; see cli/.env.example)
uv run agent --continue # resume the latest saved session for this folder (also: --resume [id], /sessions inside the app)
# /undo steps back one turn of this run: it restores files the turn's write/edit calls changed, deletes ones it created, skips files changed since, and puts the prompt back in the input box (bash and terminal effects are not undone)
# ctrl+t opens the terminal pane (a `shell` tab in the workspace; again: back to the chat, again: hide it; ctrl+\ also goes back to the chat), ctrl+pageup / ctrl+pagedown switch its tabs, ctrl+w closes an exited terminal's tab (every other key goes to the terminal while it has focus, except ctrl+t, ctrl+\ and ctrl+pageup / ctrl+pagedown; macOS Terminal.app does not send ctrl+pageup / ctrl+pagedown, so there tabs are switched with the mouse); the model starts, reads and stops long-running processes there with term_start (needs approval), term_read and term_stop (needs approval); every terminal is stopped when the agent quits
# the agent loads every AGENTS.md from the git root down to the launch folder (up to AGENT_INSTRUCTIONS_SHARE of the context window); /init has the model write one from facts scanned (statically) from makefiles, pyproject.toml, package.json and CI, then warns about make and script commands the scan did not find
```

### Install on macOS

Install `agent` as a command on your PATH, so you can run it in any folder.

Prerequisites: [uv](https://docs.astral.sh/uv/) (`brew install uv`), Ollama running at `AGENT_OLLAMA_URL`, and Python 3.15.0 (uv fetches it if missing).

```bash
cd /path/to/agent
make cli-tool-install   # editable install of cli/ as the uv tool `agent-cli`
cd ~/projects/other-app
agent                   # workspace is the current folder
```

- If `agent` is not found, uv's tool bin folder is not on your PATH: run `uv tool update-shell` and open a new shell. `command -v agent` shows which `agent` runs (another `agent` earlier on the PATH would shadow this one).
- `.env` is read from the launch folder, so export `AGENT_*` variables in `~/.zshrc` or use that folder's `.env`.
- Upgrading: the install is editable, so `git pull` is enough. Rerun `make cli-tool-install` after dependency changes in `cli/pyproject.toml` or after moving the checkout (a moved or deleted checkout breaks `agent`).
- Uninstall: `make cli-tool-uninstall`.

### Open the agent in another project

The agent's workspace is the folder it is launched from. To work on another project, run it from that project's folder:

```bash
cd /path/to/your/project
agent             # new session in this project (installed, see above)
agent --continue  # resume this project's latest session

# without installing:
uv run --project /path/to/agent/cli agent
uv run --project /path/to/agent/cli agent --continue
```

- Without the install, use `--project`, not `--directory`: `--project` keeps the current folder as the workspace, `--directory` switches to `cli/`.
- Sessions are saved per folder (under `~/.local/share/agent/sessions/`, never inside the project), so `--continue` and `--resume` only see that project's sessions.
- The project's `AGENTS.md` files (from its git root down to the launch folder) are loaded; run `/init` to write one.
- `.env` is read from the current folder, so set `AGENT_*` variables (such as `AGENT_OLLAMA_URL`) in your shell or in that project's `.env`.
- To get a plain `agent` command, see [Install on macOS](#install-on-macos).

## UI

```bash
bun install --cwd ui
bun run --cwd ui web:dev   # http://localhost:3000, proxies /api/* to the API
```

## Infra

See [infra/README.md](infra/README.md).
