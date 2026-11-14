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
# for another folder, run from it: uv run --project /path/to/agent/cli agent  (--project keeps the cwd, --directory changes it; `.env` is read from the cwd, so set AGENT_* in the environment there)
```

## UI

```bash
bun install --cwd ui
bun run --cwd ui web:dev   # http://localhost:3000, proxies /api/* to the API
```

## Infra

See [infra/README.md](infra/README.md).
