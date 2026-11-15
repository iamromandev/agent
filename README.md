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
# MCP servers (stdio): list them in ~/.config/agent/mcp.json (or $XDG_CONFIG_HOME/agent/mcp.json) as {"servers": {"name": {"command": "uvx", "args": ["notes-mcp"], "env": {"KEY": "value"}}}}; each server starts in the background in the launch folder, its tools reach the model as mcp__<server>__<tool>, every call needs approval ("Always" covers that tool only) and is refused in plan mode, a call gives up after AGENT_MCP_TIMEOUT seconds (default 60), a server that has not started and listed its tools after AGENT_MCP_START_TIMEOUT seconds (default 120) is a warning, and every server is stopped when the agent quits; a broken file or server is a warning in the transcript, never a crash; a folder's own .agent/mcp.json (same format) adds project servers (one with the same name replaces the global one), which do not start until you trust them in a prompt that lists each name and full command line (the prompt escapes odd characters and notes that the files the commands run are not checked; the choice is remembered per folder under $XDG_DATA_HOME/agent/mcp-trust.json (not configurable from the folder) for that exact file content, so editing the file asks again; declining keeps them stopped for the run); /mcp lists every server with its source, status and, when ready, its tools
# permission rules: ~/.config/agent/settings.json (or $XDG_CONFIG_HOME/agent/settings.json) and the folder's .agent/settings.json hold {"permissions": {"allow": ["bash(make test)", "bash(git status*)", "web_fetch(https://docs.python.org/*)", "mcp__notes__*", "write"], "deny": ["bash(rm -rf*)"]}}; a rule is a tool name (any call), tool(value) (the tool's key argument equals value: the command for bash and term_start, the URL for web_fetch; read, write, edit, glob, grep, task, term_read, term_stop and MCP tools have no such argument, so only their bare name works and a valued rule for one is a warning at launch and never applies), tool(prefix*) (starts with prefix; end a web_fetch prefix with a / after the host) or mcp__<server>__* (an MCP server whose name contains "__" or ends with "_" once odd characters are replaced by "_" is skipped with a warning, so no server can pose as another); an allow rule with a prefix only matches a bash or term_start command made of letters, digits, spaces and _ . / : = @ % + , - (quotes, ; && || | ` $( ( ) { } > < & * ? ~ ! # \ and a newline send the call to the dialog, which also covers the zsh and fish forms of running code from an argument); deny rules are matched the other way round: a deny prefix is looked for after leading whitespace and at the start of every command in the line (after ; & | a newline, a backtick, $( or a parenthesis), so bash(rm*) also refuses "echo hi; rm x", and bash(*) or a bare bash refuses every command; this is best effort, not a sandbox: env rm, sudo rm, /bin/rm, FOO=1 rm, time rm or a script that runs rm pass a deny rule, and a prefix allow rule such as find* or git* also accepts dangerous flags (find . -delete), so prefer exact or narrow rules; a web_fetch prefix rule is checked against the URL the model asked for, so it also covers a redirect to another public host; deny beats allow from any file and refuses without a dialog (also tools that need no approval, also in plan mode), then allow rules run without asking (never in plan mode), then this launch's "Always", then the dialog; the dialog's "Always here: <rule>" saves exactly that rule to .agent/settings.json (not offered for a command that contains *, which a rule would read as a prefix); a write or edit of .agent/settings.json or .agent/mcp.json always shows the dialog (no rule or grant skips it) and offers neither Always nor Always here; a folder's allow rules do nothing until you trust that exact file content in a prompt at launch (the broadest rules are listed first; remembered per folder under $XDG_DATA_HOME/agent/permission-trust.json, not configurable from the folder; editing the file asks again; its deny rules always apply); files are read once at launch, must be regular files of at most 1 MiB, and a bad file, list or rule (also one nested too deeply) is a warning, never a crash; an allow list that is not a list of strings loses only the allow rules, a bad deny list only the deny rules; /permissions lists every rule with its source, the untrusted ones as inactive, and the project file's trust state; text from a file is shown with odd characters escaped
# custom slash commands: save a prompt as a Markdown file in ~/.config/agent/commands/ (or $XDG_CONFIG_HOME/agent/commands/) or in the folder's .agent/commands/ (a project command replaces a global one of the same name; a built-in name is skipped with a warning) and run it as /name args: `---\ndescription: Review the staged diff\n---\nReview it. Focus on: $ARGUMENTS`; $ARGUMENTS is replaced by the text after the name (or appended after a blank line when the body has none), the transcript shows the full prompt (escaped) under a /name (project|global) label, /help lists the commands, and typing / suggests names (Up/Down, Tab or Enter completes, Esc closes; Enter on a fully typed name sends it with no arguments); files are read once at launch, must be UTF-8 regular files of at most 64 KiB, and a bad one is a warning
# the model can hand an investigation to a read-only subagent with the task tool (read, glob, grep, plus web_fetch and web_search when set up; no approval needed, also in plan mode): it runs in its own fresh context, you see its tool calls folded under one block, and only its final report goes back to the model; it is not saved in the session, and AGENT_TASK_MAX_STEPS (default 15) caps its model calls
# web search: set AGENT_SEARCH_URL to your own SearXNG to give the model a web_search tool (query, limit 1-10, default 5; titles, URLs and snippets, shown to the model as untrusted data; no approval, also in plan mode; a deny rule `web_search` turns it off; it then reads a result with web_fetch). Run SearXNG with `docker run -d -p 8888:8080 -v "$PWD/searxng:/etc/searxng" searxng/searxng`, add `json` to the `search.formats` list in searxng/settings.yml (`formats: [html, json]`, else SearXNG answers 403) and restart it, then `AGENT_SEARCH_URL=http://localhost:8888`; the request uses AGENT_FETCH_TIMEOUT, redirects are not followed, and the tool is hidden while the variable is unset
# the model keeps a checklist for multi-step work with the todo tool (no approval, also in plan mode): the list shows in a panel above the input, is reminded to the model on every request, and survives --continue, /undo and compaction; it clears with /new
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

### Real-model smoke test

`make -C cli eval` (or `make cli-eval`) runs four fixed tasks (`find`, `rename`, `rules`, `command`) against the configured Ollama model, each in a throwaway copy of `cli/evals/fixtures/sample` with its own config and data folders. It prints pass counts and median steps, tokens and time per task, and saves a JSON report to `cli/.eval/<UTC timestamp>.json` (gitignored) to compare runs over time. Settings: the usual `AGENT_*` variables, plus `EVAL_RUNS` (default 3), `EVAL_TASKS` (comma-separated), `EVAL_TIMEOUT` (seconds per run, default 300) and `EVAL_KEEP=1` (keep the temp folders and print their paths). Without Ollama it prints a skip line and exits 0. Isolation covers the agent's config, data and sessions folders and the workspace copy only: `bash` commands run with your real environment and `HOME`, so every task allows just fixed commands (exact `python -m pytest` rules, everything else refused). It is not part of `make test` or CI.

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
