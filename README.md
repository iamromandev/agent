# Agent

Placeholder project: purpose to be defined. The skeleton follows the same architecture as `exateks/auth`:
a FastAPI API, a Python CLI, a React SPA and OpenTofu infra, each self-contained, with a root `makefile`
that only delegates.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.15.0 | API and CLI runtime |
| proto | latest | Installs moon and uv at the versions in `.prototools` (`proto install`) |
| moon | 2.6.6 | Task runner; the root `makefile` delegates to it |
| uv | 0.13.10 | Python package manager |
| Docker | latest | PostgreSQL database, API container, OpenTofu |
| Bun | latest | UI package manager & runtime |

## Repository structure

```
agent/
├── apps/
│   ├── api/                # Python FastAPI backend
│   │   ├── src/
│   │   │   ├── core/       # base model/repo/schema/service, error + success envelopes, enums
│   │   │   ├── config/     # Settings (pydantic-settings), logging
│   │   │   ├── route/      # HTTP routers            ─┐
│   │   │   ├── service/    # business services        │ route → service → data/repo
│   │   │   ├── data/       # db, model, repo, schema, type, seed
│   │   │   ├── lib/        # domain logic
│   │   │   ├── deps/       # FastAPI dependencies
│   │   │   ├── client/     # outbound clients
│   │   │   ├── middleware/
│   │   │   └── main.py     # src.main:app
│   │   ├── scripts/        # migrate, lint
│   │   ├── tests/          # mirrors src/
│   │   ├── dockerfile
│   │   ├── docker-compose.yml
│   │   └── pyproject.toml
│   ├── cli/                # Python Typer CLI (`agent`) that talks to the API
│   │   ├── src/            # main.py, command/, client/, config/
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── ui/                 # TypeScript React frontend (Bun workspace)
│       └── apps/web/       # Vite + React Router SPA
├── infra/                  # OpenTofu (AWS ECS + RDS) in Docker
├── docs/                   # Design documents
└── makefile                # the only makefile; delegates to moon
```

## API

```bash
cp apps/api/.env.example apps/api/.env
make api-up    # Postgres + API on http://localhost:8000; migrations run on boot
make api-logs
```

Postgres is published on host port **5410**. `make api-run` serves on the host instead (set `DB_HOST=localhost`
and `DB_PORT=5410` in `.env`). Check it with:

```bash
curl http://localhost:8000/health/check
```

`make api-check` (ruff + ty, fixes), `make api-check-ci` (no fixes), `make api-test` (pytest).

## CLI

`agent` is a terminal coding agent (Textual TUI) that runs on a local Ollama model. It also has a headless mode (`-p`) for scripts and `agent health check` for the API. Every command below runs from the repo root unless a step says `cd`.

### 1. Prerequisites

| Tool | Why | Install (macOS) |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | Python tooling; fetches Python 3.15.0 if missing | `brew install uv` |
| [Ollama](https://ollama.com) | Runs the model | `brew install ollama` |
| git 2.20+ | `/undo` and `/diff` checkpoints (without it, `/undo` only restores `write`/`edit` changes) | `xcode-select --install` or `brew install git` |
| Docker | Only for `agent health check` (it needs the API) | Docker Desktop |

### 2. Start Ollama and pull the model

```bash
brew services start ollama
```

```bash
ollama pull qwen2.5:7b-instruct
```

```bash
ollama list
```

`ollama serve` runs it in the foreground instead of `brew services`. The default model is `qwen2.5:7b-instruct`. For another one, pull it and set `AGENT_MODEL` (step 3) or pass `--model`.

### 3. Install the dependencies and configure

```bash
make cli-sync
```

```bash
cp apps/cli/.env.example apps/cli/.env
```

`make cli-sync` runs `uv sync` in `apps/cli/` (with `--extra image` on Apple Silicon, for the `generate_image` tool). The `.env` file is optional: every value has a default. `.env` is read from the folder the agent is launched in, so for other folders export the variables in `~/.zshrc` or put a `.env` there. Main settings (all in [apps/cli/.env.example](apps/cli/.env.example)):

| Variable | Default | What it sets |
| --- | --- | --- |
| `AGENT_OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `AGENT_MODEL` | `qwen2.5:7b-instruct` | Model (`--model` and `/model` override it) |
| `AGENT_NUM_CTX` | `8192` | Context window, in tokens |
| `AGENT_MAX_STEPS` | `25` | Model calls per turn |
| `AGENT_BASH_TIMEOUT` | `120` | Seconds before a `bash` call is killed |
| `AGENT_COMPACT_AT` | `0.8` | Share of the window at which the history is compacted |
| `AGENT_SESSIONS_DIR` | `~/.local/share/agent/sessions` | Where sessions are saved |
| `AGENT_MCP_TIMEOUT` | `60` | Seconds before an MCP tool call gives up |
| `AGENT_MCP_START_TIMEOUT` | `120` | Seconds an MCP server may take to start |
| `AGENT_SEARCH_URL` | unset | SearXNG URL; turns on `web_search` (step 9) |
| `AGENT_API_URL` | `http://localhost:8000` | API for `agent health check` |

### 4. Check the setup

```bash
make cli-run ARGS="--version"
```

```bash
make cli-run ARGS='-p "say hello" < /dev/null'
```

The second command makes one real model call and prints the reply. The first call can take a minute or more while Ollama loads the model. It exits 1 if Ollama is not reachable or the model is not installed. To check the API too:

```bash
make api-up
```

```bash
make cli-run ARGS="health check"
```

### 5. Run the agent

```bash
make cli-run
```

The workspace is the folder the agent runs in: `apps/cli/` for `make cli-run` (same as `cd apps/cli && uv run agent`). To work on another project, either install the command (step 6) or run it from that project's folder without installing:

```bash
cd /path/to/your/project && uv run --project /path/to/agent/apps/cli agent
```

Use `--project`, not `--directory`: `--project` keeps the current folder as the workspace, while `--directory` switches to `apps/cli/`.

### 6. Install `agent` on your PATH (optional)

```bash
make cli-install
```

```bash
cd ~/projects/other-app && agent
```

- This is an editable install of `apps/cli/` as the uv tool `agent-cli`, so `git pull` is enough to upgrade. On Apple Silicon it includes the `image` extra (mflux). Rerun `make cli-install` after dependency changes in `apps/cli/pyproject.toml` or after moving the checkout.
- `agent` not found: run `uv tool update-shell` and open a new shell. `command -v agent` shows which `agent` runs.
- Uninstall: `make cli-uninstall`.

### 7. Launch options

| Command | What it does |
| --- | --- |
| `agent` | New session in the current folder |
| `agent --continue` | Resume the latest session for this folder |
| `agent --resume` | Pick a session of this folder from a list |
| `agent --resume <id-prefix>` | Resume one session by id |
| `agent --model <name>` | Use another installed Ollama model for this launch |
| `agent --version` | Print the version |
| `agent health check [--json]` | Check the API |

Sessions are saved per folder under `~/.local/share/agent/sessions/`, never inside the project.

### 8. Inside the app

Type a message and press Enter. The agent asks before it runs a command or changes a file: allow once, Always (this run) or Always here (saves the rule to `.agent/settings.json`).

| Slash command | What it does |
| --- | --- |
| `/help` | List commands, including custom ones |
| `/new`, `/clear` | Start a new session |
| `/sessions` | Switch to another session |
| `/undo` | Undo the last turn (files and `bash`/terminal changes in the workspace); repeat to go further back |
| `/diff` | Show what the last turn changed |
| `/compact` | Shrink the context (also automatic) |
| `/usage` | Token totals, model calls and speed for the session |
| `/model [name]` | Pick or switch the model (saved to `.agent/settings.json`) |
| `/init` | Have the model write `AGENTS.md` for this folder |
| `/permissions`, `/guardrails`, `/hooks`, `/mcp`, `/skills` | Show the loaded rules, sandbox, hooks, MCP servers and skills |
| `/quit` | Exit |

| Key or input | What it does |
| --- | --- |
| `Tab` | Cycle `BUILD → EDITS → PLAN` (accept-edits mode, shown as `EDITS`, runs `write`/`edit` inside the workspace without asking, except `.agent/settings.json`, `.agent/mcp.json`, `.agent/commands`, `.agent/skills`, anything under `.git/` and the agent's own config, data and sessions folders, which still ask (names compared case-insensitively); plan mode hides `write`, `edit`, terminals and MCP tools, and skips allow rules) |
| `Esc` | Interrupt the running turn |
| `ctrl+c` | Interrupt, or quit when idle |
| `@path` / `@"path with spaces"` | Attach a workspace file to the message (`@` suggests paths) |
| `/` | Suggest commands (Up/Down, Tab or Enter to complete) |
| `ctrl+t` | Open the terminal pane / move focus between it and the chat / hide it |
| `ctrl+\` | Back to the chat from the terminal |
| `ctrl+pageup`, `ctrl+pagedown` | Switch terminal tabs (not sent by macOS Terminal.app: use the mouse) |
| `ctrl+w` | Close an exited terminal tab |

The model can use the terminal pane too: `term_start` and `term_stop` (both need approval) start and stop long-running processes, and `term_read` reads their output. Every terminal is stopped when the agent quits.

### 9. Configure (optional)

Files are read once at launch. A bad file is a warning, never a crash. Global files are in `~/.config/agent/` (or `$XDG_CONFIG_HOME/agent/`), project files in the launch folder's `.agent/`. Project allow rules, hooks and MCP servers (`.agent/settings.json`, `.agent/mcp.json`) stay off until you trust that exact file content in a prompt at launch. The choice is remembered per folder in `~/.local/share/agent/permission-trust.json` and `mcp-trust.json`, so editing the file asks again.

| File | What it holds |
| --- | --- |
| `settings.json` | `permissions` (allow/deny rules), `guardrails` (sandbox, protected paths), `hooks`, `model` |
| `mcp.json` | stdio MCP servers; their tools appear as `mcp__<server>__<tool>`, every call needs approval and is refused in plan mode |
| `commands/<name>.md` | Custom slash command `/name` (`$ARGUMENTS` is the text after it) |
| `skills/<name>/SKILL.md` | A skill the model can load with the `skill` tool |
| `AGENTS.md` (in the project) | Instructions loaded from the git root down to the launch folder |

`~/.config/agent/settings.json`:

```json
{
  "permissions": {
    "allow": ["bash(make test)", "bash(git status*)", "web_fetch(https://docs.python.org/*)", "mcp__notes__*"],
    "deny": ["bash(rm -rf*)"]
  },
  "guardrails": { "sandbox": { "network": true, "write": ["~/go"] }, "keepEnv": ["GITHUB_TOKEN"] },
  "hooks": { "PostToolUse": [{ "matcher": "edit", "command": "make test", "timeout": 60 }] }
}
```

- Rules: `tool`, `tool(exact value)`, `tool(prefix*)` or `mcp__<server>__*`. Deny wins over allow and also applies in plan mode. A prefix allow rule only matches plain commands (no quotes, `;`, `|`, `$(` and the like).
- Guardrails: the agent's file tools never touch protected paths (`~/.ssh`, `.env`, `*.pem`, ...). On macOS, `bash`, terminals, hooks and MCP servers run in a sandbox that can write only to the workspace, temp and cache folders. Secret-looking variables (`*TOKEN*`, `*_KEY`, ...) are removed from their environment unless listed in `keepEnv`. Tool output is redacted for secrets.
- Hooks run at `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop` and `SessionStart`. Exit 2 blocks (where the event allows it).

`~/.config/agent/mcp.json`:

```json
{ "servers": { "notes": { "command": "uvx", "args": ["notes-mcp"], "env": { "KEY": "value" } } } }
```

`~/.config/agent/commands/review.md`:

```markdown
---
description: Review the staged diff
---
Review it. Focus on: $ARGUMENTS
```

Web search (optional) uses your own SearXNG:

```bash
docker run -d -p 8888:8080 -v "$PWD/searxng:/etc/searxng" searxng/searxng
```

Add `json` to `search.formats` in `searxng/settings.yml` (`formats: [html, json]`, otherwise SearXNG answers 403), restart the container, then set `AGENT_SEARCH_URL=http://localhost:8888`.

### 10. Headless mode

One turn without the UI. The final reply goes to stdout; tool lines and warnings go to stderr.

```bash
agent -p "summarise the README"
```

```bash
git diff | agent -p "review this diff"
```

```bash
agent -p "run the tests and fix failures" --allow 'bash(make test)' --allow edit
```

```bash
agent -p "list the TODOs" --output json < /dev/null
```

| Option (with `-p`) | What it does |
| --- | --- |
| `--allow '<rule>'` | Allow a call for this run (repeatable). Anything else that needs approval is refused |
| `--output text\|json\|stream-json` | Plain reply, one JSON result, or one JSON line per event |
| `--quiet` | Only warnings and errors on stderr |
| `--continue`, `--resume <id>`, `--model <name>` | As in the app |
| `--trust-project` | Trust the project's settings and `mcp.json` for this run (runs its hooks and servers) |

- `-p` with no prompt reads the prompt from stdin. A piped stdin is added below the prompt, so use `< /dev/null` when there is none.
- Exit codes: `0` done, `1` model or hook error or step limit, `2` usage error, `130`/`143` interrupted.

### 11. Develop and test

| Command | What it does |
| --- | --- |
| `make cli-check` | `ruff check --fix` + `ty check` (fixes files) |
| `make cli-check-ci` | The same without fixes, as CI runs it |
| `make cli-test` | pytest suite |
| `uv run --directory apps/cli pytest tests/path/test_file.py::test_name -q` | One test |
| `make cli-eval` | Real-model smoke test (needs Ollama) |

`make cli-eval` runs four fixed tasks (`find`, `rename`, `rules`, `command`) headlessly, each in a throwaway copy of `apps/cli/evals/fixtures/sample`. It prints pass counts and median steps, tokens and time, and writes `apps/cli/.eval/<UTC timestamp>.json`. Settings: `EVAL_RUNS` (default 3), `EVAL_TASKS=find,rename`, `EVAL_TIMEOUT` (seconds, default 300), `EVAL_KEEP=1` (keep the temp folders). Without Ollama it prints a skip line and exits 0. `bash` runs with your real environment, so the tasks allow only fixed commands. It is not part of `make test` or CI.

## UI

```bash
bun install --cwd apps/ui
bun run --cwd apps/ui web:dev   # http://localhost:3000, proxies /api/* to the API
```

## Infra

See [infra/README.md](infra/README.md).
