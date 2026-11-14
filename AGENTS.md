# AGENTS.md

Agent: placeholder project (purpose TBD) — FastAPI API (Python 3.15) + Typer CLI + React SPA + OpenTofu infra. Same architecture as `exateks/auth`. `api/`, `cli/`, `ui/` and `infra/` are self-contained; the root `makefile` only delegates.

## Commands (run from the repo root)

| Command | What it does |
| --- | --- |
| `make check` | API and CLI `ruff check --fix` + `ty check`, then UI `tsc`. **Fixes Python files**; does not check UI formatting. |
| `make -C api check-ci` / `make -C cli check-ci` | Lint + types without fixes — what CI runs. Use this before committing. |
| `make test` | API and CLI pytest suites. |
| `uv run --directory api pytest tests/path/test_file.py::test_name -q` | One API test. |
| `bun run --cwd ui web:test` | UI tests (`bun test test/`). |
| `bun run --cwd ui web:format` / `web:format.check` | Prettier write / check. |
| `make api-up` / `make api-migrate` | Start Postgres + API containers / run migrations in the container. |
| `make cli-run ARGS="health check"` | Run the CLI. |
| `make cli-run` (bare `agent`) | Open the terminal coding agent (Textual TUI); needs Ollama at `AGENT_OLLAMA_URL`. Its workspace is the directory it runs in: `cli/` for `make cli-run` and `cd cli && uv run agent`. For another folder run `uv run --project <repo>/cli agent` from it (`--project` keeps the cwd; `--directory` changes it). |
| `make cli-tool-install` / `make cli-tool-uninstall` | Install `agent` on the PATH as an editable uv tool (`agent-cli`) / remove it. Rerun the install after dependency changes or moving the checkout. |
| `uv run --directory api pre-commit install` | Install hooks once; mirrors CI (`ruff check`, `ty`, `prettier --check`). |

- Prefer `make api-up` over raw `docker compose up`: the compose project name is pinned to `agent`.
- Do not run `ruff format`: the formatter rewrites valid `except (A, B)` into invalid Python. `make`/CI/pre-commit use `ruff check` only.

## Setup / test prerequisites

- Python is pinned to exactly `3.15.0`; use `uv` for all Python tooling. Bun handles `ui/`.
- Copy `api/.env.example` to `api/.env` before tests or dev: `Settings` is instantiated at import, so even hermetic tests fail without it.
- Compose publishes Postgres on **5410** and the API on **8000**.
- `.env`, `uv.lock` and `bun.lock` are gitignored, so CI resolves fresh dependencies; direct deps are pinned in `pyproject.toml` / `package.json`.

## Architecture that is not obvious from filenames

- API entrypoint is `src.main:app`. Routers mount at the root (`/health/check`); there is no `/api` prefix. The Vite dev proxy owns `/api/*` and strips `/api` on the way out.
- Layering is `route/` → `service/` → `data/repo/`; Pydantic schemas live in `data/schema/`, enums in `data/type/`, domain logic in `lib/`, FastAPI dependencies in `deps/`. API tests mirror `src/`.
- Services are provided by `get_<name>_service` factories in `src/service/__init__.py`; repos by `get_<name>_database_repo` in `src/data/repo/__init__.py`, with interfaces in `<domain>/interface/` and implementations in `<domain>/<name>_db.py`.
- Every response is an envelope: `Success.ok(data=...).to_resp()` (`src/core/success.py`) or a raised `Error` (`src/core/error.py`), rendered by `init_global_errors`.
- Models subclass `IdBase` → `LinkBase` → `Base` in `src/core/base.py`; repos subclass `BaseRepo`; services `BaseService`; schemas `BaseSchema`.
- The CLI keeps the same shape: commands in `cli/src/command/`, and only `cli/src/client/` makes HTTP calls (`api.py` for the API, `ollama.py` for Ollama). Bare `agent` opens the coding agent: `cli/src/agent/` is the UI-free core (loop, tools, provider contract; no `textual`/`httpx` imports, enforced by `cli/tests/test_boundaries.py`) and `cli/src/tui/` is the Textual app on top of it. Conversations are saved under `~/.local/share/agent/sessions/` (`AGENT_SESSIONS_DIR`, or `$XDG_DATA_HOME/agent/sessions`; never in the workspace), one JSONL file per session, kept without limit: `agent --continue` resumes the latest for the folder, `agent --resume [id-prefix]` picks one or resumes by id, `/sessions` switches inside the app, `/new` starts a new session, `/compact` shrinks the context (also automatic at `AGENT_COMPACT_AT` of the window), `/undo` steps back one turn at a time in this run (restores what `write`/`edit` changed, deletes files the turn created, skips files changed since; bash and terminal effects (`term_start`/`term_stop`) are not undone, only counted; the prompt returns to the input box; compaction, `/new`, `/clear` and session switches discard the undo history), and `/clear` is now the same as `/new` (it used to only empty the screen). `cli/src/agent/session/` is the storage; every test runs with its own sessions folder. Instructions come from every `AGENTS.md` between the git root and the launch folder, limited to `AGENT_INSTRUCTIONS_SHARE` of the window, and `/init` has the model write one for the launch folder.
- The TUI has a terminal pane: `ctrl+t` shows or hides it (the first time it starts a `shell` tab running `$SHELL` in the workspace) and moves focus between the terminal and the chat input; `ctrl+\` also returns to the chat, from inside full-screen apps too. `ctrl+pageup`/`ctrl+pagedown` switch tabs (clicking a tab works too), and `ctrl+w` closes the tab of an exited terminal, or any tab while the tab bar has focus (stopping it first); while a running terminal has focus, `ctrl+w` and every other key (Tab, Esc, `ctrl+c`) go to its process, except `ctrl+t`, `ctrl+\`, `ctrl+pageup` and `ctrl+pagedown`, which the app takes even then. Some terminals (macOS Terminal.app) do not send `ctrl+pageup`/`ctrl+pagedown`, so there tabs are switched with the mouse only. Each terminal is a pty in its own process group: `cli/src/agent/terminal/` (UI-free: pty, manager, bounded plain-text scrollback) owns them, `cli/src/tui/widgets/terminal.py` draws them with `pyte` (only `cli/src/tui/` imports it) within a per-tick time budget, so a flood of output cannot stall the app. The model gets three tools: `term_start(name, command)` and `term_stop(name)` (both need approval and are refused in plan mode) and `term_read(name, lines?)` (the last plain-text lines and the status). Terminals belong to the run: `/new`, `/clear` and session switches keep them, and every process group is stopped (SIGTERM, then SIGKILL) however the app quits. A terminal the model starts while the pane is hidden is announced in the transcript.
- `web_fetch` (needs approval, allowed in plan mode) lives in `cli/src/agent/tools/web.py`; its HTTP goes through `cli/src/client/web.py`, which implements the `Fetcher` contract in `cli/src/agent/web.py` (limits: `AGENT_FETCH_TIMEOUT`, `AGENT_FETCH_MAX_BYTES`; output is cut so header, body and note fit the tool output limit). "Always" on `web_fetch` covers only that exact URL (`Tool.grant_arg`), like bash per command. The client follows redirects itself (max 5, 3xx bodies never read, one overall deadline) and refuses a hop from https to http, or to `localhost`/loopback/link-local/private literal addresses unless the first URL was one; hostnames are not resolved, so a public name pointing at a private address is not caught.
- MCP client (stdio servers; phase 1 reads only the global config): `~/.config/agent/mcp.json` (or `$XDG_CONFIG_HOME/agent/mcp.json`) is `{"servers": {"<name>": {"command", "args"?, "env"?}}}`; a missing file means no servers and a bad file or entry is a transcript warning, never an error. `cli/src/agent/mcp/` is UI-free and SDK-free (`config.py`, `names.py`, `contract.py`, `hub.py`, `tool.py`); only `cli/src/client/mcp.py` imports the official `mcp` package (enforced by `cli/tests/test_boundaries.py`) and implements the `McpConnector` contract with one owner task per server (anyio scopes must be entered and left by the same task). `McpHub` is created once per run in `cli/src/tui/__init__.py` (like `TerminalManager`): `AgentApp` starts it after the session is drawn, so the input is usable while servers start, and turns its events into notices. Each tool of a ready server is offered as `mcp__<server>__<tool>` (characters outside `[A-Za-z0-9_-]` become `_`; a later duplicate is skipped with a warning); `Agent(extra_tools=hub.tools)` asks for the current MCP tools at every request, so they appear when a server becomes ready and disappear when it exits. Every call needs approval ("Always" covers that tool name only, and is revoked when that tool is no longer offered, e.g. its server exited) and `mutating = True` hides and refuses MCP tools in plan mode; `uncancellable = False` keeps Esc able to interrupt a running call (write/edit calls still run to the end). Non-text results become placeholders (`[image omitted]`), an error result stays an error, output is cut at the tool output limit, and a call gives up after `AGENT_MCP_TIMEOUT` seconds (default 60) with the server left running. A server that fails to start, does not finish starting (connect and tool list) within `AGENT_MCP_START_TIMEOUT` (default 120; `AGENT_MCP_TIMEOUT` is for calls only), or exits is a warning and its tools are withdrawn (no restart). The warning for a timed-out start can arrive about 2 s after the deadline: the SDK first closes the server's stdin and waits that long before it kills it. Servers run in their own process group with the agent's environment plus the entry's `env`; when a server ends or is closed its whole group is SIGKILLed so helper processes do not outlive it. The app closes the hub first on quit, `/new` and session switches keep it, and `hub.kill_all()` (connector-level, so also servers still starting) is the synchronous fallback after `app.run()`. It relies on a private SDK function wrapped in `cli/src/client/mcp.py`; a test fails if an SDK update removes it.
- `/init` gives the model facts scanned from makefiles, `pyproject.toml`, `package.json` and CI, then warns about `make` and package-script commands in the written `AGENTS.md` that the scan did not find. The scan is static (it runs nothing) and stays inside the launch folder.

## Database

- Tortoise ORM + asyncpg on PostgreSQL. Migrations live in `api/src/data/db/migration/` (`0001_initial.py` is empty); `generate_schemas` is off, and `make api-migrate` runs `python -m scripts.migrate` in the container.
- Register every model in `api/src/data/db/model/__init__.py` and document it in `api/docs/model.md`.

## UI conventions

- Only `src/lib/api/*` calls `fetch`.
- No ESLint: formatting is Prettier (`.prettierrc.json`, with the Tailwind plugin) and typing is `tsc --noEmit`. Use the `@/*` aliases, not relative paths across directories.

## Infra

- `infra/` is OpenTofu in Docker: `make -C infra init ENV=<env> BUCKET=<state-bucket>`, then `plan` / `apply` with `IMAGE_TAG=<tag>`. It reuses the `exateks/infra` modules and stacks. The `prod` API image is `linux/arm64`.
