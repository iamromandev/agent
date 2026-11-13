"""Copies of this repo's root makefile and CI workflow, so no test scans the real repo."""

from __future__ import annotations

ROOT_MAKEFILE = r"""## core
# variables
# This file owns no logic. Each target hands off to the stack that does:
# api/makefile for Python and Docker, cli/makefile for the CLI, ui/package.json for the SPA.
API := api
CLI := cli
UI := ui
BUN := bun
UI_PORT := 3000

# phony targets
.PHONY: check test setup down restart clean help \
	api-check api-test api-run api-up api-down api-build api-build-prod api-push-prod api-restart api-ps api-logs \
	api-migrate api-install api-export api-clean-all api-clean-volumes api-clean-host \
	cli-install cli-check cli-test cli-run \
	ui-install ui-dev ui-down ui-restart ui-build ui-check ui-format

## both stacks
check: api-check cli-check ui-check # Lint + typecheck every stack

test: api-test cli-test # Run the API and CLI test suites

setup: api-migrate # Run API migrations (containers must be up)

down: api-down ui-down # Stop both stacks: remove the API containers, kill the UI dev server

restart: api-restart ui-restart # Restart both stacks: rebuild + restart the API containers, then relaunch the UI dev server

clean: api-clean-volumes # Remove the project's volumes (db)

## api — delegates to api/makefile
api-check: # Lint + typecheck the API
	$(MAKE) -C $(API) check

api-test: # Run the API test suite
	$(MAKE) -C $(API) test

api-run: # Run the API dev server on the host
	$(MAKE) -C $(API) run

api-up: # Start the API and database containers
	$(MAKE) -C $(API) up

api-down: # Remove the API containers
	$(MAKE) -C $(API) down

api-build: # Build the API Docker images
	$(MAKE) -C $(API) build

api-build-prod: # Build the prod API image (TAG=<repository>:<tag> optional)
	$(MAKE) -C $(API) build-prod

api-push-prod: # Build and push the prod API image to ECR (TAG=<ecr_repository_url>:<tag> required)
	$(MAKE) -C $(API) push-prod

api-restart: # Stop, rebuild, and start the API containers
	$(MAKE) -C $(API) restart

api-ps: # List the API containers
	$(MAKE) -C $(API) ps

api-logs: # Follow the API container logs
	$(MAKE) -C $(API) logs

api-migrate: # Run database migrations
	$(MAKE) -C $(API) migrate


api-install: # Install API dependencies
	$(MAKE) -C $(API) install

api-export: # Export requirements.txt
	$(MAKE) -C $(API) export

api-clean-all: # Stop containers, remove volumes and images
	$(MAKE) -C $(API) clean-all

api-clean-volumes: # Remove the project's volumes (db)
	$(MAKE) -C $(API) clean-volumes

api-clean-host: # Prune all unused Docker data on the host
	$(MAKE) -C $(API) clean-host

## cli — delegates to cli/makefile
cli-install: # Install CLI dependencies
	$(MAKE) -C $(CLI) install

cli-check: # Lint + typecheck the CLI
	$(MAKE) -C $(CLI) check

cli-test: # Run the CLI test suite
	$(MAKE) -C $(CLI) test

cli-run: # Run the CLI, e.g. make cli-run ARGS="health check"
	$(MAKE) -C $(CLI) run ARGS="$(ARGS)"

## ui — delegates to ui/package.json
ui-install: # Install UI dependencies
	$(BUN) install --cwd $(UI)

ui-dev: # Run the UI dev server on 3000 (formats and typechecks first)
	$(BUN) run --cwd $(UI) web:dev

ui-down: # Stop the UI dev server (kill whatever listens on UI_PORT)
	@pids=$$(lsof -tiTCP:$(UI_PORT) -sTCP:LISTEN); [ -n "$$pids" ] && kill $$pids || true

ui-restart: # Wipe node_modules + bun.lock, reinstall, then run the dev server
	$(BUN) run --cwd $(UI) web:restart

ui-build: # Build the UI for production
	$(BUN) run --cwd $(UI) web:build

ui-check: # Typecheck the UI
	$(BUN) run --cwd $(UI) web:check

ui-format: # Format the UI sources
	$(BUN) run --cwd $(UI) web:format

# help
help:
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?#"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
"""

CI_YML = r"""# Runs what the pre-commit hooks run, on the assumption that nobody ran them:
# a hook is a local courtesy that any clone can skip.
name: CI

on:
  pull_request:
  push:
    branches: [dev]

# A second push to the same branch makes the first run's answer worthless.
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  api:
    name: api — lint, types, tests
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: api
    steps:
      - uses: actions/checkout@v5

      # Keyed on the pyproject, not on the lockfile: ``**/*.lock`` is gitignored,
      # so ``uv sync`` resolves afresh here. ``requires-python = "==3.15.0"``
      # still pins the interpreter.
      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
          cache-dependency-glob: api/pyproject.toml

      - name: Install dependencies
        run: uv sync

      # ``check-ci`` rather than ``check``: ``ruff check --fix`` exits 0 once it
      # has fixed what it found, so it would pass a branch that does not lint.
      - name: Lint and type-check
        run: make check-ci

      # ``Settings`` is built at import time and its fields have no defaults, so
      # a clean checkout needs these throwaway values. Nothing connects to them.
      - name: Tests
        run: make test
        env:
          ENV: local
          DEBUG: "false"
          LOG_FORMAT: text
          DB_ENGINE: tortoise.backends.asyncpg
          DB_HOST: localhost
          DB_PORT: "5432"
          DB_NAME: ci
          DB_USER: ci
          DB_PASSWORD: ci

  cli:
    name: cli — lint, types, tests
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: cli
    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
          cache-dependency-glob: cli/pyproject.toml

      - name: Install dependencies
        run: uv sync

      - name: Lint and type-check
        run: make check-ci

      - name: Tests
        run: make test

  ui:
    name: ui — formatting, types, tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - name: Install bun
        uses: oven-sh/setup-bun@v2

      # No --frozen-lockfile: ``ui/bun.lock`` is gitignored. Direct dependencies
      # are exactly pinned in package.json; their transitives are not.
      - name: Install dependencies
        run: bun install --cwd ui

      - name: Formatting
        run: bun run --cwd ui web:format.check

      - name: Types
        run: bun run --cwd ui web:check

      - name: Tests
        run: bun run --cwd ui web:test
"""


def assert_block_shape(text: str) -> None:
    """Every line of a facts block is the intro, a blank, a section heading or a `- ` entry."""
    import re

    from src.agent.facts import INTRO

    if not text:
        return
    heading = re.compile(r"^(Project|Make targets|Package scripts|Layout|CI \(\.github/workflows/[^\n]*\))$")
    for line in text.split("\n\n", 1)[0].splitlines():
        assert line == INTRO or not line, line
    for line in text.splitlines():
        assert line == "" or line in INTRO or line.startswith("- ") or heading.match(line), repr(line)
