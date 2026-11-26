## core
# variables
# This is the only makefile and it owns no logic. Each target hands off to moon, which runs the project tasks defined
# in apps/*/moon.yml and infra/moon.yml (apps/ui's call bun). The one exception is ui-down, which only kills a port.
# Variables given on the command line (TAG=..., ENV=..., ARGS=...) reach the tasks through the environment.
# Tool versions (moon, uv) are in .prototools: run `proto install` once.
MOON := moon
UI_PORT := 3000

# phony targets
.PHONY: check test setup down restart clean help \
	api-check api-check-ci api-test api-run api-up api-down api-build api-build-prod api-push-prod api-restart api-ps api-logs \
	api-migrate api-install api-install-dev api-export api-clean-all api-clean-volumes api-clean-host \
	cli-sync cli-install cli-check cli-check-ci cli-test cli-run cli-eval cli-image-eval cli-uninstall cli-clean \
	ui-install ui-dev ui-down ui-restart ui-build ui-check ui-format \
	infra-init infra-validate infra-plan infra-apply infra-fmt

## both stacks
check: # Lint + typecheck every stack
	$(MOON) run api:check cli:check ui:check

test: # Run the API and CLI test suites
	$(MOON) run api:test cli:test

setup: # Run API migrations (containers must be up)
	$(MOON) run api:migrate

down: api-down ui-down # Stop both stacks: remove the API containers, kill the UI dev server

restart: # Restart both stacks: rebuild + restart the API containers, then relaunch the UI dev server
	$(MOON) run api:restart
	$(MOON) run ui:restart

clean: # Remove the project's volumes (db)
	$(MOON) run api:clean-volumes

## api — delegates to apps/api/moon.yml
api-check: # Lint + typecheck the API
	$(MOON) run api:check

api-check-ci: # Lint + typecheck the API without fixing anything (what CI runs)
	$(MOON) run api:check-ci

api-test: # Run the API test suite
	$(MOON) run api:test

api-run: # Run the API dev server on the host
	$(MOON) run api:run

api-up: # Start the API and database containers
	$(MOON) run api:up

api-down: # Remove the API containers
	$(MOON) run api:down

api-build: # Build the API Docker images
	$(MOON) run api:build

api-build-prod: # Build the prod API image (TAG=<repository>:<tag> optional)
	$(MOON) run api:build-prod

api-push-prod: # Build and push the prod API image to ECR (TAG=<ecr_repository_url>:<tag> required)
	$(MOON) run api:push-prod

api-restart: # Stop, rebuild, and start the API containers
	$(MOON) run api:restart

api-ps: # List the API containers
	$(MOON) run api:ps

api-logs: # Follow the API container logs
	$(MOON) run api:logs

api-migrate: # Run database migrations
	$(MOON) run api:migrate


api-install: # Install API dependencies
	$(MOON) run api:install

api-install-dev: # Add the API development dependencies (ruff, ty)
	$(MOON) run api:install-dev

api-export: # Export requirements.txt
	$(MOON) run api:export

api-clean-all: # Stop containers, remove volumes and images
	$(MOON) run api:clean-all

api-clean-volumes: # Remove the project's volumes (db)
	$(MOON) run api:clean-volumes

api-clean-host: # Prune all unused Docker data on the host
	$(MOON) run api:clean-host

## cli — delegates to apps/cli/moon.yml
cli-sync: # Install the CLI dependencies into apps/cli/.venv
	$(MOON) run cli:sync

cli-install: # Install the agent command on your PATH (with the image and media extras on Apple Silicon)
	$(MOON) run cli:install

cli-check: # Lint + typecheck the CLI
	$(MOON) run cli:check

cli-check-ci: # Lint + typecheck the CLI without fixing anything (what CI runs)
	$(MOON) run cli:check-ci

cli-test: # Run the CLI test suite
	$(MOON) run cli:test

cli-run: # Run the CLI, e.g. make cli-run ARGS="health check"
	$(MOON) run cli:run -- $(ARGS)

cli-eval: # Run the real-model smoke test (needs Ollama), e.g. make cli-eval EVAL_RUNS=1
	$(MOON) run cli:eval

cli-image-eval: # Run the real image-model smoke test (Apple Silicon + image extra; IMAGE_EVAL_CASES, IMAGE_EVAL_DOWNLOAD, IMAGE_EVAL_KEEP)
	$(MOON) run cli:image-eval

cli-uninstall: # Remove the installed agent command
	$(MOON) run cli:uninstall

cli-clean: # Remove the CLI's Python caches
	$(MOON) run cli:clean

## ui — delegates to apps/ui/package.json
ui-install: # Install UI dependencies
	$(MOON) run ui:install

ui-dev: # Run the UI dev server on 3000 (formats and typechecks first)
	$(MOON) run ui:dev

ui-down: # Stop the UI dev server (kill whatever listens on UI_PORT)
	@pids=$$(lsof -tiTCP:$(UI_PORT) -sTCP:LISTEN); [ -n "$$pids" ] && kill $$pids || true

ui-restart: # Wipe node_modules + bun.lock, reinstall, then run the dev server
	$(MOON) run ui:restart

ui-build: # Build the UI for production
	$(MOON) run ui:build

ui-check: # Typecheck the UI
	$(MOON) run ui:check

ui-format: # Format the UI sources
	$(MOON) run ui:format

## infra — delegates to infra/moon.yml (OpenTofu in Docker; needs AWS credentials)
infra-init: # Init the backend: make infra-init ENV=<env> BUCKET=<state bucket> (REGION optional)
	$(MOON) run infra:init

infra-validate: # Validate the configuration
	$(MOON) run infra:validate

infra-plan: # Plan: make infra-plan ENV=<env> IMAGE_TAG=<tag>
	$(MOON) run infra:plan

infra-apply: # Apply: make infra-apply ENV=<env> IMAGE_TAG=<tag>
	$(MOON) run infra:apply

infra-fmt: # Format the OpenTofu files
	$(MOON) run infra:fmt

# help
help:
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?#"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
