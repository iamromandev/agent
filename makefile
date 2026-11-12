## core
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
