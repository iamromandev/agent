## core
# variables
DOCKER_COMPOSE := docker compose -f docker-compose.yml -p agent
UV := uv
UVX := $(UV)x
SERVER_CONTAINER := server-agent-api
# Tag for the prod image; override with TAG=<repository>:<tag> (the ECR url is in the infra outputs).
TAG ?= agent-api:prod
# AWS region of the registry push-prod logs in to; must match the infra stack's region.
REGION ?= ap-southeast-1

# phony targets
.PHONY: clean-host clean-volumes clean-all ps build build-prod push-prod up stop down restart install install-dev check check-ci test run export add logs migrate help

## operation
# system cleanup
clean-host: # Prune all unused Docker data on the host
	docker system prune --all --force
	docker volume prune --all --force
	docker buildx prune --all --force
	docker builder prune --all --force

clean-volumes: # Remove this project's volumes (db)
	$(DOCKER_COMPOSE) down -v

clean-all: # Stop containers, remove volumes & images
	$(DOCKER_COMPOSE) down -v --rmi all

# docker control
ps: # List containers
	$(DOCKER_COMPOSE) ps -a

build: # Build Docker images
	$(DOCKER_COMPOSE) build

build-prod: # Build the prod image (arm64). TAG=<repository>:<tag> optional, default agent-api:prod
	docker buildx build --platform linux/arm64 --target prod \
		--build-arg ENV=prod --build-arg WORK_DIR=/workdir --build-arg INSTALL_DIR=/opt/install \
		--build-arg PW_DIR=/pwdir -t $(TAG) --load .

push-prod: # Build, then push the prod image to ECR. TAG=<ecr_repository_url>:<tag> required; needs AWS credentials
	@case "$(TAG)" in *.amazonaws.com/*:*) ;; *) echo "set TAG=<ecr_repository_url>:<tag> (the ecr_repository_url output of infra)"; exit 1;; esac
	$(MAKE) build-prod
	aws ecr get-login-password --region $(REGION) | docker login --username AWS --password-stdin $(firstword $(subst /, ,$(TAG)))
	docker push $(TAG)

up: # Start containers
	$(DOCKER_COMPOSE) up -d

stop: # Stop containers
	$(DOCKER_COMPOSE) stop

down: # Remove containers
	$(DOCKER_COMPOSE) down

restart: # Restart containers
	make stop
	make build
	make up

logs: # Show live logs
	$(DOCKER_COMPOSE) logs -f

# python tools (uv)
install: # Install all dependencies
	$(UV) sync

install-dev: # Add development dependencies
	$(UV) add --dev ruff ty

check: # Run lint + type checks
	make install
	$(UVX) ruff check --fix
	$(UV) run ty check

check-ci: # Run lint + type checks without fixing anything. For CI.
	make install
	$(UVX) ruff check
	$(UV) run ty check

test: # Run the unit test suite
	$(UV) run pytest

run: # Run FastAPI dev server
	make check
	$(UV) run uvicorn src.main:app --reload --loop uvloop

export: # Export requirements.txt
	$(UV) export --format requirements-txt --output requirements.txt

# git
add: # Stage all changes
	git add .

# database scripts
migrate: # Run database migrations
	docker exec $(SERVER_CONTAINER) python -m scripts.migrate

# help
help:
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?#"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
