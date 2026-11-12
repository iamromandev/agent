## core
# variables
UV := uv
UVX := $(UV)x

# phony targets
.PHONY: install check check-ci test run help

# python tools (uv)
install: # Install all dependencies
	$(UV) sync

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

run: # Run the CLI, e.g. make run ARGS="health check"
	$(UV) run agent $(ARGS)

# help
help:
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?#"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
