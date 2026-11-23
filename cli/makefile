## core
# variables
UV := uv
UVX := $(UV)x
# Apple Silicon Macs also get the `image` extra (mflux, for the generate_image tool) and the `media` extra (mlx-audio,
# for the speak tool); elsewhere they are not installed.
APPLE_SILICON := $(shell test "`uname -s`" = Darwin && test "`uname -m`" = arm64 && echo yes)
EXTRAS := $(if $(APPLE_SILICON),--extra image --extra media,)
TOOL_SPEC := $(if $(APPLE_SILICON),.[image,media],.)

# phony targets
.PHONY: sync install uninstall check check-ci test run eval image-eval tool-install tool-uninstall help

# python tools (uv)
sync: # Install the dependencies into cli/.venv (with the image extra on Apple Silicon)
	$(UV) sync $(EXTRAS)

check: # Run lint + type checks
	make sync
	$(UVX) ruff check --fix
	$(UV) run ty check

check-ci: # Run lint + type checks without fixing anything. For CI.
	make sync
	$(UVX) ruff check
	$(UV) run ty check

test: # Run the unit test suite
	$(UV) run pytest

run: # Run the CLI, e.g. make run ARGS="health check"
	$(UV) run agent $(ARGS)

eval: # Run the real-model smoke test (needs Ollama)
	$(UV) run python -m evals

image-eval: # Run the real image-model smoke test (Apple Silicon + image extra)
	$(UV) run python -m evals.images

install: # Install the agent command on your PATH (editable, from this checkout; image extra on Apple Silicon)
	$(UV) tool install --editable --force '$(TOOL_SPEC)'

tool-install: install # Same as install (older name)

uninstall: # Remove the installed agent command
	-$(UV) tool uninstall agent-cli

tool-uninstall: uninstall # Same as uninstall (older name)

# help
help:
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?#"}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
