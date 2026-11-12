# OpenTofu in Docker. Usage: make init ENV=dev, make plan ENV=dev IMAGE_TAG=<tag>
# Needs AWS credentials in the environment (for example from `aws sso login`) and BUCKET=<state bucket> for init.
ENV       ?= dev
REGION    ?= ap-southeast-1
BUCKET    ?= $(error set BUCKET=<state bucket>, the state_bucket_name output of exateks/infra stack/bootstrap)
IMAGE_TAG ?= unset
TOFU       = docker run --rm -it -v "$(CURDIR)":/work -w /work -e AWS_PROFILE -e AWS_REGION -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN -v $(HOME)/.aws:/root/.aws:ro ghcr.io/opentofu/opentofu:1.13.1

.PHONY: init plan apply fmt validate

init:
	$(TOFU) init -reconfigure -backend-config="bucket=$(BUCKET)" -backend-config="key=$(ENV)/agent.tfstate" -backend-config="region=$(REGION)"

validate:
	$(TOFU) validate

plan:
	$(TOFU) plan -var-file=env/$(ENV).tfvars -var image_tag=$(IMAGE_TAG)

apply:
	$(TOFU) apply -var-file=env/$(ENV).tfvars -var image_tag=$(IMAGE_TAG)

fmt:
	$(TOFU) fmt -recursive
