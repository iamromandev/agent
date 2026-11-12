provider "aws" {
  region = var.region

  default_tags {
    tags = {
      managed_by = "opentofu"
      repo       = "exateks/agent"
      app        = "agent"
      env        = var.env
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  state_bucket = coalesce(var.state_bucket, "exateks-tfstate-${data.aws_caller_identity.current.account_id}")
}

# Both stacks are built and applied from exateks/infra; this stack only reads what they publish.
data "terraform_remote_state" "network" {
  backend = "s3"

  config = {
    bucket = local.state_bucket
    key    = "${var.env}/network.tfstate"
    region = var.region
  }
}

data "terraform_remote_state" "shared" {
  backend = "s3"

  config = {
    bucket = local.state_bucket
    key    = "${var.env}/shared.tfstate"
    region = var.region
  }
}

locals {
  network = data.terraform_remote_state.network.outputs
  shared  = data.terraform_remote_state.shared.outputs
}
