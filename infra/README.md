# agent/infra

OpenTofu for running agent on AWS: its image repository, database, secrets and ECS service. The VPC,
cluster, load balancer and service discovery namespace are not here; they are built from
[`exateks/infra`](https://github.com/exateks/infra) (`stack/network` and `stack/shared`), and this stack reads
what they publish.

## What it creates

| File | Resources |
|---|---|
| `main.tf` | ECR repository, RDS database, ECS service, built from `exateks/infra` modules (`ecr`, `db`, `ecs`) |
| `secret.tf` | Empty Secrets Manager secrets, one per name in `secret_names` (none yet) |
| `data.tf` | AWS provider, and the network and shared stacks' state |
| `env/*.tfvars` | Per-environment values: hostnames, sizes, database settings |

## Before the first deploy

1. Apply `exateks/infra` `stack/network`, then `stack/shared`, for the environment.
2. Fill in `hostnames` in `env/<env>.tfvars`. They are placeholders.
3. `make init ENV=<env> BUCKET=<state bucket>`, then `make apply ENV=<env> IMAGE_TAG=unset` with
   `-var desired_count=0` so the service waits for an image.
4. Put a value in each secret from `secret_arns`, if any.
5. Build and push the image to `ecr_repository_url` with
   `make api-push-prod TAG=<ecr_repository_url>:<tag>` from the repo root, then apply with the real `IMAGE_TAG`.

## Not verified

- The `exateks/infra` modules are referenced with `?ref=dev`, a moving branch. Pin a tag once one exists.
- `listener_priority` defaults to 110 so it does not collide with auth's 100 on a shared listener.
- Nothing here has been applied.
