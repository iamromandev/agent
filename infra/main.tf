locals {
  name = "exateks-${var.env}-agent"

  # Plain settings. Anything sensitive is in secret.tf instead.
  environment = merge(
    {
      ENV        = var.env
      DEBUG      = "false"
      LOG_FORMAT = "json"
      DB_ENGINE  = "tortoise.backends.asyncpg"
      DB_HOST    = module.db.address
      DB_PORT    = tostring(module.db.port)
      DB_NAME    = module.db.db_name
      DB_USER    = module.db.username
      # The load balancer sits inside the VPC, so its X-Forwarded-For is the one to believe.
      TRUSTED_PROXIES = local.network.vpc_cidr_block
    },
    var.app_environment,
  )

  secrets = merge(
    # The database password is a field of the JSON secret RDS keeps.
    { DB_PASSWORD = "${module.db.master_user_secret_arn}:password::" },
    { for name, secret in aws_secretsmanager_secret.app : name => secret.arn },
  )
}

# NOTE: ref=dev follows a moving branch. Once exateks/infra tags a release, pin to it.
module "ecr" {
  source = "git::https://github.com/exateks/infra.git//module/ecr?ref=dev"

  name = "exateks/${var.env}/agent"
}

module "db" {
  source = "git::https://github.com/exateks/infra.git//module/db?ref=dev"

  name                       = local.name
  vpc_id                     = local.network.vpc_id
  subnet_ids                 = local.network.isolated_subnet_ids
  allowed_security_group_ids = [module.ecs.security_group_id]

  db_name  = "agent"
  username = "agent"

  instance_class      = var.db_instance_class
  multi_az            = var.db_multi_az
  deletion_protection = var.db_deletion_protection
  # A throwaway environment can be torn down without a final snapshot.
  skip_final_snapshot = var.env == "dev"
}

module "ecs" {
  source = "git::https://github.com/exateks/infra.git//module/ecs?ref=dev"

  name        = local.name
  cluster_arn = local.shared.cluster_arn
  vpc_id      = local.network.vpc_id
  subnet_ids  = local.network.private_subnet_ids

  image          = "${module.ecr.repository_url}:${var.image_tag}"
  container_port = 8000
  # Same start-up as the compose file, minus --reload: migrate, then serve.
  command       = ["bash", "-c", "python -m scripts.migrate && uvicorn src.main:app --host 0.0.0.0 --port 8000 --loop uvloop"]
  cpu           = var.cpu
  memory        = var.memory
  desired_count = var.desired_count

  environment = local.environment
  secrets     = local.secrets

  listener_arn          = local.shared.listener_arn
  alb_security_group_id = local.shared.alb_security_group_id
  host_headers          = var.hostnames
  listener_priority     = var.listener_priority

  health_check_path          = "/health/check"
  health_check_grace_seconds = var.health_check_grace_seconds

  service_discovery_namespace_id = local.shared.service_discovery_namespace_id
  service_discovery_name         = "agent"
}
