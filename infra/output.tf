output "ecr_repository_url" {
  description = "Where CI pushes the agent image, without a tag."
  value       = module.ecr.repository_url
}

output "service_name" {
  description = "Name of the ECS service."
  value       = module.ecs.service_name
}

output "service_security_group_id" {
  description = "Security group of the agent containers."
  value       = module.ecs.security_group_id
}

output "secret_arns" {
  description = "Secrets to fill in by hand before the first start."
  value       = { for name, secret in aws_secretsmanager_secret.app : name => secret.arn }
}

output "db_secret_arn" {
  description = "Secret holding the generated database password."
  value       = module.db.master_user_secret_arn
}

output "internal_dns_name" {
  description = "Where other services reach agent from inside the VPC, on port 8000."
  value       = "agent.${local.shared.service_discovery_namespace_name}"
}
