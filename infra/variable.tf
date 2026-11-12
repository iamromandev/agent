variable "env" {
  description = "Environment name: dev, stage or prod. Selects which network and shared stacks this service joins."
  type        = string
}

variable "region" {
  description = "AWS region. Must match the network and shared stacks."
  type        = string
}

variable "state_bucket" {
  description = "Bucket holding the other stacks' state. Leave null to use exateks-tfstate-<account id>."
  type        = string
  default     = null
}

variable "hostnames" {
  description = "Public hostnames routed to agent, such as [\"agent.dev.example.com\"]."
  type        = list(string)
}

variable "image_tag" {
  description = "Tag of the agent image in this stack's ECR repository to run. Passed at deploy time, not kept in env/*.tfvars."
  type        = string
}

variable "desired_count" {
  description = "How many copies of the container to run. Use 0 for the first apply, before an image has been pushed."
  type        = number
  default     = 1
}

variable "cpu" {
  description = "Fargate CPU units for each container."
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate memory in MiB for each container."
  type        = number
  default     = 1024
}

variable "listener_priority" {
  description = "Priority of agent's host rule on the shared load balancer listener. Unique per listener, so platform uses another number."
  type        = number
  default     = 110
}

variable "health_check_grace_seconds" {
  description = "How long a new container may take to become healthy. The API runs its migrations before it starts serving."
  type        = number
  default     = 120
}

variable "db_instance_class" {
  description = "Database instance size."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_multi_az" {
  description = "Keep a standby database in another zone."
  type        = bool
  default     = false
}

variable "db_deletion_protection" {
  description = "Refuse to delete the database."
  type        = bool
  default     = false
}

variable "app_environment" {
  description = "Extra plain environment variables for agent, merged over the defaults in main.tf. Secrets do not go here."
  type        = map(string)
  default     = {}
}
