# Containers for agent's secrets. Only the empty secrets are created here; the values are set by
# hand (console or CLI), so they never appear in code or in state. The service will not start
# until every secret below has a value.
locals {
  secret_names = toset([
    # Add a name per secret the API reads, e.g. "OPENAI_API_KEY".
  ])
}

resource "aws_secretsmanager_secret" "app" {
  for_each = local.secret_names

  name        = "exateks/${var.env}/agent/${each.key}"
  description = "Agent ${each.key} for ${var.env}"
}
