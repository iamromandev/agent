env       = "prod"
region    = "ap-southeast-1"      # assumed, not confirmed: must match exateks/infra
hostnames = ["agent.example.com"] # placeholder: the real public host for prod stage

desired_count = 2
cpu           = 1024
memory        = 2048

db_instance_class      = "db.t4g.small"
db_multi_az            = true
db_deletion_protection = true
