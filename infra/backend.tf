# State lives in the bucket stack/bootstrap in exateks/infra creates. Bucket, key and region are
# passed at init time (make init does it):
#   tofu init -backend-config="bucket=<state bucket>" \
#             -backend-config="key=<env>/agent.tfstate" \
#             -backend-config="region=<region>"
terraform {
  backend "s3" {
    encrypt      = true
    use_lockfile = true
  }
}
