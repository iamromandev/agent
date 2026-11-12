from tortoise.fields.base import StrEnum


class Env(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    STAGE = "stage"
    PROD = "prod"
