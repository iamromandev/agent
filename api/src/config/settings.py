from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.data.type import Env


class Settings(BaseSettings):
    # ``hide_input_in_errors``: a validation error must not echo what it rejected, or a secret is printed
    # into the startup log whole.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )
    # core
    env: Annotated[Env, Field(description="Application environment")]
    debug: Annotated[bool, Field(description="Enable debug mode")]
    log_format: Annotated[str, Field(default="text", description="text | json")]
    # db
    db_engine: Annotated[
        str,
        Field(default="tortoise.backends.asyncpg", description="Tortoise database engine backend"),
    ]
    db_host: Annotated[str, Field(description="Database host")]
    db_port: Annotated[int, Field(description="Database port")]
    db_name: Annotated[str, Field(description="Database name")]
    db_user: Annotated[str, Field(description="Database user")]
    db_password: Annotated[SecretStr, Field(description="Database password")]
    # cors
    cors_origins: Annotated[
        str, Field(default="", description="Comma-separated origins allowed to call the API from a browser")
    ]
    # proxies
    # Comma-separated IPs or CIDRs whose X-Forwarded-For is believed; empty
    # believes none beyond uvicorn's own 127.0.0.1 default. Set it to the load
    # balancer's address in any deployment behind one.
    trusted_proxies: Annotated[
        str, Field(default="", description="IPs/CIDRs of proxies whose X-Forwarded-For is trusted")
    ]

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def trusted_proxy_list(self) -> list[str]:
        return [host.strip() for host in self.trusted_proxies.split(",") if host.strip()]

    @property
    def is_local(self) -> bool:
        return self.env == Env.LOCAL


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
