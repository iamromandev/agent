from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="AGENT_", extra="ignore"
    )

    api_url: Annotated[str, Field(default="http://localhost:8000", description="Base URL of the agent API")]
    api_timeout: Annotated[float, Field(default=10.0, description="Seconds before an API request gives up")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
