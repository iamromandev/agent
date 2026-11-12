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

    ollama_url: Annotated[str, Field(default="http://localhost:11434", description="Base URL of the Ollama server")]
    model: Annotated[str, Field(default="qwen2.5:7b-instruct", description="Ollama model used by the agent")]
    num_ctx: Annotated[int, Field(default=8192, gt=0, description="Context window requested from Ollama, in tokens")]
    bash_timeout: Annotated[
        float, Field(default=120.0, gt=0, description="Seconds before a bash tool call is killed")
    ]
    max_steps: Annotated[int, Field(default=25, gt=0, description="Maximum model calls in one turn")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
