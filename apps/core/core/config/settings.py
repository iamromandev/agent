from __future__ import annotations

from functools import lru_cache
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import Field, field_validator
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

    task_max_steps: Annotated[
        int, Field(default=15, gt=0, description="Maximum model calls of one task subagent")
    ]

    sessions_dir: Annotated[
        str | None, Field(default=None, description="Where saved sessions live; unset means the XDG data folder")
    ]
    compact_at: Annotated[
        float, Field(default=0.8, gt=0, le=1, description="Fraction of num_ctx at which the history is compacted")
    ]
    compact_keep_turns: Annotated[
        int, Field(default=2, ge=1, description="Most recent user turns that compaction keeps verbatim")
    ]
    instructions_share: Annotated[
        float, Field(default=0.25, gt=0, le=0.5, description="Fraction of num_ctx that AGENTS.md files may use")
    ]
    fetch_timeout: Annotated[
        float, Field(default=15.0, gt=0, description="Seconds before a web_fetch request gives up")
    ]
    fetch_max_bytes: Annotated[
        int, Field(default=2_000_000, gt=0, description="Most bytes web_fetch downloads before it stops")
    ]
    search_url: Annotated[
        str | None, Field(default=None, description="SearXNG base URL for web_search; unset hides the tool")
    ]
    image_model: Annotated[
        str,
        Field(
            default="filipstrand/Z-Image-Turbo-mflux-4bit",
            min_length=1,
            description="mflux model (Hugging Face repo or folder) for generate_image",
        ),
    ]
    image_edit_model: Annotated[
        str,
        Field(
            default="black-forest-labs/FLUX.2-klein-4B",
            min_length=1,
            description="mflux model (Hugging Face repo or folder) for edit_image",
        ),
    ]
    image_upscale_model: Annotated[
        str,
        Field(
            default="seedvr2-3b",
            min_length=1,
            description="mflux model (alias, Hugging Face repo or folder) for upscale_image",
        ),
    ]
    image_timeout: Annotated[
        float, Field(default=900.0, gt=0, description="Seconds before a generate_image or edit_image call is killed")
    ]
    speech_model: Annotated[
        str,
        Field(
            default="mlx-community/Kokoro-82M-bf16",
            min_length=1,
            description="mlx-audio model (Hugging Face repo or folder) for speak",
        ),
    ]
    speech_timeout: Annotated[
        float, Field(default=300.0, gt=0, description="Seconds before a speak call is killed")
    ]
    video_timeout: Annotated[
        float, Field(default=900.0, gt=0, description="Seconds before an assemble_video call is killed")
    ]
    mcp_timeout: Annotated[
        float, Field(default=60.0, gt=0, description="Seconds before an MCP tool call gives up")
    ]
    mcp_start_timeout: Annotated[
        float,
        Field(default=120.0, gt=0, description="Seconds an MCP server may take to start and list its tools"),
    ]
    lsp_wait: Annotated[
        float, Field(default=3.0, gt=0, description="Seconds a write/edit waits for a language server's diagnostics")
    ]
    lsp_start_wait: Annotated[
        float, Field(default=10.0, gt=0, description="Seconds the first write/edit waits for a language server to start")
    ]
    lsp_max_diagnostics: Annotated[
        int, Field(default=20, gt=0, description="Diagnostics shown per file and language server")
    ]
    daemon_idle: Annotated[
        float,
        Field(default=600.0, gt=0, description="Seconds the shared-session daemon stays up with no client and no turn"),
    ]

    @field_validator("search_url")
    @classmethod
    def _blank_search_url_is_unset(cls, value: str | None) -> str | None:
        value = value.strip() if value else ""
        if not value:
            return None
        try:
            url = urlsplit(value)
            host = url.hostname
        except ValueError as e:
            raise ValueError("AGENT_SEARCH_URL must be a valid http(s) URL") from e
        if url.scheme not in ("http", "https") or not host:
            raise ValueError("AGENT_SEARCH_URL must be an http(s) URL with a host")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
