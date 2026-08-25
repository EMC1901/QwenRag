"""Configuration helpers for the model gateway."""

from functools import lru_cache
import os
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from qwenrag_runtime.paths import get_runtime_paths


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    gateway_host: str = Field(default="127.0.0.1", alias="GATEWAY_HOST")
    gateway_port: int = Field(default=8010, alias="GATEWAY_PORT")
    gateway_api_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["change-me"],
        alias="GATEWAY_API_KEYS",
    )
    gateway_allow_no_auth: bool = Field(
        default=False,
        alias="GATEWAY_ALLOW_NO_AUTH",
    )

    llm_base_url: str = Field(
        default="http://127.0.0.1:8001/v1",
        alias="LLM_BASE_URL",
    )
    llm_model: str = Field(default="qwen", alias="LLM_MODEL")
    llm_upstream_api_key: str = Field(default="", alias="LLM_UPSTREAM_API_KEY")

    embedding_base_url: str = Field(
        default="http://127.0.0.1:8002/v1",
        alias="EMBEDDING_BASE_URL",
    )
    embedding_model: str = Field(
        default="qwen3-embedding-0.6b",
        alias="EMBEDDING_MODEL",
    )
    embedding_upstream_api_key: str = Field(
        default="",
        alias="EMBEDDING_UPSTREAM_API_KEY",
    )

    http_connect_timeout_seconds: float = Field(
        default=5,
        alias="HTTP_CONNECT_TIMEOUT_SECONDS",
    )
    http_read_timeout_seconds: float = Field(
        default=180,
        alias="HTTP_READ_TIMEOUT_SECONDS",
    )
    http_write_timeout_seconds: float = Field(
        default=60,
        alias="HTTP_WRITE_TIMEOUT_SECONDS",
    )
    http_pool_timeout_seconds: float = Field(
        default=5,
        alias="HTTP_POOL_TIMEOUT_SECONDS",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_request_body: bool = Field(default=False, alias="LOG_REQUEST_BODY")

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("gateway_api_keys", mode="before")
    @classmethod
    def parse_gateway_api_keys(cls, value: Any) -> list[str]:
        """Parse comma-separated API keys from environment variables."""
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("GATEWAY_API_KEYS must be a comma-separated string or list")

    @field_validator("llm_base_url", "embedding_base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> str:
        """Normalize upstream base URLs so route joining is predictable."""
        if value is None:
            raise ValueError("base URL is required")
        normalized = str(value).strip().rstrip("/")
        if not normalized:
            raise ValueError("base URL cannot be empty")
        return normalized

    @field_validator("gateway_host", mode="before")
    @classmethod
    def validate_loopback_host(cls, value: Any) -> str:
        normalized = str(value).strip()
        if normalized not in {"127.0.0.1", "::1"}:
            raise ValueError("GATEWAY_HOST must be 127.0.0.1 or ::1")
        return normalized

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: Any) -> str:
        """Normalize log levels to uppercase names."""
        return str(value).strip().upper()

    @model_validator(mode="after")
    def validate_auth_configuration(self) -> "Settings":
        """Require at least one gateway API key unless auth is disabled."""
        if not self.gateway_allow_no_auth and not self.gateway_api_keys:
            raise ValueError(
                "GATEWAY_API_KEYS is required when GATEWAY_ALLOW_NO_AUTH=false"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached runtime settings."""
    if os.getenv("MODEL_GATEWAY_DISABLE_ENV_FILE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return Settings(_env_file=None)
    return Settings(_env_file=get_runtime_paths().gateway_env_file)


def reset_settings_cache() -> None:
    """Clear cached settings, mainly for tests after environment changes."""
    get_settings.cache_clear()
