"""Runtime configuration for the Windows-local RAG application."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env.local-rag"


class Settings(BaseSettings):
    """Settings loaded from environment variables and ``.env.local-rag``."""

    local_rag_host: str = Field(default="127.0.0.1", alias="LOCAL_RAG_HOST")
    local_rag_port: int = Field(
        default=18080,
        ge=1,
        le=65535,
        alias="LOCAL_RAG_PORT",
    )
    local_rag_model: str = Field(default="local-rag", alias="LOCAL_RAG_MODEL")
    local_rag_api_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["none"],
        alias="LOCAL_RAG_API_KEYS",
    )
    local_rag_allow_no_auth: bool = Field(
        default=False,
        alias="LOCAL_RAG_ALLOW_NO_AUTH",
    )

    local_rag_answer_mode: Literal["stub", "gateway"] = Field(
        default="stub",
        alias="LOCAL_RAG_ANSWER_MODE",
    )
    enable_rag_router: bool = Field(
        default=False,
        alias="ENABLE_RAG_ROUTER",
    )
    rag_router_max_tokens: int = Field(
        default=128,
        ge=32,
        le=256,
        alias="RAG_ROUTER_MAX_TOKENS",
    )

    model_gateway_base_url: str = Field(
        default="",
        alias="MODEL_GATEWAY_BASE_URL",
    )
    model_gateway_api_key: str = Field(
        default="",
        alias="MODEL_GATEWAY_API_KEY",
    )
    upstream_llm_model: str = Field(default="", alias="UPSTREAM_LLM_MODEL")

    http_connect_timeout_seconds: float = Field(
        default=5,
        gt=0,
        alias="HTTP_CONNECT_TIMEOUT_SECONDS",
    )
    http_read_timeout_seconds: float = Field(
        default=180,
        gt=0,
        alias="HTTP_READ_TIMEOUT_SECONDS",
    )
    http_write_timeout_seconds: float = Field(
        default=60,
        gt=0,
        alias="HTTP_WRITE_TIMEOUT_SECONDS",
    )
    http_pool_timeout_seconds: float = Field(
        default=5,
        gt=0,
        alias="HTTP_POOL_TIMEOUT_SECONDS",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_request_body: bool = Field(default=False, alias="LOG_REQUEST_BODY")

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("local_rag_host", mode="before")
    @classmethod
    def validate_loopback_host(cls, value: Any) -> str:
        """Allow only loopback listening addresses in the first release."""
        normalized = str(value).strip()
        if normalized not in {"127.0.0.1", "::1"}:
            raise ValueError(
                "LOCAL_RAG_HOST must be 127.0.0.1 or ::1 in the first release"
            )
        return normalized

    @field_validator("local_rag_model", "upstream_llm_model", mode="before")
    @classmethod
    def normalize_model_names(cls, value: Any) -> str:
        """Reject empty model names once they are required by the active mode."""
        return "" if value is None else str(value).strip()

    @field_validator("local_rag_api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, value: Any) -> list[str]:
        """Parse comma-separated local API keys without JSON decoding."""
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("LOCAL_RAG_API_KEYS must be a comma-separated string or list")

    @field_validator("model_gateway_base_url", mode="before")
    @classmethod
    def normalize_gateway_base_url(cls, value: Any) -> str:
        """Normalize the configured gateway base URL for stable route joining."""
        return "" if value is None else str(value).strip().rstrip("/")

    @field_validator("model_gateway_api_key", "log_level", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: Any, info) -> str:
        """Normalize simple text settings while preserving an empty gateway key."""
        normalized = "" if value is None else str(value).strip()
        return normalized.upper() if info.field_name == "log_level" else normalized

    @model_validator(mode="after")
    def validate_runtime_configuration(self) -> "Settings":
        """Reject unsafe auth settings and incomplete gateway mode configuration."""
        if not self.local_rag_model:
            raise ValueError("LOCAL_RAG_MODEL cannot be empty")
        if not self.local_rag_allow_no_auth and not self.local_rag_api_keys:
            raise ValueError(
                "LOCAL_RAG_API_KEYS is required when LOCAL_RAG_ALLOW_NO_AUTH=false"
            )
        if self.local_rag_answer_mode == "gateway":
            required = {
                "MODEL_GATEWAY_BASE_URL": self.model_gateway_base_url,
                "MODEL_GATEWAY_API_KEY": self.model_gateway_api_key,
                "UPSTREAM_LLM_MODEL": self.upstream_llm_model,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "gateway mode requires: " + ", ".join(missing)
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    Tests opt out of the developer's real ``.env.local-rag`` by setting the
    dedicated environment switch. Runtime processes never set this switch.
    """
    if os.getenv("LOCAL_RAG_DISABLE_ENV_FILE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return Settings(_env_file=None)
    return Settings()


def reset_settings_cache() -> None:
    """Clear cached settings; this is used by tests after environment changes."""
    get_settings.cache_clear()
