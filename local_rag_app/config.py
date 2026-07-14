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
    enable_local_retrieval: bool = Field(
        default=False,
        alias="ENABLE_LOCAL_RETRIEVAL",
    )
    enable_rag_answer_generation: bool = Field(
        default=False,
        alias="ENABLE_RAG_ANSWER_GENERATION",
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
    upstream_embedding_model: str = Field(
        default="qwen3-embedding-0.6b",
        alias="UPSTREAM_EMBEDDING_MODEL",
    )

    rag_knowledge_base_dir: Path = Field(
        default=PROJECT_ROOT / "rag_data",
        alias="RAG_KNOWLEDGE_BASE_DIR",
    )
    rag_embedding_dim: int = Field(
        default=1024,
        gt=0,
        alias="RAG_EMBEDDING_DIM",
    )
    rag_vector_top_k: int = Field(default=40, gt=0, alias="RAG_VECTOR_TOP_K")
    rag_fts_top_k: int = Field(default=40, gt=0, alias="RAG_FTS_TOP_K")
    rag_final_top_k: int = Field(default=8, gt=0, alias="RAG_FINAL_TOP_K")
    rag_max_chunks_per_doc: int = Field(
        default=3,
        gt=0,
        alias="RAG_MAX_CHUNKS_PER_DOC",
    )
    rag_enable_fts: bool = Field(default=True, alias="RAG_ENABLE_FTS")
    rag_allow_fts_fallback: bool = Field(
        default=True,
        alias="RAG_ALLOW_FTS_FALLBACK",
    )
    rag_rrf_k: int = Field(default=60, gt=0, alias="RAG_RRF_K")
    rag_vector_weight: float = Field(
        default=0.6,
        ge=0,
        alias="RAG_VECTOR_WEIGHT",
    )
    rag_fts_weight: float = Field(
        default=0.4,
        ge=0,
        alias="RAG_FTS_WEIGHT",
    )
    rag_allow_partial_index: bool = Field(
        default=False,
        alias="RAG_ALLOW_PARTIAL_INDEX",
    )

    rag_llm_context_window_tokens: int = Field(
        default=8192,
        gt=0,
        alias="RAG_LLM_CONTEXT_WINDOW_TOKENS",
    )
    rag_max_input_tokens: int = Field(
        default=6144,
        gt=0,
        alias="RAG_MAX_INPUT_TOKENS",
    )
    rag_max_output_tokens: int = Field(
        default=1024,
        gt=0,
        alias="RAG_MAX_OUTPUT_TOKENS",
    )
    rag_token_safety_margin: int = Field(
        default=1024,
        ge=0,
        alias="RAG_TOKEN_SAFETY_MARGIN",
    )
    rag_context_budget_tokens: int = Field(
        default=4096,
        gt=0,
        alias="RAG_CONTEXT_BUDGET_TOKENS",
    )
    rag_history_budget_tokens: int = Field(
        default=768,
        ge=0,
        alias="RAG_HISTORY_BUDGET_TOKENS",
    )
    rag_max_chunk_tokens: int = Field(
        default=1024,
        gt=0,
        alias="RAG_MAX_CHUNK_TOKENS",
    )
    rag_min_chunk_tokens: int = Field(
        default=64,
        gt=0,
        alias="RAG_MIN_CHUNK_TOKENS",
    )
    rag_generation_temperature: float = Field(
        default=0.2,
        ge=0,
        le=2,
        alias="RAG_GENERATION_TEMPERATURE",
    )

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

    @field_validator(
        "local_rag_model",
        "upstream_llm_model",
        "upstream_embedding_model",
        mode="before",
    )
    @classmethod
    def normalize_model_names(cls, value: Any) -> str:
        """Reject empty model names once they are required by the active mode."""
        return "" if value is None else str(value).strip()

    @field_validator("rag_knowledge_base_dir", mode="before")
    @classmethod
    def resolve_knowledge_base_dir(cls, value: Any) -> Path:
        """Resolve a configured knowledge-base root without touching the filesystem."""
        if value is None or not str(value).strip():
            raise ValueError("RAG_KNOWLEDGE_BASE_DIR cannot be empty")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve(strict=False)

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
        retrieval_is_active = (
            self.local_rag_answer_mode == "gateway"
            and self.enable_rag_router
            and self.enable_local_retrieval
        )
        if retrieval_is_active and not self.upstream_embedding_model:
            raise ValueError(
                "local retrieval requires: UPSTREAM_EMBEDDING_MODEL"
            )
        if self.enable_rag_answer_generation and not retrieval_is_active:
            raise ValueError(
                "RAG answer generation requires gateway mode, RAG router, and local retrieval"
            )
        if (
            self.rag_max_input_tokens
            + self.rag_max_output_tokens
            + self.rag_token_safety_margin
            > self.rag_llm_context_window_tokens
        ):
            raise ValueError(
                "RAG input, output, and safety budgets exceed the LLM context window"
            )
        if self.rag_context_budget_tokens > self.rag_max_input_tokens:
            raise ValueError(
                "RAG_CONTEXT_BUDGET_TOKENS cannot exceed RAG_MAX_INPUT_TOKENS"
            )
        if self.rag_history_budget_tokens > self.rag_max_input_tokens:
            raise ValueError(
                "RAG_HISTORY_BUDGET_TOKENS cannot exceed RAG_MAX_INPUT_TOKENS"
            )
        if self.rag_max_chunk_tokens > self.rag_context_budget_tokens:
            raise ValueError(
                "RAG_MAX_CHUNK_TOKENS cannot exceed RAG_CONTEXT_BUDGET_TOKENS"
            )
        if self.rag_min_chunk_tokens > self.rag_max_chunk_tokens:
            raise ValueError(
                "RAG_MIN_CHUNK_TOKENS cannot exceed RAG_MAX_CHUNK_TOKENS"
            )
        if self.rag_final_top_k > self.rag_vector_top_k + self.rag_fts_top_k:
            raise ValueError(
                "RAG_FINAL_TOP_K cannot exceed the combined candidate top-k"
            )
        if self.rag_max_chunks_per_doc > self.rag_final_top_k:
            raise ValueError(
                "RAG_MAX_CHUNKS_PER_DOC cannot exceed RAG_FINAL_TOP_K"
            )
        if self.rag_vector_weight + self.rag_fts_weight <= 0:
            raise ValueError("RAG retrieval weights must have a positive sum")
        if not self.rag_enable_fts and self.rag_vector_weight <= 0:
            raise ValueError(
                "RAG_VECTOR_WEIGHT must be positive when RAG_ENABLE_FTS=false"
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
