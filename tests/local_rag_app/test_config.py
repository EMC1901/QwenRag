"""Configuration tests for stages 1 and 2."""

import pytest
from pydantic import ValidationError

from local_rag_app.config import Settings
from local_rag_app.main import create_app


def test_default_settings_are_safe() -> None:
    """The initial defaults keep the service local and in deterministic stub mode."""
    settings = Settings(_env_file=None)

    assert settings.local_rag_host == "127.0.0.1"
    assert settings.local_rag_port == 18080
    assert settings.local_rag_model == "local-rag"
    assert settings.local_rag_answer_mode == "stub"
    assert settings.enable_rag_router is False
    assert settings.enable_rag_answer_generation is False
    assert settings.rag_router_max_tokens == 128
    assert settings.rag_llm_context_window_tokens == 8192
    assert settings.rag_max_input_tokens == 6144
    assert settings.rag_max_output_tokens == 1024
    assert settings.local_rag_allow_no_auth is False
    assert settings.local_rag_api_keys == ["none"]


def test_rejects_non_loopback_host() -> None:
    """The first release must not bind a local RAG endpoint to the network."""
    with pytest.raises(ValidationError, match="LOCAL_RAG_HOST"):
        Settings(LOCAL_RAG_HOST="0.0.0.0", _env_file=None)


def test_gateway_mode_requires_gateway_settings() -> None:
    """Gateway mode must fail early instead of starting with unusable credentials."""
    with pytest.raises(ValidationError, match="gateway mode requires"):
        Settings(LOCAL_RAG_ANSWER_MODE="gateway", _env_file=None)


def test_router_enabled_gateway_mode_keeps_gateway_settings_required() -> None:
    """The router switch must not allow an unusable gateway configuration."""
    with pytest.raises(ValidationError, match="gateway mode requires"):
        Settings(
            LOCAL_RAG_ANSWER_MODE="gateway",
            ENABLE_RAG_ROUTER="true",
            _env_file=None,
        )


def test_router_enabled_gateway_mode_accepts_complete_gateway_settings() -> None:
    """The router can be enabled once the existing gateway contract is complete."""
    settings = Settings(
        LOCAL_RAG_ANSWER_MODE="gateway",
        ENABLE_RAG_ROUTER="true",
        MODEL_GATEWAY_BASE_URL="http://gateway.test:8010/v1",
        MODEL_GATEWAY_API_KEY="test-key",
        UPSTREAM_LLM_MODEL="qwen",
        _env_file=None,
    )

    assert settings.enable_rag_router is True


@pytest.mark.parametrize("max_tokens", [32, 128, 256])
def test_accepts_rag_router_token_limit_in_supported_range(max_tokens: int) -> None:
    """The router needs a small but configurable JSON-output limit."""
    settings = Settings(RAG_ROUTER_MAX_TOKENS=max_tokens, _env_file=None)

    assert settings.rag_router_max_tokens == max_tokens


@pytest.mark.parametrize("max_tokens", [31, 257, "not-an-integer", ""])
def test_rejects_invalid_rag_router_token_limit(max_tokens: int | str) -> None:
    """Invalid limits must fail before the local server accepts traffic."""
    with pytest.raises(ValidationError, match="RAG_ROUTER_MAX_TOKENS"):
        Settings(RAG_ROUTER_MAX_TOKENS=max_tokens, _env_file=None)


def test_rejects_empty_local_api_keys_when_auth_is_enabled() -> None:
    """An authenticated local service needs at least one accepted key."""
    with pytest.raises(ValidationError, match="LOCAL_RAG_API_KEYS"):
        Settings(
            LOCAL_RAG_API_KEYS="",
            LOCAL_RAG_ALLOW_NO_AUTH="false",
            _env_file=None,
        )


def test_normalizes_gateway_base_url() -> None:
    """Gateway paths can be joined safely when the supplied base URL has a slash."""
    settings = Settings(
        LOCAL_RAG_ANSWER_MODE="gateway",
        MODEL_GATEWAY_BASE_URL="http://127.0.0.1:8010/v1/",
        MODEL_GATEWAY_API_KEY="test-key",
        UPSTREAM_LLM_MODEL="qwen",
        _env_file=None,
    )

    assert settings.model_gateway_base_url == "http://127.0.0.1:8010/v1"


def test_app_can_be_created_without_routes() -> None:
    """Stage 1 creates an importable app shell; routes arrive in stage 3."""
    app = create_app()

    assert app.title == "QwenRag Local RAG App"


def _generation_settings(**overrides: object) -> Settings:
    """Create one valid settings object with every stage-7 prerequisite enabled."""
    values: dict[str, object] = {
        "LOCAL_RAG_ANSWER_MODE": "gateway",
        "ENABLE_RAG_ROUTER": "true",
        "ENABLE_LOCAL_RETRIEVAL": "true",
        "ENABLE_RAG_ANSWER_GENERATION": "true",
        "MODEL_GATEWAY_BASE_URL": "http://gateway.test:8010/v1",
        "MODEL_GATEWAY_API_KEY": "test-key",
        "UPSTREAM_LLM_MODEL": "qwen",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"LOCAL_RAG_ANSWER_MODE": "stub"},
        {"ENABLE_RAG_ROUTER": "false"},
        {"ENABLE_LOCAL_RETRIEVAL": "false"},
    ],
)
def test_generation_requires_the_complete_rag_route(
    overrides: dict[str, object],
) -> None:
    """Answer generation cannot be enabled without its gateway and retrieval inputs."""
    with pytest.raises(ValidationError, match="RAG answer generation requires"):
        _generation_settings(**overrides)


def test_generation_accepts_complete_rag_route() -> None:
    """The feature remains opt-in but loads once its complete route exists."""
    settings = _generation_settings()

    assert settings.enable_rag_answer_generation is True


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "RAG_MAX_INPUT_TOKENS": 7000,
                "RAG_MAX_OUTPUT_TOKENS": 1000,
                "RAG_TOKEN_SAFETY_MARGIN": 1000,
            },
            "budgets exceed",
        ),
        (
            {"RAG_CONTEXT_BUDGET_TOKENS": 7000},
            "CONTEXT_BUDGET_TOKENS",
        ),
        (
            {"RAG_HISTORY_BUDGET_TOKENS": 7000},
            "HISTORY_BUDGET_TOKENS",
        ),
        (
            {
                "RAG_CONTEXT_BUDGET_TOKENS": 1000,
                "RAG_MAX_CHUNK_TOKENS": 1001,
            },
            "MAX_CHUNK_TOKENS",
        ),
        (
            {
                "RAG_MAX_CHUNK_TOKENS": 100,
                "RAG_MIN_CHUNK_TOKENS": 101,
            },
            "MIN_CHUNK_TOKENS",
        ),
    ],
)
def test_rejects_inconsistent_rag_generation_budgets(
    overrides: dict[str, object],
    message: str,
) -> None:
    """Budget errors must be caught during startup, before any private prompt exists."""
    with pytest.raises(ValidationError, match=message):
        _generation_settings(**overrides)
