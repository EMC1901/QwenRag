"""Tests for model gateway configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from model_gateway.config import Settings
from model_gateway.http_client import build_timeout


def test_default_config_loads() -> None:
    settings = Settings(_env_file=None)

    assert settings.gateway_host == "0.0.0.0"
    assert settings.gateway_port == 8010
    assert settings.gateway_api_keys == ["change-me"]
    assert settings.llm_base_url == "http://127.0.0.1:8001/v1"
    assert settings.embedding_base_url == "http://127.0.0.1:8002/v1"


def test_base_urls_are_normalized() -> None:
    settings = Settings(
        GATEWAY_API_KEYS="test-key",
        LLM_BASE_URL="http://127.0.0.1:8001/v1/",
        EMBEDDING_BASE_URL="http://127.0.0.1:8002/v1///",
        _env_file=None,
    )

    assert settings.llm_base_url == "http://127.0.0.1:8001/v1"
    assert settings.embedding_base_url == "http://127.0.0.1:8002/v1"


def test_gateway_api_keys_are_parsed_from_csv() -> None:
    settings = Settings(GATEWAY_API_KEYS="alpha, beta,, gamma ", _env_file=None)

    assert settings.gateway_api_keys == ["alpha", "beta", "gamma"]


def test_settings_can_load_from_env_file() -> None:
    env_file = Path(__file__).parent / "fixtures" / "env.gateway"

    settings = Settings(_env_file=env_file)

    assert settings.gateway_api_keys == ["from-env-file"]
    assert settings.llm_base_url == "http://llm.example.test/v1"
    assert settings.embedding_base_url == "http://embedding.example.test/v1"
    assert settings.llm_model == "env-llm"
    assert settings.embedding_model == "env-embedding"
    assert settings.log_request_body is True


def test_allow_no_auth_permits_empty_api_key_list() -> None:
    settings = Settings(
        GATEWAY_ALLOW_NO_AUTH=True,
        GATEWAY_API_KEYS="",
        _env_file=None,
    )

    assert settings.gateway_allow_no_auth is True
    assert settings.gateway_api_keys == []


def test_disallow_no_auth_rejects_empty_api_key_list() -> None:
    with pytest.raises(ValidationError, match="GATEWAY_API_KEYS is required"):
        Settings(GATEWAY_ALLOW_NO_AUTH=False, GATEWAY_API_KEYS="", _env_file=None)


def test_log_level_is_normalized_to_uppercase() -> None:
    settings = Settings(GATEWAY_API_KEYS="test-key", LOG_LEVEL="debug", _env_file=None)

    assert settings.log_level == "DEBUG"


def test_build_timeout_uses_configured_values() -> None:
    settings = Settings(
        GATEWAY_API_KEYS="test-key",
        HTTP_CONNECT_TIMEOUT_SECONDS=1.5,
        HTTP_READ_TIMEOUT_SECONDS=2.5,
        HTTP_WRITE_TIMEOUT_SECONDS=3.5,
        HTTP_POOL_TIMEOUT_SECONDS=4.5,
        _env_file=None,
    )

    timeout = build_timeout(settings)

    assert timeout.connect == 1.5
    assert timeout.read == 2.5
    assert timeout.write == 3.5
    assert timeout.pool == 4.5
