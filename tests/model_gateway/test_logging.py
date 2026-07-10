"""Tests for request IDs and access logging."""

from collections.abc import Iterator
import logging
import re

import pytest
from fastapi.testclient import TestClient

from model_gateway.config import reset_settings_cache
from model_gateway.logging_config import LOGGER_NAME
from model_gateway.main import app


@pytest.fixture(autouse=True)
def clean_gateway_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_NO_AUTH", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_response_includes_generated_request_id(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    request_id = response.headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)


def test_response_reuses_client_request_id(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "client-request-1"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "client-request-1"


def test_successful_protected_response_includes_request_id(client: TestClient) -> None:
    response = client.get(
        "/v1/models",
        headers={
            "Authorization": "Bearer change-me",
            "X-Request-ID": "models-request-1",
        },
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "models-request-1"


def test_auth_error_response_includes_request_id(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"X-Request-ID": "auth-request-1"})

    assert response.status_code == 401
    assert response.headers["x-request-id"] == "auth-request-1"


def test_access_log_contains_request_metadata_without_sensitive_values(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer secret-api-key",
            "X-Request-ID": "log-request-1",
        },
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "secret prompt text"}],
        },
    )

    assert response.status_code == 401
    log_text = caplog.text
    assert "request_id=log-request-1" in log_text
    assert "method=POST" in log_text
    assert "path=/v1/chat/completions" in log_text
    assert "status_code=401" in log_text
    assert "error_code=invalid_api_key" in log_text
    assert "secret-api-key" not in log_text
    assert "secret prompt text" not in log_text
