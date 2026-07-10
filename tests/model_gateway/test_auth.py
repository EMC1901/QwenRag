"""Tests for model gateway API key authentication."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from model_gateway.config import reset_settings_cache
from model_gateway.main import app


@pytest.fixture(autouse=True)
def clean_gateway_auth_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep auth tests independent from the caller's shell environment."""
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_NO_AUTH", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def assert_auth_error(response) -> None:
    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "Unauthorized",
            "type": "authentication_error",
            "code": "invalid_api_key",
        }
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/health/upstreams"),
        ("get", "/v1/models"),
        ("post", "/v1/chat/completions"),
        ("post", "/v1/embeddings"),
    ],
)
def test_protected_routes_reject_missing_authorization(
    client: TestClient,
    method: str,
    path: str,
) -> None:
    if method == "get":
        response = getattr(client, method)(path)
    else:
        response = getattr(client, method)(path, json={})

    assert_auth_error(response)


def test_protected_route_rejects_non_bearer_authorization(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": "Token change-me"})

    assert_auth_error(response)


def test_protected_route_rejects_wrong_bearer_token(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})

    assert_auth_error(response)


def test_protected_route_accepts_configured_bearer_token(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": "Bearer change-me"})

    assert response.status_code == 200


def test_protected_route_accepts_one_of_multiple_configured_tokens(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_API_KEYS", "alpha, beta")
    reset_settings_cache()

    response = client.get("/v1/models", headers={"Authorization": "Bearer beta"})

    assert response.status_code == 200


def test_allow_no_auth_skips_api_key_check(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_ALLOW_NO_AUTH", "true")
    monkeypatch.setenv("GATEWAY_API_KEYS", "")
    reset_settings_cache()

    response = client.get("/v1/models")

    assert response.status_code == 200


def test_health_does_not_require_authorization(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "model-gateway"}
