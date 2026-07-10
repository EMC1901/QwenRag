"""Tests for model list route."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from model_gateway.config import reset_settings_cache
from model_gateway.main import app


@pytest.fixture(autouse=True)
def clean_gateway_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_NO_AUTH", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_models_requires_auth(client: TestClient) -> None:
    response = client.get("/v1/models")

    assert response.status_code == 401


def test_models_returns_configured_model_list(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": "Bearer change-me"})

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "qwen",
                "object": "model",
                "owned_by": "model-gateway",
            },
            {
                "id": "qwen3-embedding-0.6b",
                "object": "model",
                "owned_by": "model-gateway",
            },
        ],
    }


def test_models_uses_environment_model_names(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MODEL", "custom-llm")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding")
    reset_settings_cache()

    response = client.get("/v1/models", headers={"Authorization": "Bearer change-me"})

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == [
        "custom-llm",
        "custom-embedding",
    ]
