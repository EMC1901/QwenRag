"""Tests for model gateway health routes."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from model_gateway.config import reset_settings_cache
from model_gateway.errors import upstream_connection_error
from model_gateway.main import app


@pytest.fixture(autouse=True)
def clean_gateway_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_NO_AUTH", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_does_not_check_upstreams(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "model-gateway"}


def test_upstream_health_requires_auth(client: TestClient) -> None:
    response = client.get("/health/upstreams")

    assert response.status_code == 401


def test_upstream_health_returns_ok_when_both_upstreams_pass(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_check_models(upstream_name, base_url, **kwargs):
        calls.append((upstream_name, base_url, kwargs.get("upstream_api_key")))
        return {"object": "list", "data": []}

    monkeypatch.setattr("model_gateway.routes.check_models", fake_check_models)

    response = client.get(
        "/health/upstreams",
        headers={"Authorization": "Bearer change-me"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "upstreams": {
            "llm": {
                "ok": True,
                "base_url": "http://127.0.0.1:8001/v1",
            },
            "embedding": {
                "ok": True,
                "base_url": "http://127.0.0.1:8002/v1",
            },
        },
    }
    assert calls == [
        ("llm", "http://127.0.0.1:8001/v1", None),
        ("embedding", "http://127.0.0.1:8002/v1", None),
    ]


def test_upstream_health_forwards_request_id(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_ids = []

    async def fake_check_models(upstream_name, base_url, **kwargs):
        request_ids.append(kwargs.get("request_id"))
        return {"object": "list", "data": []}

    monkeypatch.setattr("model_gateway.routes.check_models", fake_check_models)

    response = client.get(
        "/health/upstreams",
        headers={
            "Authorization": "Bearer change-me",
            "X-Request-ID": "health-request-123",
        },
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "health-request-123"
    assert request_ids == ["health-request-123", "health-request-123"]


def test_upstream_health_returns_degraded_when_one_upstream_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_models(upstream_name, base_url, **kwargs):
        if upstream_name == "llm":
            raise upstream_connection_error("llm unavailable")
        return {"object": "list", "data": []}

    monkeypatch.setattr("model_gateway.routes.check_models", fake_check_models)

    response = client.get(
        "/health/upstreams",
        headers={"Authorization": "Bearer change-me"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["upstreams"]["llm"] == {
        "ok": False,
        "base_url": "http://127.0.0.1:8001/v1",
        "error": "llm unavailable",
        "code": "upstream_connection_failed",
    }
    assert body["upstreams"]["embedding"] == {
        "ok": True,
        "base_url": "http://127.0.0.1:8002/v1",
    }


def test_upstream_health_wraps_unexpected_exception(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_models(upstream_name, base_url, **kwargs):
        if upstream_name == "embedding":
            raise RuntimeError("unexpected failure")
        return {"object": "list", "data": []}

    monkeypatch.setattr("model_gateway.routes.check_models", fake_check_models)

    response = client.get(
        "/health/upstreams",
        headers={"Authorization": "Bearer change-me"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["upstreams"]["llm"] == {
        "ok": True,
        "base_url": "http://127.0.0.1:8001/v1",
    }
    assert body["upstreams"]["embedding"] == {
        "ok": False,
        "base_url": "http://127.0.0.1:8002/v1",
        "error": "Unexpected upstream health check failure",
        "code": "internal_server_error",
    }
