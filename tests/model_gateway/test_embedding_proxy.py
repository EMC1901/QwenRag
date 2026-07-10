"""Tests for upstream embedding and model-list HTTP client behavior."""

from collections.abc import Iterator
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from model_gateway.config import reset_settings_cache
from model_gateway.errors import (
    GatewayError,
    upstream_connection_error,
    upstream_http_error,
    upstream_timeout_error,
)
from model_gateway.http_client import call_json, check_models
from model_gateway.main import app


@pytest.fixture(autouse=True)
def clean_gateway_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_NO_AUTH", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_UPSTREAM_API_KEY", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.asyncio
async def test_call_json_posts_embedding_request_to_upstream() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await call_json(
            "embedding",
            "http://127.0.0.1:8002/v1/",
            "/embeddings",
            {"model": "qwen3-embedding-0.6b", "input": "测试"},
            request_id="request-456",
            client=client,
        )

    assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8002/v1/embeddings"
    assert captured["headers"]["x-request-id"] == "request-456"
    assert "authorization" not in captured["headers"]


@pytest.mark.asyncio
async def test_check_models_uses_get_models_endpoint() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "qwen3-embedding-0.6b", "object": "model"}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await check_models(
            "embedding",
            "http://127.0.0.1:8002/v1",
            client=client,
        )

    assert captured["method"] == "GET"
    assert captured["url"] == "http://127.0.0.1:8002/v1/models"
    assert result["data"][0]["id"] == "qwen3-embedding-0.6b"


@pytest.mark.asyncio
async def test_check_models_forwards_request_id_and_upstream_api_key() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"object": "list", "data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await check_models(
            "embedding",
            "http://127.0.0.1:8002/v1",
            upstream_api_key="embedding-internal-key",
            request_id="models-request-123",
            client=client,
        )

    assert captured["headers"]["authorization"] == "Bearer embedding-internal-key"
    assert captured["headers"]["x-request-id"] == "models-request-123"


@pytest.mark.asyncio
async def test_embedding_client_maps_invalid_json_as_upstream_connection_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc_info:
            await call_json(
                "embedding",
                "http://127.0.0.1:8002/v1",
                "/embeddings",
                {"model": "qwen3-embedding-0.6b", "input": "测试"},
                client=client,
            )

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == "upstream_connection_failed"


def test_embeddings_route_requires_auth(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={})

    assert response.status_code == 401


def test_embeddings_route_forwards_valid_request(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    async def fake_call_json(
        upstream_name,
        base_url,
        path,
        body,
        **kwargs,
    ):
        captured["upstream_name"] = upstream_name
        captured["base_url"] = base_url
        captured["path"] = path
        captured["body"] = body
        captured["upstream_api_key"] = kwargs.get("upstream_api_key")
        captured["request_id"] = kwargs.get("request_id")
        captured["source_headers"] = kwargs.get("source_headers")
        return {
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "qwen3-embedding-0.6b",
        }

    monkeypatch.setattr("model_gateway.routes.call_json", fake_call_json)

    response = client.post(
        "/v1/embeddings",
        headers={
            "Authorization": "Bearer change-me",
            "X-Request-ID": "request-embedding-123",
        },
        json={"model": "qwen3-embedding-0.6b", "input": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert captured["upstream_name"] == "embedding"
    assert captured["base_url"] == "http://127.0.0.1:8002/v1"
    assert captured["path"] == "/embeddings"
    assert captured["body"] == {"model": "qwen3-embedding-0.6b", "input": "hello"}
    assert captured["upstream_api_key"] is None
    assert captured["request_id"] == "request-embedding-123"
    assert captured["source_headers"]["x-request-id"] == "request-embedding-123"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({}, "missing_model"),
        ({"model": "qwen3-embedding-0.6b"}, "missing_input"),
    ],
)
def test_embeddings_route_validates_request_body(
    client: TestClient,
    payload: dict,
    code: str,
) -> None:
    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer change-me"},
        json=payload,
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == code


def test_embeddings_route_rejects_invalid_json(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        headers={
            "Authorization": "Bearer change-me",
            "Content-Type": "application/json",
        },
        content=b"{not-json",
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_embeddings_route_rejects_non_object_json(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer change-me"},
        json=["not", "an", "object"],
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_embeddings_route_maps_upstream_connection_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_json(*args, **kwargs):
        raise upstream_connection_error("embedding unavailable")

    monkeypatch.setattr("model_gateway.routes.call_json", fake_call_json)

    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer change-me"},
        json={"model": "qwen3-embedding-0.6b", "input": "hello"},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_connection_failed"


def test_embeddings_route_maps_upstream_timeout(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_json(*args, **kwargs):
        raise upstream_timeout_error("embedding timed out")

    monkeypatch.setattr("model_gateway.routes.call_json", fake_call_json)

    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer change-me"},
        json={"model": "qwen3-embedding-0.6b", "input": "hello"},
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout"


def test_embeddings_route_maps_upstream_http_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_json(*args, **kwargs):
        raise upstream_http_error(503, "embedding upstream returned HTTP 503")

    monkeypatch.setattr("model_gateway.routes.call_json", fake_call_json)

    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer change-me"},
        json={"model": "qwen3-embedding-0.6b", "input": "hello"},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "upstream_http_error"
