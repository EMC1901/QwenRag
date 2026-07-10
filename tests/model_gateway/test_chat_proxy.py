"""Tests for upstream chat HTTP client behavior."""

from collections.abc import Iterator
from contextlib import asynccontextmanager
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

from model_gateway.config import reset_settings_cache
from model_gateway.errors import (
    GatewayError,
    upstream_connection_error,
    upstream_timeout_error,
)
from model_gateway.http_client import (
    build_upstream_headers,
    call_json,
    filter_forward_headers,
    join_url,
    stream,
)
from model_gateway.logging_config import LOGGER_NAME
from model_gateway.main import app


@pytest.fixture(autouse=True)
def clean_gateway_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_NO_AUTH", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_UPSTREAM_API_KEY", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_join_url_normalizes_slashes() -> None:
    assert (
        join_url("http://127.0.0.1:8001/v1/", "/chat/completions")
        == "http://127.0.0.1:8001/v1/chat/completions"
    )
    assert (
        join_url("http://127.0.0.1:8001/v1", "chat/completions")
        == "http://127.0.0.1:8001/v1/chat/completions"
    )


def test_build_upstream_headers_uses_internal_api_key_only() -> None:
    headers = build_upstream_headers(
        upstream_api_key="internal-key",
        source_headers={
            "Authorization": "Bearer client-key",
            "X-Request-ID": "request-123",
        },
    )

    assert headers["Authorization"] == "Bearer internal-key"
    assert headers["X-Request-ID"] == "request-123"
    assert headers["Content-Type"] == "application/json"


def test_filter_forward_headers_removes_hop_by_hop_and_credentials() -> None:
    headers = filter_forward_headers(
        {
            "Authorization": "Bearer client-key",
            "Connection": "keep-alive",
            "X-Request-ID": "request-123",
        }
    )

    assert headers == {"X-Request-ID": "request-123"}


@pytest.mark.asyncio
async def test_call_json_posts_chat_request_to_upstream() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "chatcmpl-test"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await call_json(
            "llm",
            "http://127.0.0.1:8001/v1/",
            "/chat/completions",
            {"model": "qwen", "messages": []},
            upstream_api_key="internal-key",
            source_headers={"Authorization": "Bearer client-key"},
            request_id="request-123",
            client=client,
        )

    assert result == {"id": "chatcmpl-test"}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert captured["body"] == {"model": "qwen", "messages": []}
    assert captured["headers"]["authorization"] == "Bearer internal-key"
    assert captured["headers"]["x-request-id"] == "request-123"


@pytest.mark.asyncio
async def test_call_json_logs_upstream_metadata_without_sensitive_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "chatcmpl-test"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await call_json(
            "llm",
            "http://127.0.0.1:8001/v1",
            "/chat/completions",
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "secret prompt text"}],
            },
            upstream_api_key="internal-key",
            request_id="upstream-log-123",
            client=client,
        )

    log_text = caplog.text
    assert "request_id=upstream-log-123" in log_text
    assert "upstream_name=llm" in log_text
    assert "upstream_url=http://127.0.0.1:8001/v1/chat/completions" in log_text
    assert "upstream_status_code=200" in log_text
    assert "internal-key" not in log_text
    assert "secret prompt text" not in log_text


@pytest.mark.asyncio
async def test_call_json_maps_upstream_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc_info:
            await call_json(
                "llm",
                "http://127.0.0.1:8001/v1",
                "/chat/completions",
                {"model": "qwen"},
                client=client,
            )

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == "upstream_connection_failed"


@pytest.mark.asyncio
async def test_call_json_maps_upstream_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow response", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc_info:
            await call_json(
                "llm",
                "http://127.0.0.1:8001/v1",
                "/chat/completions",
                {"model": "qwen"},
                client=client,
            )

    assert exc_info.value.status_code == 504
    assert exc_info.value.code == "upstream_timeout"


@pytest.mark.asyncio
async def test_call_json_maps_upstream_http_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc_info:
            await call_json(
                "llm",
                "http://127.0.0.1:8001/v1",
                "/chat/completions",
                {"model": "qwen"},
                client=client,
            )

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "upstream_http_error"


@pytest.mark.asyncio
async def test_call_json_logs_upstream_http_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError):
            await call_json(
                "llm",
                "http://127.0.0.1:8001/v1",
                "/chat/completions",
                {"model": "qwen"},
                request_id="upstream-error-123",
                client=client,
            )

    log_text = caplog.text
    assert "request_id=upstream-error-123" in log_text
    assert "upstream_name=llm" in log_text
    assert "upstream_status_code=503" in log_text
    assert "error_code=upstream_http_error" in log_text


@pytest.mark.asyncio
async def test_stream_yields_upstream_response_bytes() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: first\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        async with stream(
            "llm",
            "http://127.0.0.1:8001/v1",
            "/chat/completions",
            {"model": "qwen", "stream": True},
            client=client,
        ) as response:
            chunks = [chunk async for chunk in response.aiter_bytes()]

    assert b"".join(chunks) == b"data: first\n\ndata: [DONE]\n\n"
    assert response.headers["content-type"] == "text/event-stream"


@pytest.mark.asyncio
async def test_stream_maps_upstream_http_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"busy")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc_info:
            async with stream(
                "llm",
                "http://127.0.0.1:8001/v1",
                "/chat/completions",
                {"model": "qwen", "stream": True},
                client=client,
            ):
                pass

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "upstream_http_error"


@pytest.mark.asyncio
async def test_stream_maps_upstream_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow stream", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc_info:
            async with stream(
                "llm",
                "http://127.0.0.1:8001/v1",
                "/chat/completions",
                {"model": "qwen", "stream": True},
                client=client,
            ):
                pass

    assert exc_info.value.status_code == 504
    assert exc_info.value.code == "upstream_timeout"


@pytest.mark.asyncio
async def test_stream_maps_upstream_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc_info:
            async with stream(
                "llm",
                "http://127.0.0.1:8001/v1",
                "/chat/completions",
                {"model": "qwen", "stream": True},
                client=client,
            ):
                pass

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == "upstream_connection_failed"


def test_chat_completions_route_requires_auth(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 401


def test_chat_completions_route_forwards_valid_request(
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
            "id": "chatcmpl-route-test",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }

    monkeypatch.setattr("model_gateway.routes.call_json", fake_call_json)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer change-me",
            "X-Request-ID": "request-route-123",
        },
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-route-test"
    assert captured["upstream_name"] == "llm"
    assert captured["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["path"] == "/chat/completions"
    assert captured["body"]["model"] == "qwen"
    assert captured["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["upstream_api_key"] is None
    assert captured["request_id"] == "request-route-123"
    assert captured["source_headers"]["x-request-id"] == "request-route-123"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({}, "missing_model"),
        ({"model": "qwen"}, "missing_messages"),
        ({"model": "qwen", "messages": "not-array"}, "invalid_messages"),
    ],
)
def test_chat_completions_route_validates_request_body(
    client: TestClient,
    payload: dict,
    code: str,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer change-me"},
        json=payload,
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == code


def test_chat_completions_route_rejects_invalid_json(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer change-me",
            "Content-Type": "application/json",
        },
        content=b"{not-json",
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_chat_completions_route_rejects_non_object_json(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer change-me"},
        json=["not", "an", "object"],
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_chat_completions_route_maps_upstream_connection_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_json(*args, **kwargs):
        raise upstream_connection_error("llm unavailable")

    monkeypatch.setattr("model_gateway.routes.call_json", fake_call_json)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer change-me"},
        json={"model": "qwen", "messages": []},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_connection_failed"


def test_chat_completions_route_maps_upstream_timeout(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_json(*args, **kwargs):
        raise upstream_timeout_error("llm timed out")

    monkeypatch.setattr("model_gateway.routes.call_json", fake_call_json)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer change-me"},
        json={"model": "qwen", "messages": []},
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout"


def test_chat_completions_route_streams_upstream_chunks(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeStreamResponse:
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            yield b"data: first\n\n"
            yield b"data: [DONE]\n\n"

    @asynccontextmanager
    async def fake_stream_upstream(
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
        yield FakeStreamResponse()

    monkeypatch.setattr("model_gateway.routes.stream_upstream", fake_stream_upstream)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer change-me",
            "X-Request-ID": "request-stream-123",
        },
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.content == b"data: first\n\ndata: [DONE]\n\n"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert captured["upstream_name"] == "llm"
    assert captured["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["path"] == "/chat/completions"
    assert captured["body"]["stream"] is True
    assert captured["upstream_api_key"] is None
    assert captured["request_id"] == "request-stream-123"
    assert captured["source_headers"]["x-request-id"] == "request-stream-123"


def test_chat_completions_route_stream_connection_failure_returns_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def fake_stream_upstream(*args, **kwargs):
        raise upstream_connection_error("llm unavailable")
        yield

    monkeypatch.setattr("model_gateway.routes.stream_upstream", fake_stream_upstream)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer change-me"},
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_connection_failed"


def test_chat_completions_route_stream_timeout_returns_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def fake_stream_upstream(*args, **kwargs):
        raise upstream_timeout_error("llm stream timed out")
        yield

    monkeypatch.setattr("model_gateway.routes.stream_upstream", fake_stream_upstream)

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer change-me"},
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout"
