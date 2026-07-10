"""Unit tests for stage 6 communication with the server-side model gateway."""

import json

import httpx
import pytest

from local_rag_app.config import Settings
from local_rag_app.errors import LocalRagError
from local_rag_app.gateway_client import ModelGatewayClient
from local_rag_app.schemas import ChatCompletionRequest


def _settings() -> Settings:
    return Settings(
        LOCAL_RAG_ANSWER_MODE="gateway",
        MODEL_GATEWAY_BASE_URL="http://gateway.test:8010/v1",
        MODEL_GATEWAY_API_KEY="gateway-secret",
        UPSTREAM_LLM_MODEL="qwen",
        _env_file=None,
    )


def _request(*, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "local-rag",
            "messages": [{"role": "user", "content": "gateway test"}],
            "stream": stream,
            "temperature": 0.2,
        }
    )


def _successful_completion() -> dict:
    return {
        "id": "chatcmpl-upstream-1",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "gateway answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    }


@pytest.mark.asyncio
async def test_complete_rewrites_model_and_uses_gateway_credentials_only() -> None:
    """The local model contract and service-to-service credentials must stay separate."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_successful_completion(), request=request)

    client = ModelGatewayClient(_settings(), transport=httpx.MockTransport(handler))
    response = await client.complete_chat(
        _request(),
        local_model="local-rag",
        request_id="local-request-1",
    )

    assert captured["url"] == "http://gateway.test:8010/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer gateway-secret"
    assert captured["headers"]["x-request-id"] == "local-request-1"
    assert captured["body"]["model"] == "qwen"
    assert captured["body"]["temperature"] == 0.2
    assert response["model"] == "local-rag"


@pytest.mark.asyncio
async def test_complete_maps_connection_and_timeout_errors() -> None:
    """Network failure classes must become documented local API errors."""
    def connection_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    connection_client = ModelGatewayClient(
        _settings(),
        transport=httpx.MockTransport(connection_handler),
    )
    with pytest.raises(LocalRagError, match="connection failed") as connection_error:
        await connection_client.complete_chat(_request(), local_model="local-rag")
    assert connection_error.value.status_code == 502
    assert connection_error.value.code == "gateway_connection_failed"

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    timeout_client = ModelGatewayClient(
        _settings(),
        transport=httpx.MockTransport(timeout_handler),
    )
    with pytest.raises(LocalRagError, match="timed out") as timeout_error:
        await timeout_client.complete_chat(_request(), local_model="local-rag")
    assert timeout_error.value.status_code == 504
    assert timeout_error.value.code == "gateway_timeout"


@pytest.mark.asyncio
async def test_complete_maps_gateway_auth_and_invalid_json_errors() -> None:
    """Gateway failures must not leak upstream response details to Chatbox."""
    def unauthorized_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"}, request=request)

    unauthorized_client = ModelGatewayClient(
        _settings(),
        transport=httpx.MockTransport(unauthorized_handler),
    )
    with pytest.raises(LocalRagError) as auth_error:
        await unauthorized_client.complete_chat(_request(), local_model="local-rag")
    assert auth_error.value.code == "gateway_auth_failed"

    def invalid_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", request=request)

    invalid_json_client = ModelGatewayClient(
        _settings(),
        transport=httpx.MockTransport(invalid_json_handler),
    )
    with pytest.raises(LocalRagError) as invalid_json_error:
        await invalid_json_client.complete_chat(_request(), local_model="local-rag")
    assert invalid_json_error.value.code == "gateway_invalid_response"


@pytest.mark.asyncio
async def test_stream_rewrites_model_and_preserves_done_marker() -> None:
    """SSE relay must retain framing while hiding the upstream qwen model name."""
    upstream_events = (
        b'data: {"id":"chunk-1","object":"chat.completion.chunk","created":1,'
        b'"model":"qwen","choices":[{"index":0,"delta":{"content":"hello"},'
        b'"finish_reason":null}]}\n\n'
        b"data: [DONE]\n\n"
    )
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=upstream_events,
            request=request,
        )

    client = ModelGatewayClient(_settings(), transport=httpx.MockTransport(handler))
    stream = await client.open_chat_stream(_request(stream=True), local_model="local-rag")
    body = b"".join([chunk async for chunk in stream]).decode("utf-8")

    assert captured["headers"]["authorization"] == "Bearer gateway-secret"
    assert captured["body"]["model"] == "qwen"
    assert '"model": "local-rag"' in body
    assert body.endswith("data: [DONE]\n\n")
