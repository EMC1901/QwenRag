"""Tests for stage 4 non-streaming chat completions in stub mode."""

from fastapi.testclient import TestClient

from local_rag_app.answer_service import STUB_ANSWER
from local_rag_app.main import create_app


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer none"}


def _valid_request() -> dict:
    return {
        "model": "local-rag",
        "messages": [{"role": "user", "content": "请确认本地接口已经启动。"}],
        "stream": False,
    }


def test_stub_chat_completion_returns_openai_compatible_response() -> None:
    """The first interface response must be usable before any RAG logic exists."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json=_valid_request(),
        )

    payload = response.json()
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert payload["id"].startswith("chatcmpl-local-")
    assert payload["object"] == "chat.completion"
    assert isinstance(payload["created"], int)
    assert payload["model"] == "local-rag"
    assert payload["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": STUB_ANSWER},
            "finish_reason": "stop",
        }
    ]
    assert payload["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_chat_completion_rejects_missing_model() -> None:
    """Missing OpenAI-required fields must return the documented 400 envelope."""
    request = _valid_request()
    del request["model"]

    with TestClient(create_app()) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=request)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_model"


def test_chat_completion_rejects_missing_messages() -> None:
    """A request cannot become a RAG question without a messages array."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={"model": "local-rag"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_messages"


def test_chat_completion_rejects_an_upstream_model_name() -> None:
    """Chatbox must use the stable local-rag API model, never qwen directly."""
    request = _valid_request()
    request["model"] = "qwen"

    with TestClient(create_app()) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=request)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model"


def test_chat_completion_rejects_non_text_message_content() -> None:
    """Multimodal message content is intentionally outside the first-release scope."""
    request = _valid_request()
    request["messages"][0]["content"] = [{"type": "text", "text": "hello"}]

    with TestClient(create_app()) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=request)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_messages"


def test_chat_completion_rejects_non_boolean_stream() -> None:
    """Prevent truthy strings from accidentally switching clients into SSE mode."""
    request = _valid_request()
    request["stream"] = "true"

    with TestClient(create_app()) as client:
        response = client.post("/v1/chat/completions", headers=_headers(), json=request)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_stream"
