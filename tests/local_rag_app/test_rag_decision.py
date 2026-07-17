"""Contract and service tests for the stage-5 RAG decision module."""

import json

import httpx
import pytest
from pydantic import ValidationError

from local_rag_app.config import Settings
from local_rag_app.errors import LocalRagError, gateway_timeout_error
from local_rag_app.gateway_client import ModelGatewayClient
from local_rag_app.rag_decision import (
    RAG_ROUTER_SYSTEM_PROMPT,
    RagDecision,
    RagDecisionService,
    build_router_request,
    parse_rag_decision,
)
from local_rag_app.schemas import ChatCompletionRequest


@pytest.mark.parametrize(
    ("need_rag", "reason_code"),
    [
        (True, "private_knowledge"),
        (False, "general_knowledge"),
    ],
)
def test_rag_decision_accepts_boolean_route_results(
    need_rag: bool,
    reason_code: str,
) -> None:
    """Both explicit route directions are valid internal decision results."""
    decision = RagDecision(need_rag=need_rag, reason_code=reason_code)

    assert decision.need_rag is need_rag
    assert decision.reason_code == reason_code


@pytest.mark.parametrize("invalid_value", ["true", "false", 1, 0, None])
def test_rag_decision_rejects_non_boolean_need_rag(invalid_value: object) -> None:
    """A malformed model response must never silently become a route decision."""
    with pytest.raises(ValidationError):
        RagDecision(need_rag=invalid_value, reason_code="ambiguous")


def test_rag_decision_rejects_unknown_reason_code() -> None:
    """Only documented diagnostic categories may cross the service boundary."""
    with pytest.raises(ValidationError):
        RagDecision(need_rag=True, reason_code="customer-secret")


def test_build_router_request_omits_client_system_and_preserves_chat_roles() -> None:
    """Chatbox system prompts must neither become roles nor influence classification."""
    settings = _settings()
    original = ChatCompletionRequest.model_validate(
        {
            "model": "local-rag",
            "messages": [
                {"role": "system", "content": "请用 Markdown 回答。"},
                {"role": "user", "content": "我们在讨论项目资料。"},
                {"role": "assistant", "content": "好的。"},
                {"role": "user", "content": "第二条呢？"},
            ],
            "stream": True,
        }
    )

    router_request = build_router_request(original, settings)
    dumped_messages = [message.model_dump(mode="json") for message in router_request.messages]

    assert [message["role"] for message in dumped_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert dumped_messages[0]["content"] == RAG_ROUTER_SYSTEM_PROMPT
    assert dumped_messages[1:] == [
        {"role": "user", "content": "我们在讨论项目资料。"},
        {"role": "assistant", "content": "好的。"},
        {"role": "user", "content": "第二条呢？"},
    ]
    assert router_request.stream is False


def test_router_prompt_is_specific_to_national_and_local_regulations() -> None:
    """The route contract must follow the deployed law/regulation corpus."""
    for expected_text in (
        "全国性法律",
        "地方性法规",
        "第几条/第几款",
        "法规库也可能没有依据",
        "不得因此改走 Direct",
    ):
        assert expected_text in RAG_ROUTER_SYSTEM_PROMPT
    assert "涉及客户、公司、项目、合同、内部制度" not in RAG_ROUTER_SYSTEM_PROMPT


def _settings() -> Settings:
    return Settings(
        LOCAL_RAG_ANSWER_MODE="gateway",
        MODEL_GATEWAY_BASE_URL="http://gateway.test:8010/v1",
        MODEL_GATEWAY_API_KEY="gateway-secret",
        UPSTREAM_LLM_MODEL="qwen",
        RAG_ROUTER_MAX_TOKENS=128,
        _env_file=None,
    )


def _request(*, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "local-rag",
            "messages": [
                {"role": "system", "content": "你是有帮助的助手。"},
                {"role": "user", "content": "根据我们的项目资料，验收日期是什么？"},
            ],
            "stream": stream,
            "temperature": 0.8,
            "max_tokens": 4096,
        }
    )


def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-router-1",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.parametrize(
    ("content", "need_rag", "reason_code"),
    [
        ('{"need_rag": true, "reason_code": "private_knowledge"}', True, "private_knowledge"),
        ('{"need_rag": false, "reason_code": "general_knowledge"}', False, "general_knowledge"),
        ('{"need_rag": true}', True, "ambiguous"),
        ('{"need_rag": false, "reason_code": "unrecognized"}', False, "ambiguous"),
    ],
)
def test_parse_rag_decision_accepts_only_documented_json_shape(
    content: str,
    need_rag: bool,
    reason_code: str,
) -> None:
    """Optional or unknown diagnostic labels normalize without weakening need_rag."""
    decision = parse_rag_decision(_completion(content))

    assert decision.need_rag is need_rag
    assert decision.reason_code == reason_code


@pytest.mark.parametrize(
    "content",
    [
        "```json\n{\"need_rag\": true}\n```",
        '{"need_rag": "false"}',
        "我认为需要查询资料",
        "[]",
    ],
)
def test_parse_rag_decision_rejects_informal_or_unsafe_model_output(content: str) -> None:
    """Only a strict JSON boolean can control whether private questions bypass RAG."""
    with pytest.raises(ValueError):
        parse_rag_decision(_completion(content))


@pytest.mark.asyncio
async def test_decide_forces_safe_router_request_and_gateway_credentials(monkeypatch) -> None:
    """The decision call must be deterministic and use only service credentials."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_completion('{"need_rag": true, "reason_code": "private_knowledge"}'),
            request=request,
        )

    settings = _settings()
    client = ModelGatewayClient(settings, transport=httpx.MockTransport(handler))
    monkeypatch.setattr("local_rag_app.rag_decision.get_request_id", lambda: "router-request-1")
    decision = await RagDecisionService(settings, gateway_client=client).decide(_request(stream=True))

    body = captured["body"]
    headers = captured["headers"]
    assert decision.need_rag is True
    assert captured["url"] == "http://gateway.test:8010/v1/chat/completions"
    assert headers["authorization"] == "Bearer gateway-secret"
    assert headers["x-request-id"] == "router-request-1"
    assert body["model"] == "qwen"
    assert body["stream"] is False
    assert body["temperature"] == 0
    assert body["max_tokens"] == 128
    assert body["messages"][0] == {
        "role": "system",
        "content": RAG_ROUTER_SYSTEM_PROMPT,
    }
    assert body["messages"][1:] == [
        {"role": "user", "content": "根据我们的项目资料，验收日期是什么？"},
    ]


class _FailingGatewayClient:
    async def complete_chat(self, *args, **kwargs) -> dict:
        raise gateway_timeout_error()


@pytest.mark.asyncio
async def test_decide_maps_gateway_and_parse_failures_to_safe_router_error() -> None:
    """The service must fail closed instead of treating an outage as need_rag=false."""
    service = RagDecisionService(_settings(), gateway_client=_FailingGatewayClient())

    with pytest.raises(LocalRagError) as error:
        await service.decide(_request())

    assert getattr(error.value, "status_code") == 503
    assert getattr(error.value, "code") == "rag_decision_unavailable"
