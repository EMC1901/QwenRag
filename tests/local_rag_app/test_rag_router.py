"""Routing and HTTP tests for the stage-5 answer-service integration."""

import logging
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from local_rag_app import routes
from local_rag_app.answer_service import (
    GatewayAnswerService,
    RagRouterAnswerService,
    StubAnswerService,
    get_answer_service,
)
from local_rag_app.config import Settings
from local_rag_app.errors import (
    LocalRagError,
    rag_decision_unavailable_error,
    rag_retrieval_unavailable_error,
)
from local_rag_app.logging_config import LOGGER_NAME
from local_rag_app.main import create_app
from local_rag_app.rag_decision import RagDecision
from local_rag_app.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    Usage,
)


def _settings(
    *,
    answer_mode: str = "gateway",
    router_enabled: bool = True,
    retrieval_enabled: bool = False,
) -> Settings:
    return Settings(
        LOCAL_RAG_ANSWER_MODE=answer_mode,
        ENABLE_RAG_ROUTER=router_enabled,
        ENABLE_LOCAL_RETRIEVAL=retrieval_enabled,
        MODEL_GATEWAY_BASE_URL="http://gateway.test:8010/v1",
        MODEL_GATEWAY_API_KEY="gateway-secret",
        UPSTREAM_LLM_MODEL="qwen",
        _env_file=None,
    )


def _request(*, stream: bool = False, content: str = "测试问题") -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "local-rag",
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        }
    )


class _FakeDecisionService:
    def __init__(self, result: RagDecision | LocalRagError) -> None:
        self.result = result
        self.calls = 0

    async def decide(self, request: ChatCompletionRequest) -> RagDecision:
        self.calls += 1
        if isinstance(self.result, LocalRagError):
            raise self.result
        return self.result


class _FakeGatewayAnswerService:
    def __init__(self) -> None:
        self.complete_calls = 0
        self.stream_calls = 0

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.complete_calls += 1
        return ChatCompletionResponse(
            id="chatcmpl-test",
            created=1,
            model="local-rag",
            choices=[CompletionChoice(message=AssistantMessage(content="direct answer"))],
            usage=Usage(),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        self.stream_calls += 1
        return self._events()

    async def _events(self) -> AsyncIterator[bytes]:
        yield b"data: direct answer\n\n"
        yield b"data: [DONE]\n\n"


class _FakeRetriever:
    """Record stage-8 retrieval calls without loading assets or contacting a gateway."""

    def __init__(self, error: LocalRagError | None = None) -> None:
        self.error = error
        self.queries: list[str] = []

    async def retrieve(self, query: str) -> "_FakeRetrievalResult":
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return _FakeRetrievalResult()


class _FakeRetrievalResult:
    """Metric-only shape consumed by the stage-9 request-log integration."""

    retrieval_mode = "hybrid"
    hits = [object(), object()]
    vector_candidate_count = 3
    fts_candidate_count = 2


@pytest.mark.asyncio
async def test_router_directs_non_rag_request_to_existing_gateway_answer_service() -> None:
    """A general question must preserve the existing direct-answer behavior."""
    decision_service = _FakeDecisionService(
        RagDecision(need_rag=False, reason_code="general_knowledge")
    )
    direct_service = _FakeGatewayAnswerService()
    service = RagRouterAnswerService(
        _settings(),
        decision_service=decision_service,
        gateway_answer_service=direct_service,
    )

    response = await service.complete(_request())

    assert response.choices[0].message.content == "direct answer"
    assert decision_service.calls == 1
    assert direct_service.complete_calls == 1
    assert direct_service.stream_calls == 0


@pytest.mark.asyncio
async def test_router_never_calls_retriever_for_a_non_rag_decision() -> None:
    """The new dependency must not affect normal direct gateway answers."""
    direct_service = _FakeGatewayAnswerService()
    retriever = _FakeRetriever()
    service = RagRouterAnswerService(
        _settings(retrieval_enabled=True),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=False, reason_code="general_knowledge")
        ),
        gateway_answer_service=direct_service,
        retriever=retriever,  # type: ignore[arg-type]
    )

    await service.complete(_request())

    assert retriever.queries == []
    assert direct_service.complete_calls == 1


@pytest.mark.asyncio
async def test_router_directs_non_rag_stream_to_existing_gateway_answer_service() -> None:
    """The direct route must retain normal SSE completion behavior."""
    direct_service = _FakeGatewayAnswerService()
    service = RagRouterAnswerService(
        _settings(),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=False, reason_code="general_knowledge")
        ),
        gateway_answer_service=direct_service,
    )

    stream = await service.stream(_request(stream=True))
    body = b"".join([event async for event in stream])

    assert body.endswith(b"data: [DONE]\n\n")
    assert direct_service.complete_calls == 0
    assert direct_service.stream_calls == 1


@pytest.mark.asyncio
async def test_router_blocks_rag_request_before_any_direct_answer_is_opened() -> None:
    """Private-data questions cannot silently receive an ungrounded direct answer."""
    direct_service = _FakeGatewayAnswerService()
    service = RagRouterAnswerService(
        _settings(),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=True, reason_code="private_knowledge")
        ),
        gateway_answer_service=direct_service,
    )

    with pytest.raises(LocalRagError) as error:
        await service.complete(_request())

    assert error.value.code == "rag_retrieval_not_ready"
    assert direct_service.complete_calls == 0
    assert direct_service.stream_calls == 0


@pytest.mark.asyncio
async def test_router_blocks_rag_stream_before_any_sse_is_opened() -> None:
    """A retrieval-pending stream request must return a JSON error before SSE starts."""
    direct_service = _FakeGatewayAnswerService()
    service = RagRouterAnswerService(
        _settings(),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=True, reason_code="conversation")
        ),
        gateway_answer_service=direct_service,
    )

    with pytest.raises(LocalRagError) as error:
        await service.stream(_request(stream=True))

    assert error.value.code == "rag_retrieval_not_ready"
    assert direct_service.stream_calls == 0


@pytest.mark.asyncio
async def test_router_retrieves_latest_user_query_then_reports_answer_generation_pending() -> None:
    """Successful retrieval reaches the staged 503 without opening a direct LLM answer."""
    direct_service = _FakeGatewayAnswerService()
    retriever = _FakeRetriever()
    request = ChatCompletionRequest.model_validate(
        {
            "model": "local-rag",
            "messages": [
                {"role": "user", "content": "旧问题"},
                {"role": "assistant", "content": "旧回答"},
                {"role": "user", "content": "   "},
                {"role": "user", "content": " 最新私有问题 "},
            ],
        }
    )
    service = RagRouterAnswerService(
        _settings(retrieval_enabled=True),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=True, reason_code="private_knowledge")
        ),
        gateway_answer_service=direct_service,
        retriever=retriever,  # type: ignore[arg-type]
    )

    with pytest.raises(LocalRagError) as error:
        await service.complete(request)

    assert error.value.code == "rag_answer_generation_not_ready"
    assert retriever.queries == ["最新私有问题"]
    assert direct_service.complete_calls == 0
    assert direct_service.stream_calls == 0


@pytest.mark.asyncio
async def test_router_retrieves_before_streaming_and_returns_json_error_before_sse() -> None:
    """The retrieval and staged failure happen before a streaming response is opened."""
    direct_service = _FakeGatewayAnswerService()
    retriever = _FakeRetriever()
    service = RagRouterAnswerService(
        _settings(retrieval_enabled=True),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=True, reason_code="private_knowledge")
        ),
        gateway_answer_service=direct_service,
        retriever=retriever,  # type: ignore[arg-type]
    )

    with pytest.raises(LocalRagError) as error:
        await service.stream(_request(stream=True))

    assert error.value.code == "rag_answer_generation_not_ready"
    assert retriever.queries == ["测试问题"]
    assert direct_service.stream_calls == 0


@pytest.mark.asyncio
async def test_router_propagates_retrieval_failure_without_direct_llm_fallback() -> None:
    """A failed local retrieval remains fail-closed for private-data requests."""
    direct_service = _FakeGatewayAnswerService()
    retriever = _FakeRetriever(rag_retrieval_unavailable_error())
    service = RagRouterAnswerService(
        _settings(retrieval_enabled=True),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=True, reason_code="private_knowledge")
        ),
        gateway_answer_service=direct_service,
        retriever=retriever,  # type: ignore[arg-type]
    )

    with pytest.raises(LocalRagError) as error:
        await service.complete(_request())

    assert error.value.code == "rag_retrieval_unavailable"
    assert retriever.queries == ["测试问题"]
    assert direct_service.complete_calls == 0


@pytest.mark.asyncio
async def test_router_does_not_fallback_to_direct_answer_when_decision_fails() -> None:
    """An unavailable decision service must fail closed for private-data safety."""
    direct_service = _FakeGatewayAnswerService()
    service = RagRouterAnswerService(
        _settings(),
        decision_service=_FakeDecisionService(rag_decision_unavailable_error()),
        gateway_answer_service=direct_service,
    )

    with pytest.raises(LocalRagError) as error:
        await service.complete(_request())

    assert error.value.code == "rag_decision_unavailable"
    assert direct_service.complete_calls == 0
    assert direct_service.stream_calls == 0


def test_service_selection_preserves_stub_and_existing_gateway_modes() -> None:
    """The feature flag only changes the configured gateway implementation."""
    assert isinstance(get_answer_service(_settings(answer_mode="stub")), StubAnswerService)
    assert isinstance(get_answer_service(_settings(router_enabled=False)), GatewayAnswerService)
    assert isinstance(get_answer_service(_settings(router_enabled=True)), RagRouterAnswerService)


def test_rag_route_error_is_openai_json_and_private_values_are_not_logged(
    monkeypatch,
    caplog,
) -> None:
    """The HTTP route exposes a safe 503 and writes only enum routing metadata."""
    private_prompt = "客户私有问题：合同编号-SECRET-123"
    local_key = "local-api-key-SECRET"
    monkeypatch.setenv("LOCAL_RAG_API_KEYS", local_key)
    service = RagRouterAnswerService(
        _settings(),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=True, reason_code="private_knowledge")
        ),
        gateway_answer_service=_FakeGatewayAnswerService(),
    )
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {local_key}",
                "X-Request-ID": "router-http-1",
            },
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": private_prompt}],
                "stream": True,
            },
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "router-http-1"
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.json()["error"] == {
        "message": "This question requires the local knowledge base, but retrieval is not ready",
        "type": "service_unavailable_error",
        "code": "rag_retrieval_not_ready",
    }
    assert "rag_route=retrieval_pending" in log_text
    assert "rag_decision_source=llm" in log_text
    assert private_prompt not in log_text
    assert local_key not in log_text


def test_decision_failure_is_openai_json_and_never_falls_back_to_direct_answer(
    monkeypatch,
    caplog,
) -> None:
    """A routing outage is visible as a safe 503 before any direct LLM call occurs."""
    direct_service = _FakeGatewayAnswerService()
    service = RagRouterAnswerService(
        _settings(),
        decision_service=_FakeDecisionService(rag_decision_unavailable_error()),
        gateway_answer_service=direct_service,
    )
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "普通测试问题"}],
                "stream": False,
            },
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rag_decision_unavailable"
    assert direct_service.complete_calls == 0
    assert direct_service.stream_calls == 0
    assert "rag_route=decision_unavailable" in log_text
    assert "rag_decision_source= " in log_text


def test_retrieval_success_is_openai_json_error_before_streaming_begins(
    monkeypatch,
    caplog,
) -> None:
    """The real HTTP path returns the staged 503 instead of an SSE response."""
    private_prompt = "客户私有问题：合同编号-SECRET-123"
    retriever = _FakeRetriever()
    service = RagRouterAnswerService(
        _settings(retrieval_enabled=True),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=True, reason_code="private_knowledge")
        ),
        gateway_answer_service=_FakeGatewayAnswerService(),
        retriever=retriever,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": private_prompt}],
                "stream": True,
            },
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 503
    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.json()["error"]["code"] == "rag_answer_generation_not_ready"
    assert retriever.queries == [private_prompt]
    assert "rag_route=retrieval_ready" in log_text
    assert "retrieval_mode=hybrid" in log_text
    assert "retrieval_hit_count=2" in log_text
    assert "retrieval_vector_candidate_count=3" in log_text
    assert "retrieval_fts_candidate_count=2" in log_text
    assert "retrieval_duration_ms=" in log_text
    assert private_prompt not in log_text


def test_retrieval_failure_logs_only_route_and_duration_without_private_content(
    monkeypatch,
    caplog,
) -> None:
    """A failed retrieval records no query, source text, or local credential."""
    private_prompt = "客户私有问题：合同编号-SECRET-456"
    local_key = "local-api-key-SECRET"
    service = RagRouterAnswerService(
        _settings(retrieval_enabled=True),
        decision_service=_FakeDecisionService(
            RagDecision(need_rag=True, reason_code="private_knowledge")
        ),
        gateway_answer_service=_FakeGatewayAnswerService(),
        retriever=_FakeRetriever(rag_retrieval_unavailable_error()),  # type: ignore[arg-type]
    )
    monkeypatch.setenv("LOCAL_RAG_API_KEYS", local_key)
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {local_key}"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": private_prompt}],
                "stream": False,
            },
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rag_retrieval_unavailable"
    assert "rag_route=retrieval_unavailable" in log_text
    assert "retrieval_mode= " in log_text
    assert "retrieval_duration_ms=" in log_text
    assert private_prompt not in log_text
    assert local_key not in log_text
