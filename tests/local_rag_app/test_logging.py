"""Privacy and diagnostic-field tests for stage 7 request logging."""

import logging
import json
import re
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from local_rag_app import routes
from local_rag_app.answer_service import RagRouterAnswerService, StubAnswerService
from local_rag_app.config import Settings
from local_rag_app.context_builder import ContextBuildError
from local_rag_app.context_models import ContextBuildResult, SelectedContextHit
from local_rag_app.errors import gateway_timeout_error
from local_rag_app.rag_decision import RagDecision
from local_rag_app.rag_generation import RagGenerationService
from local_rag_app.reference_formatter import ReferenceFormatError
from local_rag_app.reference_models import ReferenceBuildResult, ReferenceFile
from local_rag_app.retrieval_models import RetrievalHit, RetrievalResult
from local_rag_app.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    Usage,
)
from local_rag_app.logging_config import LOGGER_NAME
from local_rag_app.main import create_app


def test_request_log_has_diagnostic_fields_without_prompt_or_key(monkeypatch, caplog) -> None:
    """Logging must make failures traceable without storing private conversation data."""
    private_prompt = "客户私有问题：合同编号-SECRET-123"
    local_key = "local-api-key-SECRET"
    monkeypatch.setenv("LOCAL_RAG_API_KEYS", local_key)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {local_key}",
                "X-Request-ID": "privacy-log-1",
            },
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": private_prompt}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "request_id=privacy-log-1" in log_text
    assert "method=POST" in log_text
    assert "path=/v1/chat/completions" in log_text
    assert "status_code=200" in log_text
    assert "answer_mode=stub" in log_text
    assert "rag_route=router_disabled" in log_text
    assert "rag_decision_source= " in log_text
    assert "retrieval_mode= " in log_text
    assert "retrieval_hit_count= " in log_text
    assert "retrieval_duration_ms= " in log_text
    assert "context_selected_hit_count= " in log_text
    assert "context_dropped_hit_count= " in log_text
    assert "context_estimated_input_tokens= " in log_text
    assert "context_estimated_evidence_tokens= " in log_text
    assert "context_estimated_history_tokens= " in log_text
    assert "context_truncated_hit_count= " in log_text
    assert "reference_display_enabled= " in log_text
    assert "reference_appended= " in log_text
    assert "reference_file_count= " in log_text
    assert "reference_location_count= " in log_text
    assert "reference_evidence_count= " in log_text
    assert "reference_text_chars= " in log_text
    assert "generation_stream= " in log_text
    assert "generation_duration_ms= " in log_text
    assert "error_code=" in log_text
    assert private_prompt not in log_text
    assert local_key not in log_text
    assert "Authorization" not in log_text


def test_authentication_failure_is_logged_with_error_code(caplog) -> None:
    """Failed auth is diagnosable through the code, never the supplied key value."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key-SECRET"},
        )

    assert response.status_code == 401
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "status_code=401" in log_text
    assert "error_code=invalid_api_key" in log_text
    assert "wrong-key-SECRET" not in log_text


def _generation_settings(*, reference_display: bool = False) -> Settings:
    return Settings(
        LOCAL_RAG_ANSWER_MODE="gateway",
        ENABLE_RAG_ROUTER=True,
        ENABLE_LOCAL_RETRIEVAL=True,
        ENABLE_RAG_ANSWER_GENERATION=True,
        ENABLE_REFERENCE_DISPLAY=reference_display,
        MODEL_GATEWAY_BASE_URL="http://gateway.test:8010/v1",
        MODEL_GATEWAY_API_KEY="gateway-key-SECRET",
        UPSTREAM_LLM_MODEL="qwen",
        _env_file=None,
    )


def _private_hit() -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        chunk_id="chunk-private",
        doc_id="doc-private",
        chunk_text="PRIVATE-CHUNK-SECRET-456",
        title="private.docx",
        relative_path="private.docx",
        final_score=0.9,
        matched_by="both",
    )


def _private_result(*, hits: list[RetrievalHit] | None = None) -> RetrievalResult:
    values = hits if hits is not None else [_private_hit()]
    return RetrievalResult(
        hits=values,
        candidate_count=len(values) + 2,
        vector_candidate_count=len(values) + 1,
        fts_candidate_count=len(values),
        embedding_model="embed-test",
        embedding_dim=1024,
        retrieval_mode="hybrid",
    )


class _PrivateDecisionService:
    async def decide(self, request: ChatCompletionRequest) -> RagDecision:
        return RagDecision(need_rag=True, reason_code="private_knowledge")


class _PrivateRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result

    async def retrieve(self, query: str) -> RetrievalResult:
        return self.result


class _PrivateContextBuilder:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def build(
        self,
        request: ChatCompletionRequest,
        result: RetrievalResult,
    ) -> ContextBuildResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        hit = _private_hit()
        return ContextBuildResult(
            system_prompt="PRIVATE-SYSTEM-PROMPT-SECRET",
            user_prompt="PRIVATE-QUESTION-SECRET-123\nPRIVATE-CHUNK-SECRET-456",
            selected_hits=[
                SelectedContextHit(
                    evidence_no=1,
                    hit=hit,
                    text_for_prompt=hit.chunk_text,
                    estimated_tokens=42,
                    truncated=True,
                )
            ],
            dropped_hit_count=2,
            estimated_input_tokens=333,
            estimated_context_tokens=222,
            estimated_history_tokens=11,
            history_message_count=1,
        )


class _PrivateGatewayAnswerService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self.error is not None:
            raise self.error
        return ChatCompletionResponse(
            id="chatcmpl-private",
            created=1,
            model="local-rag",
            choices=[
                CompletionChoice(
                    message=AssistantMessage(content="PRIVATE-ANSWER-SECRET-789")
                )
            ],
            usage=Usage(),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        if self.error is not None:
            raise self.error
        return self._events()

    async def _events(self) -> AsyncIterator[bytes]:
        metadata = {"id": "chatcmpl-private", "created": 1, "model": "local-rag"}
        for choice in (
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
            {
                "index": 0,
                "delta": {"content": "PRIVATE-ANSWER-SECRET-789"},
                "finish_reason": None,
            },
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ):
            yield (
                f"data: {json.dumps({**metadata, 'choices': [choice]})}\n\n"
            ).encode("utf-8")
        yield b"data: [DONE]\n\n"


class _PrivateReferenceFormatter:
    """Return deliberately sensitive source text to prove logs retain only counts."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[list[SelectedContextHit]] = []
        self.result = ReferenceBuildResult(
            section_text=(
                "参考文件：\n[1] PRIVATE-FILENAME-SECRET.docx\n"
                "    位置：C:\\PRIVATE-PATH-SECRET\\file.docx / PRIVATE-SECTION-SECRET\n"
                "    对应资料：[资料1]"
            ),
            files=[
                ReferenceFile(
                    reference_no=1,
                    display_name="PRIVATE-FILENAME-SECRET.docx",
                    locations=[
                        "C:\\PRIVATE-PATH-SECRET\\file.docx",
                        "PRIVATE-SECTION-SECRET",
                    ],
                    evidence_nos=[1],
                )
            ],
            selected_hit_count=1,
            location_count=2,
        )

    def build(self, selected_hits: list[SelectedContextHit]) -> ReferenceBuildResult:
        self.calls.append(selected_hits)
        if self.error is not None:
            raise self.error
        return self.result


def _generation_router(
    *,
    result: RetrievalResult | None = None,
    builder: _PrivateContextBuilder | None = None,
    reference_display: bool = False,
    formatter: _PrivateReferenceFormatter | None = None,
    gateway: _PrivateGatewayAnswerService | None = None,
) -> RagRouterAnswerService:
    settings = _generation_settings(reference_display=reference_display)
    active_gateway = gateway or _PrivateGatewayAnswerService()
    generation = RagGenerationService(
        settings,
        context_builder=builder or _PrivateContextBuilder(),  # type: ignore[arg-type]
        gateway_answer_service=active_gateway,
        reference_formatter=formatter,
    )
    return RagRouterAnswerService(
        settings,
        decision_service=_PrivateDecisionService(),  # type: ignore[arg-type]
        gateway_answer_service=active_gateway,  # type: ignore[arg-type]
        retriever=_PrivateRetriever(result or _private_result()),  # type: ignore[arg-type]
        rag_generation_service=generation,
    )


def _assert_generation_secrets_absent(log_text: str) -> None:
    for secret in (
        "PRIVATE-QUESTION-SECRET-123",
        "PRIVATE-CHUNK-SECRET-456",
        "PRIVATE-ANSWER-SECRET-789",
        "PRIVATE-SYSTEM-PROMPT-SECRET",
        "gateway-key-SECRET",
        "PRIVATE-FILENAME-SECRET.docx",
        "C:\\PRIVATE-PATH-SECRET\\file.docx",
        "PRIVATE-SECTION-SECRET",
    ):
        assert secret not in log_text


def test_rag_generation_logs_safe_context_aggregates_without_private_bodies(
    monkeypatch,
    caplog,
) -> None:
    """A complete RAG answer logs metrics, not the prompt, chunk, answer, or key."""
    service = _generation_router()
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none", "X-Request-ID": "generation-log-1"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "PRIVATE-QUESTION-SECRET-123"}],
                "stream": False,
            },
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 200
    assert "request_id=generation-log-1" in log_text
    assert "rag_route=rag_generation" in log_text
    assert "context_selected_hit_count=1" in log_text
    assert "context_dropped_hit_count=2" in log_text
    assert "context_estimated_input_tokens=333" in log_text
    assert "context_estimated_evidence_tokens=222" in log_text
    assert "context_estimated_history_tokens=11" in log_text
    assert "context_truncated_hit_count=1" in log_text
    assert "reference_display_enabled=false" in log_text
    assert "reference_appended=false" in log_text
    assert "reference_file_count=0" in log_text
    assert "reference_location_count=0" in log_text
    assert "reference_evidence_count=0" in log_text
    assert "reference_text_chars=0" in log_text
    assert "generation_stream=false" in log_text
    assert re.search(r"generation_duration_ms=\d+\.\d{2}", log_text)
    _assert_generation_secrets_absent(log_text)


def test_stream_and_no_evidence_generation_logs_are_safe_and_do_not_leak_content(
    monkeypatch,
    caplog,
) -> None:
    """Both SSE and the intentional no-evidence branch retain only safe aggregates."""
    stream_service = _generation_router()
    monkeypatch.setattr(routes, "get_answer_service", lambda _: stream_service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "PRIVATE-QUESTION-SECRET-123"}],
                "stream": True,
            },
        ) as response:
            stream_body = b"".join(response.iter_bytes()).decode("utf-8")

    first_log = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 200
    assert "PRIVATE-ANSWER-SECRET-789" in stream_body
    assert "generation_stream=true" in first_log
    _assert_generation_secrets_absent(first_log)

    caplog.clear()
    no_evidence_service = _generation_router(
        result=_private_result(hits=[]),
        reference_display=True,
    )
    monkeypatch.setattr(routes, "get_answer_service", lambda _: no_evidence_service)
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "PRIVATE-QUESTION-SECRET-123"}],
                "stream": False,
            },
        )

    second_log = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 200
    assert "rag_route=rag_no_evidence" in second_log
    assert "context_selected_hit_count=0" in second_log
    assert "context_dropped_hit_count=0" in second_log
    assert "reference_display_enabled=true" in second_log
    assert "reference_appended=false" in second_log
    assert "reference_file_count=0" in second_log
    assert "reference_location_count=0" in second_log
    assert "reference_evidence_count=0" in second_log
    assert "reference_text_chars=0" in second_log
    assert "generation_stream=false" in second_log
    _assert_generation_secrets_absent(second_log)


def test_visible_references_log_only_aggregate_counts_for_normal_and_stream(
    monkeypatch,
    caplog,
) -> None:
    """Visible source text is returned to the client but never copied into logs."""
    formatter = _PrivateReferenceFormatter()
    service = _generation_router(reference_display=True, formatter=formatter)
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none", "X-Request-ID": "reference-log-1"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "PRIVATE-QUESTION-SECRET-123"}],
                "stream": False,
            },
        )

    normal_log = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 200
    assert formatter.result.section_text in response.json()["choices"][0]["message"]["content"]
    assert len(formatter.calls) == 1
    assert "request_id=reference-log-1" in normal_log
    assert "reference_display_enabled=true" in normal_log
    assert "reference_appended=true" in normal_log
    assert "reference_file_count=1" in normal_log
    assert "reference_location_count=2" in normal_log
    assert "reference_evidence_count=1" in normal_log
    assert f"reference_text_chars={len(formatter.result.section_text)}" in normal_log
    _assert_generation_secrets_absent(normal_log)

    caplog.clear()
    stream_formatter = _PrivateReferenceFormatter()
    stream_service = _generation_router(
        reference_display=True,
        formatter=stream_formatter,
    )
    monkeypatch.setattr(routes, "get_answer_service", lambda _: stream_service)
    with TestClient(create_app()) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none", "X-Request-ID": "reference-log-2"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "PRIVATE-QUESTION-SECRET-123"}],
                "stream": True,
            },
        ) as response:
            stream_body = b"".join(response.iter_bytes()).decode("utf-8")

    stream_log = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 200
    assert "data: [DONE]" in stream_body
    assert len(stream_formatter.calls) == 1
    assert "request_id=reference-log-2" in stream_log
    assert "reference_display_enabled=true" in stream_log
    assert "reference_appended=true" in stream_log
    assert "reference_file_count=1" in stream_log
    assert "reference_location_count=2" in stream_log
    assert "reference_evidence_count=1" in stream_log
    assert f"reference_text_chars={len(stream_formatter.result.section_text)}" in stream_log
    assert "generation_stream=true" in stream_log
    _assert_generation_secrets_absent(stream_log)


def test_reference_display_failures_log_zero_aggregates_without_source_details(
    monkeypatch,
    caplog,
) -> None:
    """Formatter and gateway failures never leak prepared source text or errors."""
    formatter_failure = _PrivateReferenceFormatter(
        ReferenceFormatError("PRIVATE-FORMATTER-SECRET")
    )
    service = _generation_router(
        reference_display=True,
        formatter=formatter_failure,
    )
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "PRIVATE-QUESTION-SECRET-123"}],
                "stream": False,
            },
        )

    formatter_log = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "reference_display_failed"
    assert "reference_display_enabled=true" in formatter_log
    assert "reference_appended=false" in formatter_log
    assert "reference_file_count=0" in formatter_log
    assert "reference_location_count=0" in formatter_log
    assert "reference_evidence_count=0" in formatter_log
    assert "reference_text_chars=0" in formatter_log
    assert "PRIVATE-FORMATTER-SECRET" not in formatter_log
    _assert_generation_secrets_absent(formatter_log)

    caplog.clear()
    gateway_failure = _PrivateGatewayAnswerService(gateway_timeout_error())
    service = _generation_router(
        reference_display=True,
        formatter=_PrivateReferenceFormatter(),
        gateway=gateway_failure,
    )
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "PRIVATE-QUESTION-SECRET-123"}],
                "stream": False,
            },
        )

    gateway_log = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "gateway_timeout"
    assert "reference_display_enabled=true" in gateway_log
    assert "reference_appended=false" in gateway_log
    assert "reference_file_count=0" in gateway_log
    assert "reference_location_count=0" in gateway_log
    assert "reference_evidence_count=0" in gateway_log
    assert "reference_text_chars=0" in gateway_log
    _assert_generation_secrets_absent(gateway_log)


def test_generation_failure_logs_duration_without_private_context_or_error_detail(
    monkeypatch,
    caplog,
) -> None:
    """A builder diagnostic stays private while the request keeps safe failure timing."""
    service = _generation_router(
        builder=_PrivateContextBuilder(error=ContextBuildError("PRIVATE-BUILDER-SECRET"))
    )
    monkeypatch.setattr(routes, "get_answer_service", lambda _: service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "PRIVATE-QUESTION-SECRET-123"}],
                "stream": False,
            },
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rag_context_build_failed"
    assert "rag_route=rag_generation_unavailable" in log_text
    assert "generation_stream=false" in log_text
    assert re.search(r"generation_duration_ms=\d+\.\d{2}", log_text)
    assert "PRIVATE-BUILDER-SECRET" not in log_text
    _assert_generation_secrets_absent(log_text)


def test_generation_log_context_resets_before_the_next_request(monkeypatch, caplog) -> None:
    """A later stub request must not inherit the prior RAG request's metrics."""
    generation_service = _generation_router()
    monkeypatch.setattr(routes, "get_answer_service", lambda _: generation_service)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    app = create_app()

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "第一条私有问题"}],
                "stream": False,
            },
        )
        app.state.answer_service = StubAnswerService("local-rag")
        second = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": "第二条普通问题"}],
                "stream": False,
            },
        )

    request_logs = [
        record.getMessage()
        for record in caplog.records
        if "path=/v1/chat/completions" in record.getMessage()
    ]
    assert first.status_code == second.status_code == 200
    assert len(request_logs) == 2
    assert "context_selected_hit_count=1" in request_logs[0]
    assert "reference_display_enabled=false" in request_logs[0]
    assert "generation_stream=false" in request_logs[0]
    assert "context_selected_hit_count= " in request_logs[1]
    assert "reference_display_enabled= " in request_logs[1]
    assert "reference_appended= " in request_logs[1]
    assert "generation_stream= " in request_logs[1]
