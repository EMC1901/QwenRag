"""Offline tests for stage-7 context-aware answer generation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json

import pytest

from local_rag_app.completion_utils import build_fixed_completion
from local_rag_app.config import Settings
from local_rag_app.context_builder import ContextBuildError
from local_rag_app.context_models import ContextBuildResult, SelectedContextHit
from local_rag_app.errors import LocalRagError, gateway_timeout_error
from local_rag_app.rag_generation import (
    NO_EVIDENCE_ANSWER,
    RagGenerationService,
    build_generation_request,
)
from local_rag_app.reference_formatter import ReferenceFormatError
from local_rag_app.reference_models import ReferenceBuildResult, ReferenceFile
from local_rag_app.retrieval_models import RetrievalHit, RetrievalResult
from local_rag_app.schemas import ChatCompletionRequest, ChatCompletionResponse


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)


def _reference_settings(**overrides: object) -> Settings:
    """Create valid settings with every prerequisite for visible references enabled."""
    values: dict[str, object] = {
        "LOCAL_RAG_ANSWER_MODE": "gateway",
        "ENABLE_RAG_ROUTER": "true",
        "ENABLE_LOCAL_RETRIEVAL": "true",
        "ENABLE_RAG_ANSWER_GENERATION": "true",
        "ENABLE_REFERENCE_DISPLAY": "true",
        "MODEL_GATEWAY_BASE_URL": "http://gateway.test:8010/v1",
        "MODEL_GATEWAY_API_KEY": "test-key",
        "UPSTREAM_LLM_MODEL": "qwen",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def _request(*, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "client-supplied-model",
            "messages": [{"role": "user", "content": "客户问题"}],
            "stream": stream,
            "temperature": 1.7,
            "max_tokens": 99999,
            "tools": [{"type": "function", "function": {"name": "not_forwarded"}}],
        }
    )


def _hit() -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        chunk_id="chunk-1",
        doc_id="doc-1",
        chunk_text="检索资料正文",
        title="测试资料",
        relative_path="fixtures/test.docx",
        final_score=0.8,
        matched_by="both",
    )


def _result(*, hits: list[RetrievalHit] | None = None) -> RetrievalResult:
    values = hits if hits is not None else [_hit()]
    return RetrievalResult(
        hits=values,
        candidate_count=len(values),
        vector_candidate_count=len(values),
        fts_candidate_count=len(values),
        embedding_model="embed-test",
        embedding_dim=1024,
        retrieval_mode="hybrid",
    )


def _context(*, suffix: str = "") -> ContextBuildResult:
    hit = _hit()
    return ContextBuildResult(
        system_prompt=f"本地系统提示词{suffix}",
        user_prompt=f"本地用户提示词{suffix}",
        selected_hits=[
            SelectedContextHit(
                evidence_no=1,
                hit=hit,
                text_for_prompt=hit.chunk_text,
                estimated_tokens=10,
            )
        ],
        dropped_hit_count=0,
        estimated_input_tokens=100,
        estimated_context_tokens=10,
        estimated_history_tokens=0,
        history_message_count=0,
    )


def _empty_context() -> ContextBuildResult:
    """Represent retrieval hits that could not safely fit into the final prompt."""
    return ContextBuildResult(
        system_prompt="本地系统提示词",
        user_prompt="本地用户提示词",
        selected_hits=[],
        dropped_hit_count=1,
        estimated_input_tokens=100,
        estimated_context_tokens=0,
        estimated_history_tokens=0,
        history_message_count=0,
    )


def _reference() -> ReferenceBuildResult:
    return ReferenceBuildResult(
        section_text=(
            "参考文件：\n"
            "[1] 测试资料.docx\n"
            "    位置：第一章 / 段落 1-3\n"
            "    对应资料：[资料1]"
        ),
        files=[
            ReferenceFile(
                reference_no=1,
                display_name="测试资料.docx",
                locations=["第一章 / 段落 1-3"],
                evidence_nos=[1],
            )
        ],
        selected_hit_count=1,
        location_count=1,
    )


class FakeContextBuilder:
    """Capture context construction without calling the real pure builder."""

    def __init__(
        self,
        *,
        result: ContextBuildResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or _context()
        self.error = error
        self.calls: list[tuple[ChatCompletionRequest, RetrievalResult]] = []

    def build(
        self,
        request: ChatCompletionRequest,
        retrieval_result: RetrievalResult,
    ) -> ContextBuildResult:
        self.calls.append((request, retrieval_result))
        if self.error is not None:
            raise self.error
        return self.result


class EchoContextBuilder:
    """Make concurrent calls observable without sharing prompt state."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def build(
        self,
        request: ChatCompletionRequest,
        retrieval_result: RetrievalResult,
    ) -> ContextBuildResult:
        question = request.messages[-1].content
        self.calls.append(question)
        return _context(suffix=f":{question}")


class FakeReferenceFormatter:
    """Capture source formatting without depending on display text internals."""

    def __init__(
        self,
        *,
        result: ReferenceBuildResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or _reference()
        self.error = error
        self.calls: list[list[SelectedContextHit]] = []

    def build(self, selected_hits: list[SelectedContextHit]) -> ReferenceBuildResult:
        self.calls.append(selected_hits)
        if self.error is not None:
            raise self.error
        return self.result


class FakeGatewayAnswerService:
    """Record generation requests and provide deterministic normal/SSE responses."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.complete_requests: list[ChatCompletionRequest] = []
        self.stream_requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.complete_requests.append(request)
        if self.error is not None:
            raise self.error
        return build_fixed_completion("local-rag", request.messages[-1].content)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[bytes]:
        self.stream_requests.append(request)
        if self.error is not None:
            raise self.error
        return self._events(request.messages[-1].content)

    async def _events(self, content: str) -> AsyncIterator[bytes]:
        metadata = {"id": "chatcmpl-test", "created": 1, "model": "local-rag"}
        for choice in (
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
            {"index": 0, "delta": {"content": content}, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ):
            yield (
                f"data: {json.dumps({**metadata, 'choices': [choice]}, ensure_ascii=False)}\n\n"
            ).encode("utf-8")
        yield b"data: [DONE]\n\n"


def test_build_generation_request_uses_only_controlled_fields() -> None:
    """Client extras cannot bypass local RAG model and generation limits."""
    original = _request(stream=True)
    settings = _settings(RAG_GENERATION_TEMPERATURE=0.3, RAG_MAX_OUTPUT_TOKENS=321)

    generated = build_generation_request(original, _context(), settings)

    assert generated.model == "local-rag"
    assert generated.stream is True
    assert [message.role for message in generated.messages] == ["system", "user"]
    assert [message.content for message in generated.messages] == [
        "本地系统提示词",
        "本地用户提示词",
    ]
    assert generated.model_extra == {"temperature": 0.3, "max_tokens": 321}


@pytest.mark.asyncio
async def test_complete_builds_context_once_and_delegates_a_safe_request() -> None:
    """A non-empty retrieval result becomes exactly one controlled gateway call."""
    builder = FakeContextBuilder()
    gateway = FakeGatewayAnswerService()
    service = RagGenerationService(
        _settings(),
        context_builder=builder,  # type: ignore[arg-type]
        gateway_answer_service=gateway,
    )
    original = _request()
    before = original.model_dump()

    response = await service.complete(original, _result())

    assert response.model == "local-rag"
    assert response.choices[0].message.content == "本地用户提示词"
    assert len(builder.calls) == 1
    assert len(gateway.complete_requests) == 1
    assert gateway.stream_requests == []
    assert gateway.complete_requests[0].model_extra == {
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    assert original.model_dump() == before


@pytest.mark.asyncio
async def test_stream_builds_context_before_opening_gateway_sse() -> None:
    """The gateway is only opened after a prompt has been prepared successfully."""
    builder = FakeContextBuilder()
    gateway = FakeGatewayAnswerService()
    service = RagGenerationService(
        _settings(),
        context_builder=builder,  # type: ignore[arg-type]
        gateway_answer_service=gateway,
    )

    stream = await service.stream(_request(stream=True), _result())
    body = b"".join([event async for event in stream])

    assert len(builder.calls) == 1
    assert len(gateway.stream_requests) == 1
    assert gateway.complete_requests == []
    assert body.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_no_evidence_returns_local_answer_without_builder_or_gateway() -> None:
    """A private-data query with no evidence must not hallucinate through the LLM."""
    builder = FakeContextBuilder()
    gateway = FakeGatewayAnswerService()
    service = RagGenerationService(
        _settings(),
        context_builder=builder,  # type: ignore[arg-type]
        gateway_answer_service=gateway,
    )

    response = await service.complete(_request(), _result(hits=[]))
    stream = await service.stream(_request(stream=True), _result(hits=[]))
    body = b"".join([event async for event in stream]).decode("utf-8")

    assert response.choices[0].message.content == NO_EVIDENCE_ANSWER
    assert response.usage.total_tokens == 0
    assert NO_EVIDENCE_ANSWER in body
    assert body.endswith("data: [DONE]\n\n")
    assert builder.calls == []
    assert gateway.complete_requests == []
    assert gateway.stream_requests == []


@pytest.mark.asyncio
async def test_selected_evidence_appends_references_from_actual_prompt_hits() -> None:
    """Only prompt-usable hits become the visible source section."""
    context = _context()
    formatter = FakeReferenceFormatter()
    gateway = FakeGatewayAnswerService()
    service = RagGenerationService(
        _reference_settings(),
        context_builder=FakeContextBuilder(result=context),  # type: ignore[arg-type]
        gateway_answer_service=gateway,
        reference_formatter=formatter,
    )

    response = await service.complete(_request(), _result())

    assert response.choices[0].message.content.endswith(_reference().section_text)
    assert formatter.calls == [context.selected_hits]
    assert len(gateway.complete_requests) == 1


@pytest.mark.asyncio
async def test_selected_evidence_inserts_references_before_sse_finish_and_done() -> None:
    """The reference delta is part of the answer and precedes stream termination."""
    service = RagGenerationService(
        _reference_settings(),
        context_builder=FakeContextBuilder(),  # type: ignore[arg-type]
        gateway_answer_service=FakeGatewayAnswerService(),
        reference_formatter=FakeReferenceFormatter(),
    )

    stream = await service.stream(_request(stream=True), _result())
    events = [event async for event in stream]
    payloads = [
        json.loads(event.decode("utf-8").removeprefix("data: "))
        if event != b"data: [DONE]\n\n"
        else None
        for event in events
    ]
    reference_index = next(
        index
        for index, payload in enumerate(payloads)
        if payload is not None
        and _reference().section_text in payload["choices"][0]["delta"].get("content", "")
    )
    finish_index = next(
        index
        for index, payload in enumerate(payloads)
        if payload is not None and payload["choices"][0]["finish_reason"] == "stop"
    )

    assert reference_index < finish_index < len(events) - 1
    assert events[-1] == b"data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_normal_and_streamed_answers_reconstruct_to_the_same_text() -> None:
    """The two OpenAI response modes expose the same answer and source block."""
    formatter = FakeReferenceFormatter()
    gateway = FakeGatewayAnswerService()
    service = RagGenerationService(
        _reference_settings(),
        context_builder=FakeContextBuilder(),  # type: ignore[arg-type]
        gateway_answer_service=gateway,
        reference_formatter=formatter,
    )

    completion = await service.complete(_request(), _result())
    stream = await service.stream(_request(stream=True), _result())
    events = [event async for event in stream]
    reconstructed = "".join(
        json.loads(event.decode("utf-8").removeprefix("data: "))["choices"][0][
            "delta"
        ].get("content", "")
        for event in events
        if event != b"data: [DONE]\n\n"
    )

    assert reconstructed == completion.choices[0].message.content
    assert formatter.calls == [_context().selected_hits, _context().selected_hits]


@pytest.mark.asyncio
async def test_reference_format_error_is_safe_and_prevents_gateway_call() -> None:
    """Formatter internals never leak and an unsafe source block is never emitted."""
    gateway = FakeGatewayAnswerService()
    service = RagGenerationService(
        _reference_settings(),
        context_builder=FakeContextBuilder(),  # type: ignore[arg-type]
        gateway_answer_service=gateway,
        reference_formatter=FakeReferenceFormatter(
            error=ReferenceFormatError("private formatter diagnostic")
        ),
    )

    with pytest.raises(LocalRagError) as error:
        await service.complete(_request(), _result())

    assert error.value.code == "reference_display_failed"
    assert "private formatter" not in error.value.message
    assert gateway.complete_requests == []


@pytest.mark.asyncio
async def test_context_without_selected_hits_returns_no_evidence_without_gateway() -> None:
    """Retrieved candidates that do not fit the prompt must not trigger a model call."""
    builder = FakeContextBuilder(result=_empty_context())
    formatter = FakeReferenceFormatter()
    gateway = FakeGatewayAnswerService()
    service = RagGenerationService(
        _reference_settings(),
        context_builder=builder,  # type: ignore[arg-type]
        gateway_answer_service=gateway,
        reference_formatter=formatter,
    )

    response = await service.complete(_request(), _result())
    stream = await service.stream(_request(stream=True), _result())
    body = b"".join([event async for event in stream]).decode("utf-8")

    assert response.choices[0].message.content == NO_EVIDENCE_ANSWER
    assert NO_EVIDENCE_ANSWER in body
    assert formatter.calls == []
    assert gateway.complete_requests == []
    assert gateway.stream_requests == []


@pytest.mark.asyncio
async def test_disabled_reference_display_does_not_format_or_append_sources() -> None:
    """The feature flag preserves the pre-reference RAG response path."""
    formatter = FakeReferenceFormatter()
    service = RagGenerationService(
        _settings(),
        context_builder=FakeContextBuilder(),  # type: ignore[arg-type]
        gateway_answer_service=FakeGatewayAnswerService(),
        reference_formatter=formatter,
    )

    response = await service.complete(_request(), _result())

    assert formatter.calls == []
    assert _reference().section_text not in response.choices[0].message.content


@pytest.mark.asyncio
async def test_context_build_error_maps_to_the_stable_safe_error() -> None:
    """Internal prompt details must not become an API error message."""
    service = RagGenerationService(
        _settings(),
        context_builder=FakeContextBuilder(
            error=ContextBuildError("private prompt length diagnostic")
        ),  # type: ignore[arg-type]
        gateway_answer_service=FakeGatewayAnswerService(),
    )

    with pytest.raises(LocalRagError) as error:
        await service.complete(_request(), _result())

    assert error.value.code == "rag_context_build_failed"
    assert "private prompt" not in error.value.message


@pytest.mark.asyncio
async def test_gateway_errors_are_not_rewritten_as_a_generic_rag_answer() -> None:
    """Existing upstream codes remain useful to the route and operations logs."""
    service = RagGenerationService(
        _settings(),
        context_builder=FakeContextBuilder(),  # type: ignore[arg-type]
        gateway_answer_service=FakeGatewayAnswerService(error=gateway_timeout_error()),
    )

    with pytest.raises(LocalRagError) as error:
        await service.complete(_request(), _result())

    assert error.value.code == "gateway_timeout"


@pytest.mark.asyncio
async def test_concurrent_generation_requests_do_not_share_context() -> None:
    """Each request carries only its own built user prompt to the gateway."""
    builder = EchoContextBuilder()
    gateway = FakeGatewayAnswerService()
    service = RagGenerationService(
        _settings(),
        context_builder=builder,  # type: ignore[arg-type]
        gateway_answer_service=gateway,
    )
    first_request = ChatCompletionRequest.model_validate(
        {"model": "local-rag", "messages": [{"role": "user", "content": "问题一"}]}
    )
    second_request = ChatCompletionRequest.model_validate(
        {"model": "local-rag", "messages": [{"role": "user", "content": "问题二"}]}
    )

    first, second = await asyncio.gather(
        service.complete(first_request, _result()),
        service.complete(second_request, _result()),
    )

    assert builder.calls == ["问题一", "问题二"]
    assert first.choices[0].message.content == "本地用户提示词:问题一"
    assert second.choices[0].message.content == "本地用户提示词:问题二"
