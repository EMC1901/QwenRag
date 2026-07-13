"""Answer-service implementations for the local OpenAI-compatible API."""

from collections.abc import AsyncIterator
from time import time
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from local_rag_app.config import Settings
from local_rag_app.errors import (
    LocalRagError,
    rag_answer_generation_not_ready_error,
    rag_retrieval_not_ready_error,
)
from local_rag_app.gateway_client import ModelGatewayClient
from local_rag_app.logging_config import (
    get_request_id,
    record_rag_route,
    record_retrieval_failure,
    record_retrieval_success,
)
from local_rag_app.rag_decision import RagDecisionService
from local_rag_app.retrieval import LocalRetriever, extract_latest_user_query
from local_rag_app.schemas import (
    AssistantMessage,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkChoice,
    CompletionChoice,
    Delta,
    Usage,
)


STUB_ANSWER = "本地 RAG 应用接口已启动，当前为接口联调测试回答。"


class AnswerService(Protocol):
    """A service capable of returning full or streaming chat completions."""

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Return a complete assistant answer for a non-streaming request."""

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[bytes]:
        """Open and return a byte stream containing OpenAI-compatible SSE events."""


class StubAnswerService:
    """Deterministic answer service used before RAG and gateway logic are added."""

    def __init__(self, model: str) -> None:
        self._model = model

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Return a fixed response while preserving the local API model name."""
        completion_id, created = self._new_completion_metadata()
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=self._model,
            choices=[
                CompletionChoice(
                    message=AssistantMessage(content=STUB_ANSWER),
                )
            ],
            usage=Usage(),
        )

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[bytes]:
        """Return a byte stream containing role, content, and finish SSE events."""
        return self._iter_sse_events()

    async def _iter_sse_events(self) -> AsyncIterator[bytes]:
        """Yield role, content, and finish chunks in standard SSE order."""
        completion_id, created = self._new_completion_metadata()
        yield _encode_sse_event(ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self._model,
            choices=[ChunkChoice(delta=Delta(role="assistant"))],
        ))
        yield _encode_sse_event(ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self._model,
            choices=[ChunkChoice(delta=Delta(content=STUB_ANSWER))],
        ))
        yield _encode_sse_event(ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=self._model,
            choices=[ChunkChoice(delta=Delta(), finish_reason="stop")],
        ))
        yield b"data: [DONE]\n\n"

    @staticmethod
    def _new_completion_metadata() -> tuple[str, int]:
        return f"chatcmpl-local-{uuid4().hex}", int(time())


class GatewayAnswerService:
    """Answer service that delegates generation to the configured model gateway."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._gateway_client = ModelGatewayClient(settings)

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Return a validated gateway completion under the stable local model name."""
        payload = await self._gateway_client.complete_chat(
            request,
            local_model=self._settings.local_rag_model,
            request_id=get_request_id(),
        )
        try:
            return ChatCompletionResponse.model_validate(payload)
        except ValueError as exc:
            from local_rag_app.errors import gateway_invalid_response_error

            raise gateway_invalid_response_error() from exc

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[bytes]:
        """Open a model-gateway SSE response before the local SSE response starts."""
        return await self._gateway_client.open_chat_stream(
            request,
            local_model=self._settings.local_rag_model,
            request_id=get_request_id(),
        )


class RagRouterAnswerService:
    """Route chats to direct LLM generation or safe staged local retrieval."""

    def __init__(
        self,
        settings: Settings,
        *,
        decision_service: RagDecisionService | None = None,
        gateway_answer_service: AnswerService | None = None,
        retriever: LocalRetriever | None = None,
    ) -> None:
        self._settings = settings
        self._decision_service = decision_service or RagDecisionService(settings)
        self._gateway_answer_service = (
            gateway_answer_service or GatewayAnswerService(settings)
        )
        self._retriever = retriever or (
            LocalRetriever(settings) if settings.enable_local_retrieval else None
        )

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Return a direct answer only when the strict route does not need RAG."""
        decision = await self._decide(request)
        if not decision.need_rag:
            record_rag_route("direct_llm", "llm")
            return await self._gateway_answer_service.complete(request)

        await self._retrieve_before_answer_generation(request)
        raise rag_answer_generation_not_ready_error()

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[bytes]:
        """Open SSE only after deciding that direct LLM generation is safe."""
        decision = await self._decide(request)
        if not decision.need_rag:
            record_rag_route("direct_llm", "llm")
            return await self._gateway_answer_service.stream(request)

        await self._retrieve_before_answer_generation(request)
        raise rag_answer_generation_not_ready_error()

    async def _decide(self, request: ChatCompletionRequest):
        """Preserve a safe route marker when the decision service is unavailable."""
        try:
            return await self._decision_service.decide(request)
        except LocalRagError:
            record_rag_route("decision_unavailable")
            raise

    async def _retrieve_before_answer_generation(
        self,
        request: ChatCompletionRequest,
    ) -> None:
        """Run local retrieval or preserve the earlier fail-closed disabled behavior."""
        if not self._settings.enable_local_retrieval or self._retriever is None:
            record_rag_route("retrieval_pending", "llm")
            raise rag_retrieval_not_ready_error()
        started_at = perf_counter()
        try:
            result = await self._retriever.retrieve(extract_latest_user_query(request))
        except LocalRagError:
            record_retrieval_failure(duration_ms=(perf_counter() - started_at) * 1000)
            record_rag_route("retrieval_unavailable", "llm")
            raise
        record_retrieval_success(
            retrieval_mode=result.retrieval_mode,
            hit_count=len(result.hits),
            vector_candidate_count=result.vector_candidate_count,
            fts_candidate_count=result.fts_candidate_count,
            duration_ms=(perf_counter() - started_at) * 1000,
        )
        record_rag_route("retrieval_ready", "llm")


def _encode_sse_event(chunk: ChatCompletionChunk) -> bytes:
    """Render a local chunk using the standard SSE data framing."""
    return (
        f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")
    )


def get_answer_service(settings: Settings) -> AnswerService:
    """Select the service implementation supported by the current development stage."""
    if settings.local_rag_answer_mode == "stub":
        return StubAnswerService(settings.local_rag_model)
    if settings.enable_rag_router:
        return RagRouterAnswerService(settings)
    return GatewayAnswerService(settings)
