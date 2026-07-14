"""Answer-service implementations for the local OpenAI-compatible API."""

from collections.abc import AsyncIterator
from time import perf_counter
from typing import Protocol

from local_rag_app.completion_utils import build_fixed_completion, iter_fixed_sse_events
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
from local_rag_app.rag_generation import RagGenerationService
from local_rag_app.rag_decision import RagDecisionService
from local_rag_app.retrieval import LocalRetriever, extract_latest_user_query
from local_rag_app.retrieval_models import RetrievalResult
from local_rag_app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
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
        return build_fixed_completion(self._model, STUB_ANSWER)

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[bytes]:
        """Return a byte stream containing role, content, and finish SSE events."""
        return iter_fixed_sse_events(self._model, STUB_ANSWER)


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
        rag_generation_service: RagGenerationService | None = None,
    ) -> None:
        self._settings = settings
        self._decision_service = decision_service or RagDecisionService(settings)
        self._gateway_answer_service = (
            gateway_answer_service or GatewayAnswerService(settings)
        )
        self._retriever = retriever or (
            LocalRetriever(settings) if settings.enable_local_retrieval else None
        )
        self._rag_generation_service = rag_generation_service or (
            RagGenerationService(settings)
            if settings.enable_rag_answer_generation
            else None
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

        result = await self._retrieve_before_answer_generation(request)
        if not self._settings.enable_rag_answer_generation:
            record_rag_route("retrieval_ready", "llm")
            raise rag_answer_generation_not_ready_error()
        return await self._complete_rag_answer(request, result)

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[bytes]:
        """Open SSE only after deciding that direct LLM generation is safe."""
        decision = await self._decide(request)
        if not decision.need_rag:
            record_rag_route("direct_llm", "llm")
            return await self._gateway_answer_service.stream(request)

        result = await self._retrieve_before_answer_generation(request)
        if not self._settings.enable_rag_answer_generation:
            record_rag_route("retrieval_ready", "llm")
            raise rag_answer_generation_not_ready_error()
        return await self._stream_rag_answer(request, result)

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
    ) -> RetrievalResult:
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
        return result

    async def _complete_rag_answer(
        self,
        request: ChatCompletionRequest,
        result: RetrievalResult,
    ) -> ChatCompletionResponse:
        """Generate a non-streaming RAG answer without any direct-LLM fallback."""
        generator = self._get_rag_generation_service()
        record_rag_route("rag_no_evidence" if not result.hits else "rag_generation", "llm")
        try:
            return await generator.complete(request, result)
        except LocalRagError:
            record_rag_route("rag_generation_unavailable", "llm")
            raise

    async def _stream_rag_answer(
        self,
        request: ChatCompletionRequest,
        result: RetrievalResult,
    ) -> AsyncIterator[bytes]:
        """Open RAG SSE only after all retrieval and prompt work has succeeded."""
        generator = self._get_rag_generation_service()
        record_rag_route("rag_no_evidence" if not result.hits else "rag_generation", "llm")
        try:
            return await generator.stream(request, result)
        except LocalRagError:
            record_rag_route("rag_generation_unavailable", "llm")
            raise

    def _get_rag_generation_service(self) -> RagGenerationService:
        """Keep a missing enabled generation service fail-closed and diagnosable."""
        if self._rag_generation_service is None:
            raise rag_answer_generation_not_ready_error()
        return self._rag_generation_service


def get_answer_service(settings: Settings) -> AnswerService:
    """Select the service implementation supported by the current development stage."""
    if settings.local_rag_answer_mode == "stub":
        return StubAnswerService(settings.local_rag_model)
    if settings.enable_rag_router:
        return RagRouterAnswerService(settings)
    return GatewayAnswerService(settings)
