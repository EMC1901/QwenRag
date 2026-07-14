"""Context-aware RAG answer generation without router integration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from local_rag_app.completion_utils import build_fixed_completion, iter_fixed_sse_events
from local_rag_app.config import Settings
from local_rag_app.context_builder import ContextBuildError, ContextBuilder
from local_rag_app.context_models import ContextBuildResult
from local_rag_app.errors import rag_context_build_error
from local_rag_app.logging_config import (
    record_context_build_success,
    record_generation_failure,
    record_generation_success,
)
from local_rag_app.retrieval_models import RetrievalResult
from local_rag_app.schemas import ChatCompletionRequest, ChatCompletionResponse


NO_EVIDENCE_ANSWER = (
    "未在当前知识库中找到足够资料来回答这个问题。"
    "请补充更具体的信息，或确认相关资料已包含在知识库中。"
)


class GatewayAnswerServiceProtocol(Protocol):
    """The existing gateway answer boundary used by RAG generation."""

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Return one completed answer from the configured model gateway."""

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[bytes]:
        """Open the already-validated upstream SSE stream."""


if TYPE_CHECKING:
    from local_rag_app.answer_service import GatewayAnswerService


class RagGenerationService:
    """Build a bounded RAG prompt and delegate generation to the existing gateway."""

    def __init__(
        self,
        settings: Settings,
        *,
        context_builder: ContextBuilder | None = None,
        gateway_answer_service: GatewayAnswerServiceProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._context_builder = context_builder or ContextBuilder(settings)
        if gateway_answer_service is None:
            # Delayed import prevents a future router integration from creating
            # an answer_service <-> rag_generation module import cycle.
            from local_rag_app.answer_service import GatewayAnswerService

            gateway_answer_service = GatewayAnswerService(settings)
        self._gateway_answer_service = gateway_answer_service

    async def complete(
        self,
        request: ChatCompletionRequest,
        retrieval_result: RetrievalResult,
    ) -> ChatCompletionResponse:
        """Return a grounded answer or a local no-evidence response."""
        started_at = perf_counter()
        try:
            if not retrieval_result.hits:
                self._record_no_evidence_context()
                response = build_fixed_completion(
                    self._settings.local_rag_model,
                    NO_EVIDENCE_ANSWER,
                )
            else:
                context = self._build_context(request, retrieval_result)
                response = await self._gateway_answer_service.complete(
                    build_generation_request(request, context, self._settings)
                )
        except Exception:
            record_generation_failure(
                stream=False,
                duration_ms=(perf_counter() - started_at) * 1000,
            )
            raise
        record_generation_success(
            stream=False,
            duration_ms=(perf_counter() - started_at) * 1000,
        )
        return response

    async def stream(
        self,
        request: ChatCompletionRequest,
        retrieval_result: RetrievalResult,
    ) -> AsyncIterator[bytes]:
        """Open SSE only after the context has been safely prepared."""
        started_at = perf_counter()
        try:
            if not retrieval_result.hits:
                self._record_no_evidence_context()
                response_stream = iter_fixed_sse_events(
                    self._settings.local_rag_model,
                    NO_EVIDENCE_ANSWER,
                )
            else:
                context = self._build_context(request, retrieval_result)
                response_stream = await self._gateway_answer_service.stream(
                    build_generation_request(request, context, self._settings)
                )
        except Exception:
            record_generation_failure(
                stream=True,
                duration_ms=(perf_counter() - started_at) * 1000,
            )
            raise
        record_generation_success(
            stream=True,
            duration_ms=(perf_counter() - started_at) * 1000,
        )
        return response_stream

    def _build_context(
        self,
        request: ChatCompletionRequest,
        retrieval_result: RetrievalResult,
    ) -> ContextBuildResult:
        """Map only expected context assembly failures to a safe public error."""
        try:
            context = self._context_builder.build(request, retrieval_result)
        except ContextBuildError as exc:
            raise rag_context_build_error() from exc
        record_context_build_success(
            selected_hit_count=len(context.selected_hits),
            dropped_hit_count=context.dropped_hit_count,
            estimated_input_tokens=context.estimated_input_tokens,
            estimated_evidence_tokens=context.estimated_context_tokens,
            estimated_history_tokens=context.estimated_history_tokens,
            truncated_hit_count=sum(item.truncated for item in context.selected_hits),
        )
        return context

    @staticmethod
    def _record_no_evidence_context() -> None:
        """Represent the intentional empty-evidence branch with safe zero metrics."""
        record_context_build_success(
            selected_hit_count=0,
            dropped_hit_count=0,
            estimated_input_tokens=0,
            estimated_evidence_tokens=0,
            estimated_history_tokens=0,
            truncated_hit_count=0,
        )


def build_generation_request(
    original: ChatCompletionRequest,
    context: ContextBuildResult,
    settings: Settings,
) -> ChatCompletionRequest:
    """Create a minimal internal request instead of forwarding client extras.

    Client-provided temperature, token limits, tools, and other unknown options
    are intentionally omitted.  The local service owns all RAG prompt and
    generation limits.
    """
    return ChatCompletionRequest.model_validate(
        {
            "model": settings.local_rag_model,
            "messages": [
                {"role": "system", "content": context.system_prompt},
                {"role": "user", "content": context.user_prompt},
            ],
            "stream": original.stream,
            "temperature": settings.rag_generation_temperature,
            "max_tokens": settings.rag_max_output_tokens,
        }
    )
