"""Context-aware RAG answer generation without router integration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from local_rag_app.completion_utils import build_fixed_completion, iter_fixed_sse_events
from local_rag_app.config import Settings
from local_rag_app.context_builder import ContextBuildError, ContextBuilder
from local_rag_app.context_models import ContextBuildResult, SelectedContextHit
from local_rag_app.errors import rag_context_build_error, reference_display_error
from local_rag_app.logging_config import (
    record_context_build_success,
    record_generation_failure,
    record_generation_success,
    record_reference_display,
    record_rag_route,
)
from local_rag_app.reference_appender import (
    append_references_to_completion,
    append_references_to_sse_stream,
)
from local_rag_app.reference_formatter import ReferenceFormatError, ReferenceFormatter
from local_rag_app.reference_models import ReferenceBuildResult
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


class ReferenceFormatterProtocol(Protocol):
    """The pure source-formatting boundary used before a RAG response is opened."""

    def build(
        self,
        selected_hits: list[SelectedContextHit],
    ) -> ReferenceBuildResult:
        """Build a deterministic source section from actual prompt evidence."""


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
        reference_formatter: ReferenceFormatterProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._context_builder = context_builder or ContextBuilder(settings)
        self._reference_formatter = reference_formatter or ReferenceFormatter()
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
        self._record_unappended_references()
        try:
            if not retrieval_result.hits:
                response = self._no_evidence_completion(record_empty_context=True)
            else:
                context = self._build_context(request, retrieval_result)
                if not context.selected_hits:
                    response = self._no_evidence_completion(record_empty_context=False)
                else:
                    reference = self._build_reference(context)
                    response = await self._gateway_answer_service.complete(
                        build_generation_request(request, context, self._settings)
                    )
                    if reference is not None:
                        response = append_references_to_completion(response, reference)
                        self._record_appended_references(context, reference)
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
        self._record_unappended_references()
        try:
            if not retrieval_result.hits:
                response_stream = self._no_evidence_stream(record_empty_context=True)
            else:
                context = self._build_context(request, retrieval_result)
                if not context.selected_hits:
                    response_stream = self._no_evidence_stream(record_empty_context=False)
                else:
                    reference = self._build_reference(context)
                    response_stream = await self._gateway_answer_service.stream(
                        build_generation_request(request, context, self._settings)
                    )
                    if reference is not None:
                        response_stream = append_references_to_sse_stream(
                            response_stream,
                            reference,
                        )
                        self._record_appended_references(context, reference)
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

    def _build_reference(
        self,
        context: ContextBuildResult,
    ) -> ReferenceBuildResult | None:
        """Prepare references before opening the model gateway or local SSE response."""
        if not self._settings.enable_reference_display:
            return None
        try:
            return self._reference_formatter.build(context.selected_hits)
        except ReferenceFormatError as exc:
            raise reference_display_error() from exc

    def _record_unappended_references(self) -> None:
        """Initialize safe zero-valued reference metrics for every RAG request."""
        record_reference_display(
            enabled=self._settings.enable_reference_display,
            appended=False,
            file_count=0,
            location_count=0,
            evidence_count=0,
            text_chars=0,
        )

    @staticmethod
    def _record_appended_references(
        context: ContextBuildResult,
        reference: ReferenceBuildResult,
    ) -> None:
        """Record only counts after a source block joins the returned response."""
        record_reference_display(
            enabled=True,
            appended=True,
            file_count=len(reference.files),
            location_count=sum(len(item.locations) for item in reference.files),
            evidence_count=len(context.selected_hits),
            text_chars=len(reference.section_text),
        )

    def _no_evidence_completion(
        self,
        *,
        record_empty_context: bool,
    ) -> ChatCompletionResponse:
        """Return a fixed answer when retrieval has no prompt-usable evidence."""
        if record_empty_context:
            self._record_no_evidence_context()
        record_rag_route("rag_no_evidence", "llm")
        return build_fixed_completion(
            self._settings.local_rag_model,
            NO_EVIDENCE_ANSWER,
        )

    def _no_evidence_stream(
        self,
        *,
        record_empty_context: bool,
    ) -> AsyncIterator[bytes]:
        """Return fixed SSE only when no evidence is available to the model."""
        if record_empty_context:
            self._record_no_evidence_context()
        record_rag_route("rag_no_evidence", "llm")
        return iter_fixed_sse_events(
            self._settings.local_rag_model,
            NO_EVIDENCE_ANSWER,
        )

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
