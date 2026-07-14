"""Offline tests for the stage-8 RAG-generation acceptance checker."""

from __future__ import annotations

import json
from types import SimpleNamespace

from local_rag_app.completion_utils import build_fixed_completion
from local_rag_app.context_builder import ContextBuildError
from local_rag_app.context_models import ContextBuildResult, SelectedContextHit
from local_rag_app.errors import gateway_timeout_error, rag_retrieval_unavailable_error
from local_rag_app.retrieval_models import RetrievalHit
from scripts.check_local_rag_generation import (
    EXIT_ARGUMENT_ERROR,
    EXIT_RUNTIME_FAILURE,
    EXIT_SUCCESS,
    run_check,
)


class FakeKnowledgeBase:
    """Record the asset preflight without touching the real knowledge base."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        if self.error is not None:
            raise self.error


class FakeRetriever:
    """Return a synthetic result or a safe expected retrieval failure."""

    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.queries: list[str] = []

    async def retrieve(self, query: str):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.result


class FakeContextBuilder:
    """Return a supplied context without constructing a private prompt."""

    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.result = result or context()
        self.error = error
        self.calls = []

    def build(self, request, retrieval_result):
        self.calls.append((request, retrieval_result))
        if self.error is not None:
            raise self.error
        return self.result


class FakeGenerationService:
    """Return deterministic normal or streaming answers without network access."""

    def __init__(self, *, answer: str = "PRIVATE_MODEL_ANSWER", error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.complete_calls = []
        self.stream_calls = []

    async def complete(self, request, retrieval_result):
        self.complete_calls.append((request, retrieval_result))
        if self.error is not None:
            raise self.error
        return build_fixed_completion("local-rag", self.answer)

    async def stream(self, request, retrieval_result):
        self.stream_calls.append((request, retrieval_result))
        if self.error is not None:
            raise self.error
        return self._events()

    async def _events(self):
        payload = json.dumps(
            {"choices": [{"delta": {"content": self.answer}}]},
            ensure_ascii=False,
        )
        yield f"data: {payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


def settings() -> SimpleNamespace:
    """Return the active settings fields inspected by the acceptance checker."""
    return SimpleNamespace(
        local_rag_model="local-rag",
        local_rag_answer_mode="gateway",
        enable_rag_router=True,
        enable_local_retrieval=True,
        enable_rag_answer_generation=True,
        model_gateway_base_url="http://gateway.test/v1",
        model_gateway_api_key="test-key",
        upstream_llm_model="llm-test",
        upstream_embedding_model="embed-test",
        rag_max_input_tokens=6144,
    )


def hit() -> RetrievalHit:
    """Build a hit with synthetic content that must be hidden by default."""
    return RetrievalHit(
        rank=1,
        chunk_id="chunk-1",
        doc_id="doc-1",
        chunk_text="PRIVATE_CONTEXT_TEXT_SHOULD_NOT_APPEAR_BY_DEFAULT",
        title="Synthetic regulation",
        section_path="Chapter 1",
        relative_path="private/full/path/source.docx",
        final_score=0.1,
        matched_by="both",
    )


def retrieval_result(*, hits=None) -> SimpleNamespace:
    """Build a minimal structured retrieval result for one acceptance query."""
    return SimpleNamespace(
        hits=hits if hits is not None else [hit()],
        retrieval_mode="hybrid",
    )


def context() -> ContextBuildResult:
    """Build a private prompt result while keeping safe aggregate metrics public."""
    selected_hit = hit()
    return ContextBuildResult(
        system_prompt="PRIVATE_SYSTEM_PROMPT",
        user_prompt="PRIVATE_USER_PROMPT",
        selected_hits=[
            SelectedContextHit(
                evidence_no=1,
                hit=selected_hit,
                text_for_prompt=selected_hit.chunk_text,
                estimated_tokens=100,
            )
        ],
        dropped_hit_count=2,
        estimated_input_tokens=500,
        estimated_context_tokens=100,
        estimated_history_tokens=0,
        history_message_count=0,
    )


def _run(*, query: str = " synthetic question ", **overrides):
    """Run a normal offline check with concise defaults for individual tests."""
    lines: list[str] = []
    values = {
        "show_answer": False,
        "show_context": False,
        "stream": False,
        "settings": settings(),
        "knowledge_base": FakeKnowledgeBase(),
        "retriever": FakeRetriever(retrieval_result()),
        "context_builder": FakeContextBuilder(),
        "generation_service": FakeGenerationService(),
        "output": lines.append,
    }
    values.update(overrides)
    return run_check(query, **values), lines, values


def test_empty_query_returns_argument_error_without_accessing_assets() -> None:
    """Whitespace is a CLI error and must not start runtime validation work."""
    knowledge_base = FakeKnowledgeBase()

    code, _, values = _run(query="   ", knowledge_base=knowledge_base)

    assert code == EXIT_ARGUMENT_ERROR
    assert knowledge_base.load_calls == 0
    assert values["retriever"].queries == []


def test_success_output_is_summary_only_by_default() -> None:
    """Default output exposes only aggregate acceptance results, never private bodies."""
    code, lines, values = _run()
    output = "\n".join(lines)

    assert code == EXIT_SUCCESS
    assert values["retriever"].queries == ["synthetic question"]
    assert "retrieval returned 1 hits in hybrid mode" in output
    assert "context selected 1 hits and dropped 2" in output
    assert "estimated input is 500 / 6144 tokens" in output
    assert "generation returned non-empty text" in output
    assert "PRIVATE_CONTEXT_TEXT_SHOULD_NOT_APPEAR_BY_DEFAULT" not in output
    assert "PRIVATE_MODEL_ANSWER" not in output
    assert "PRIVATE_SYSTEM_PROMPT" not in output
    assert "private/full/path/source.docx" not in output


def test_show_context_requires_explicit_flag_and_warns() -> None:
    """The context preview is both bounded and guarded by a privacy warning."""
    long_context = context()
    long_context.selected_hits[0].text_for_prompt = "x" * 150

    code, lines, _ = _run(
        show_context=True,
        context_builder=FakeContextBuilder(result=long_context),
    )

    preview = next(line for line in lines if line.startswith("evidence="))
    assert code == EXIT_SUCCESS
    assert any(line.startswith("[WARNING]") for line in lines)
    assert preview.endswith(f"preview={'x' * 100}")


def test_show_answer_requires_explicit_flag_and_warns() -> None:
    """Answer output has a clear warning because answers can quote private evidence."""
    code, lines, _ = _run(show_answer=True)

    assert code == EXIT_SUCCESS
    assert any(line.startswith("[WARNING]") for line in lines)
    assert "answer=PRIVATE_MODEL_ANSWER" in lines


def test_runtime_asset_retrieval_context_and_generation_failures_return_one() -> None:
    """Every expected runtime failure is concise and uses the documented code 1."""
    cases = [
        {"knowledge_base": FakeKnowledgeBase(error=RuntimeError("private asset path"))},
        {"retriever": FakeRetriever(error=rag_retrieval_unavailable_error())},
        {"context_builder": FakeContextBuilder(error=ContextBuildError("private prompt"))},
        {"generation_service": FakeGenerationService(error=gateway_timeout_error())},
    ]

    for overrides in cases:
        code, lines, _ = _run(**overrides)

        assert code == EXIT_RUNTIME_FAILURE
        assert "private asset path" not in "\n".join(lines)
        assert "private prompt" not in "\n".join(lines)


def test_streaming_check_consumes_openai_sse_and_requires_done() -> None:
    """The optional stream mode validates generation content and the SSE terminator."""
    generation = FakeGenerationService()

    code, lines, _ = _run(stream=True, generation_service=generation)

    assert code == EXIT_SUCCESS
    assert len(generation.complete_calls) == 0
    assert len(generation.stream_calls) == 1
    assert "RAG generation check passed." in lines
