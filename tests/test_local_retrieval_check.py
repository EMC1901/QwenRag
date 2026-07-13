"""Unit tests for the stage-10 standalone local-retrieval acceptance checker."""

from types import SimpleNamespace

from local_rag_app.errors import rag_retrieval_unavailable_error
from scripts.check_local_retrieval import (
    EXIT_ARGUMENT_ERROR,
    EXIT_RUNTIME_FAILURE,
    EXIT_SUCCESS,
    run_check,
)


class FakeKnowledgeBase:
    """Record the asset preflight performed by the acceptance checker."""

    def __init__(self) -> None:
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1


class FakeRetriever:
    """Return a supplied outcome without accessing a gateway, FAISS, or SQLite."""

    def __init__(self, result=None, error=None) -> None:
        self.result = result
        self.error = error
        self.queries: list[str] = []

    async def retrieve(self, query: str):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.result


def settings() -> SimpleNamespace:
    """Return only the gateway fields the standalone checker needs to inspect."""
    return SimpleNamespace(
        model_gateway_base_url="http://gateway.test/v1",
        model_gateway_api_key="test-key",
        upstream_embedding_model="embed-test",
    )


def result(*, hits) -> SimpleNamespace:
    """Build a minimal structured retrieval result with synthetic private source text."""
    return SimpleNamespace(
        embedding_dim=3,
        vector_candidate_count=4,
        retrieval_mode="hybrid",
        hits=hits,
    )


def hit() -> SimpleNamespace:
    """Build one source hit whose private body must stay out of default output."""
    return SimpleNamespace(
        rank=1,
        title="Synthetic regulation",
        doc_title="Synthetic document",
        relative_path="private/full/path/source.docx",
        section_path="Chapter 1",
        article_range=None,
        article_no="Article 1",
        final_score=0.0159,
        matched_by="both",
        chunk_text="PRIVATE_CHUNK_TEXT_SHOULD_NOT_APPEAR_BY_DEFAULT",
    )


def test_empty_query_returns_argument_error_without_accessing_assets() -> None:
    """Whitespace input has documented exit code 2 and avoids all runtime work."""
    knowledge_base = FakeKnowledgeBase()
    retriever = FakeRetriever()
    lines: list[str] = []

    code = run_check(
        "   ",
        show_preview=False,
        settings=settings(),
        knowledge_base=knowledge_base,
        retriever=retriever,
        output=lines.append,
    )

    assert code == EXIT_ARGUMENT_ERROR
    assert knowledge_base.load_calls == 0
    assert retriever.queries == []


def test_success_output_is_summary_only_by_default() -> None:
    """The normal acceptance output has metrics and source fields, but no body or path."""
    knowledge_base = FakeKnowledgeBase()
    retriever = FakeRetriever(result(hits=[hit()]))
    lines: list[str] = []

    code = run_check(
        "  synthetic query  ",
        show_preview=False,
        settings=settings(),
        knowledge_base=knowledge_base,
        retriever=retriever,
        output=lines.append,
    )

    output = "\n".join(lines)
    assert code == EXIT_SUCCESS
    assert knowledge_base.load_calls == 1
    assert retriever.queries == ["synthetic query"]
    assert "embedding model returned 3 dimensions" in output
    assert "vector search returned 4 candidates" in output
    assert "retrieval returned 1 hits in hybrid mode" in output
    assert "PRIVATE_CHUNK_TEXT_SHOULD_NOT_APPEAR_BY_DEFAULT" not in output
    assert "private/full/path/source.docx" not in output


def test_preview_requires_explicit_flag_and_is_bounded() -> None:
    """Preview output warns about private data and never emits more than 100 characters."""
    source_hit = hit()
    source_hit.chunk_text = "x" * 150
    lines: list[str] = []

    code = run_check(
        "synthetic query",
        show_preview=True,
        settings=settings(),
        knowledge_base=FakeKnowledgeBase(),
        retriever=FakeRetriever(result(hits=[source_hit])),
        output=lines.append,
    )

    preview_line = next(line for line in lines if line.startswith("   preview="))
    assert code == EXIT_SUCCESS
    assert any(line.startswith("[WARNING]") for line in lines)
    assert preview_line == f"   preview={'x' * 100}"


def test_runtime_retrieval_failure_uses_exit_code_one_without_error_details() -> None:
    """Expected gateway/retrieval failures are concise and never print exception contents."""
    lines: list[str] = []

    code = run_check(
        "synthetic query",
        show_preview=False,
        settings=settings(),
        knowledge_base=FakeKnowledgeBase(),
        retriever=FakeRetriever(error=rag_retrieval_unavailable_error()),
        output=lines.append,
    )

    output = "\n".join(lines)
    assert code == EXIT_RUNTIME_FAILURE
    assert "embedding or local retrieval failed" in output
    assert "rag_retrieval_unavailable" not in output
