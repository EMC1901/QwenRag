"""Tests for stage-6 pure ranking and stage-7 retrieval orchestration."""

import asyncio

import pytest

from local_rag_app.config import Settings
from local_rag_app.errors import gateway_connection_error
from local_rag_app.knowledge_base import (
    ChunkMetadata,
    FtsSearchCandidate,
    FtsSearchFallbackError,
    KnowledgeBaseLoadError,
    KnowledgeBaseQueryError,
    VectorSearchHit,
)
from local_rag_app.retrieval import (
    LocalRetriever,
    RetrievalFusionError,
    fuse_candidates,
    select_final_candidates,
)
from local_rag_app.retrieval_models import RankedCandidate


def vector_hit(
    chunk_id: str,
    *,
    rank: int,
    score: float,
    vector_id: int,
    doc_id: str = "doc-1",
) -> VectorSearchHit:
    """Create a minimal valid stage-4 hit without any database or FAISS dependency."""
    return VectorSearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_text=f"text for {chunk_id}",
        title=None,
        doc_title=None,
        section_path=None,
        article_no=None,
        article_range=None,
        relative_path=f"fixtures/{doc_id}.docx",
        paragraph_start=None,
        paragraph_end=None,
        vector_id=vector_id,
        vector_score=score,
        vector_rank=rank,
    )


def fts_hit(chunk_id: str, *, rank: int, score: float = -1.0) -> FtsSearchCandidate:
    """Create a minimal valid stage-5 FTS candidate."""
    return FtsSearchCandidate(chunk_id=chunk_id, bm25_score=score, fts_rank=rank)


def test_vector_only_candidates_keep_vector_order_and_metadata() -> None:
    """A vector-only retrieval remains usable without any keyword candidates."""
    fused = fuse_candidates(
        [vector_hit("chunk-1", rank=1, score=0.6, vector_id=10), vector_hit("chunk-2", rank=2, score=0.9, vector_id=11)],
        [],
        rrf_k=60,
        vector_weight=0.6,
        fts_weight=0.4,
    )

    assert [candidate.chunk_id for candidate in fused] == ["chunk-1", "chunk-2"]
    assert [candidate.matched_by for candidate in fused] == ["vector", "vector"]
    assert [candidate.vector_rank for candidate in fused] == [1, 2]
    assert all(candidate.fts_rank is None for candidate in fused)


def test_fts_only_candidates_are_ranked_without_using_bm25_direction() -> None:
    """FTS-only candidates use FTS rank, not the raw (lower-is-better) BM25 score."""
    fused = fuse_candidates(
        [],
        [fts_hit("chunk-2", rank=2, score=-100), fts_hit("chunk-1", rank=1, score=-0.01)],
        rrf_k=60,
        vector_weight=0.6,
        fts_weight=0.4,
    )

    assert [candidate.chunk_id for candidate in fused] == ["chunk-1", "chunk-2"]
    assert [candidate.matched_by for candidate in fused] == ["fts", "fts"]
    assert [candidate.vector_score for candidate in fused] == [None, None]


def test_rrf_fuses_overlapping_candidates_once_and_adds_rank_contributions() -> None:
    """One chunk matching both paths is de-duplicated and gets both RRF terms."""
    fused = fuse_candidates(
        [vector_hit("chunk-1", rank=1, score=0.7, vector_id=10), vector_hit("chunk-2", rank=2, score=0.8, vector_id=11)],
        [fts_hit("chunk-2", rank=1), fts_hit("chunk-3", rank=2)],
        rrf_k=60,
        vector_weight=0.6,
        fts_weight=0.4,
    )

    assert [candidate.chunk_id for candidate in fused] == ["chunk-2", "chunk-1", "chunk-3"]
    both = fused[0]
    assert both.matched_by == "both"
    assert both.final_score == pytest.approx(0.6 / 62 + 0.4 / 61)
    assert both.vector_id == 11
    assert both.vector_rank == 2
    assert both.fts_rank == 1


def test_tied_rrf_scores_use_vector_score_then_chunk_id_stably() -> None:
    """Equal scores have deterministic ordering instead of depending on input order."""
    fused = fuse_candidates(
        [vector_hit("chunk-z", rank=2, score=0.3, vector_id=12)],
        [fts_hit("chunk-a", rank=1)],
        rrf_k=60,
        vector_weight=62,
        fts_weight=61,
    )

    assert [candidate.chunk_id for candidate in fused] == ["chunk-z", "chunk-a"]
    assert [candidate.final_score for candidate in fused] == pytest.approx([1.0, 1.0])


def test_complete_rrf_tie_uses_chunk_id_as_the_final_stable_tiebreaker() -> None:
    """Equal score, source, and vector score never leave result order to dict order."""
    fused = fuse_candidates(
        [
            vector_hit("chunk-z", rank=1, score=0.5, vector_id=10),
            vector_hit("chunk-a", rank=2, score=0.5, vector_id=11),
        ],
        [fts_hit("chunk-a", rank=1), fts_hit("chunk-z", rank=2)],
        rrf_k=60,
        vector_weight=1.0,
        fts_weight=1.0,
    )

    assert [candidate.chunk_id for candidate in fused] == ["chunk-a", "chunk-z"]
    assert [candidate.final_score for candidate in fused] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 61 + 1 / 62]
    )


def test_document_cap_and_final_limit_preserve_fused_order() -> None:
    """A single long document cannot occupy every final retrieval slot."""
    candidates = [
        RankedCandidate("chunk-1", final_score=0.9, matched_by="vector"),
        RankedCandidate("chunk-2", final_score=0.8, matched_by="vector"),
        RankedCandidate("chunk-3", final_score=0.7, matched_by="fts"),
        RankedCandidate("chunk-4", final_score=0.6, matched_by="fts"),
    ]
    selected = select_final_candidates(
        candidates,
        document_ids={
            "chunk-1": "doc-a",
            "chunk-2": "doc-a",
            "chunk-3": "doc-b",
            "chunk-4": "doc-c",
        },
        final_top_k=3,
        max_chunks_per_doc=1,
    )

    assert [candidate.chunk_id for candidate in selected] == [
        "chunk-1",
        "chunk-3",
        "chunk-4",
    ]


def test_selection_returns_actual_count_when_candidates_are_fewer_than_limit() -> None:
    """Selection never invents candidates to fill a requested final top-k."""
    selected = select_final_candidates(
        [RankedCandidate("chunk-1", final_score=0.1, matched_by="fts")],
        document_ids={"chunk-1": "doc-a"},
        final_top_k=8,
        max_chunks_per_doc=3,
    )

    assert [candidate.chunk_id for candidate in selected] == ["chunk-1"]


@pytest.mark.parametrize(
    "vector_candidates, fts_candidates",
    [
        ([vector_hit("chunk-1", rank=1, score=0.8, vector_id=1), vector_hit("chunk-1", rank=2, score=0.7, vector_id=2)], []),
        ([], [fts_hit("chunk-1", rank=1), fts_hit("chunk-2", rank=1)]),
    ],
)
def test_fusion_rejects_inconsistent_per_source_candidates(
    vector_candidates: list[VectorSearchHit],
    fts_candidates: list[FtsSearchCandidate],
) -> None:
    """Duplicate IDs or ranks cannot silently distort reciprocal-rank scoring."""
    with pytest.raises(RetrievalFusionError):
        fuse_candidates(
            vector_candidates,
            fts_candidates,
            rrf_k=60,
            vector_weight=0.6,
            fts_weight=0.4,
        )


def test_selection_rejects_missing_document_mapping() -> None:
    """Document limiting cannot proceed if a candidate cannot be traced to a document."""
    with pytest.raises(RetrievalFusionError):
        select_final_candidates(
            [RankedCandidate("chunk-1", final_score=0.1, matched_by="vector")],
            document_ids={},
            final_top_k=1,
            max_chunks_per_doc=1,
        )


def retrieval_settings(**overrides: object) -> Settings:
    """Create a small, no-I/O retrieval configuration for orchestrator tests."""
    values = {
        "UPSTREAM_EMBEDDING_MODEL": "embed-test",
        "RAG_EMBEDDING_DIM": 3,
        "RAG_VECTOR_TOP_K": 3,
        "RAG_FTS_TOP_K": 3,
        "RAG_FINAL_TOP_K": 2,
        "RAG_MAX_CHUNKS_PER_DOC": 1,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def chunk_metadata(chunk_id: str, *, doc_id: str, vector_id: int | None) -> ChunkMetadata:
    """Create one complete source record used by a fake read-only knowledge base."""
    return ChunkMetadata(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_text=f"text for {chunk_id}",
        title=f"title for {chunk_id}",
        doc_title=f"document {doc_id}",
        section_path="section 1",
        article_no="article 1",
        article_range=None,
        relative_path=f"fixtures/{doc_id}.docx",
        paragraph_start=1,
        paragraph_end=1,
        vector_id=vector_id,
    )


class FakeGateway:
    """Record query embedding calls without requiring a model gateway."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    async def create_embedding(self, text: str, *, request_id: str | None = None) -> list[float]:
        self.calls.append((text, request_id))
        if self.error is not None:
            raise self.error
        return [1.0 if text == "first" else 2.0, 0.0, 0.0]


class FakeKnowledgeBase:
    """Synchronous fake that exposes exactly the stage-7 local retrieval boundary."""

    def __init__(
        self,
        *,
        load_error: Exception | None = None,
        vector_error: Exception | None = None,
        fts_error: Exception | None = None,
    ) -> None:
        self.load_error = load_error
        self.vector_error = vector_error
        self.fts_error = fts_error
        self.load_calls = 0
        self.vector_calls: list[tuple[list[float], int]] = []
        self.fts_calls: list[tuple[str, int]] = []
        self.metadata_calls: list[list[str]] = []
        self.metadata = {
            "chunk-1": chunk_metadata("chunk-1", doc_id="doc-1", vector_id=10),
            "chunk-2": chunk_metadata("chunk-2", doc_id="doc-2", vector_id=11),
        }

    def load(self) -> None:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error

    def search_vector(self, embedding: list[float], top_k: int) -> list[VectorSearchHit]:
        self.vector_calls.append((embedding, top_k))
        if self.vector_error is not None:
            raise self.vector_error
        if embedding[0] == 1.0:
            return [vector_hit("chunk-1", rank=1, score=0.8, vector_id=10)]
        return [vector_hit("chunk-2", rank=1, score=0.8, vector_id=11, doc_id="doc-2")]

    def search_fts(self, query: str, top_k: int) -> list[FtsSearchCandidate]:
        self.fts_calls.append((query, top_k))
        if self.fts_error is not None:
            raise self.fts_error
        return [fts_hit("chunk-1", rank=1)] if query == "first" else []

    def load_chunk_metadata(self, chunk_ids: list[str]) -> dict[str, ChunkMetadata]:
        self.metadata_calls.append(chunk_ids)
        return {chunk_id: self.metadata[chunk_id] for chunk_id in chunk_ids}


@pytest.mark.asyncio
async def test_local_retriever_returns_structured_hybrid_result_and_safe_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One valid query uses injected dependencies and exposes no query field in output."""
    gateway = FakeGateway()
    knowledge_base = FakeKnowledgeBase()
    monkeypatch.setattr("local_rag_app.retrieval.get_request_id", lambda: "request-7")
    retriever = LocalRetriever(
        retrieval_settings(),
        gateway_client=gateway,
        knowledge_base=knowledge_base,
    )

    result = await retriever.retrieve("  first  ")

    assert gateway.calls == [("first", "request-7")]
    assert knowledge_base.vector_calls == [([1.0, 0.0, 0.0], 3)]
    assert knowledge_base.fts_calls == [("first", 3)]
    assert knowledge_base.metadata_calls == [["chunk-1"]]
    assert result.candidate_count == 1
    assert result.vector_candidate_count == 1
    assert result.fts_candidate_count == 1
    assert result.embedding_model == "embed-test"
    assert result.embedding_dim == 3
    assert result.retrieval_mode == "hybrid"
    assert result.hits[0].rank == 1
    assert result.hits[0].matched_by == "both"
    assert "query" not in result.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["   ", "x" * 4097, None])
async def test_local_retriever_rejects_invalid_query_without_calling_embedding(
    query: str | None,
) -> None:
    """Blank and oversized input fail as 400 before any upstream or local work."""
    gateway = FakeGateway()
    knowledge_base = FakeKnowledgeBase()
    retriever = LocalRetriever(
        retrieval_settings(),
        gateway_client=gateway,
        knowledge_base=knowledge_base,
    )

    with pytest.raises(Exception) as error:
        await retriever.retrieve(query)  # type: ignore[arg-type]

    assert error.value.code in {"missing_retrieval_query", "retrieval_query_too_long"}
    assert gateway.calls == []
    assert knowledge_base.load_calls == 0


@pytest.mark.asyncio
async def test_embedding_failure_skips_local_retrieval_and_preserves_gateway_error() -> None:
    """Gateway failures keep their existing stable error rather than being remapped."""
    gateway = FakeGateway(error=gateway_connection_error())
    knowledge_base = FakeKnowledgeBase()
    retriever = LocalRetriever(
        retrieval_settings(),
        gateway_client=gateway,
        knowledge_base=knowledge_base,
    )

    with pytest.raises(Exception) as error:
        await retriever.retrieve("first")

    assert error.value.code == "gateway_connection_failed"
    assert knowledge_base.load_calls == 0


@pytest.mark.asyncio
async def test_local_asset_and_query_failures_map_to_distinct_stable_errors() -> None:
    """Broken assets and query execution failures remain distinguishable to the API layer."""
    unavailable = LocalRetriever(
        retrieval_settings(),
        gateway_client=FakeGateway(),
        knowledge_base=FakeKnowledgeBase(load_error=KnowledgeBaseLoadError()),
    )
    failed_query = LocalRetriever(
        retrieval_settings(),
        gateway_client=FakeGateway(),
        knowledge_base=FakeKnowledgeBase(vector_error=KnowledgeBaseQueryError()),
    )

    with pytest.raises(Exception) as asset_error:
        await unavailable.retrieve("first")
    with pytest.raises(Exception) as query_error:
        await failed_query.retrieve("first")

    assert asset_error.value.code == "rag_knowledge_base_unavailable"
    assert query_error.value.code == "rag_retrieval_unavailable"


@pytest.mark.asyncio
async def test_recoverable_fts_failure_falls_back_to_vector_only_when_allowed() -> None:
    """Optional FTS failure does not discard healthy FAISS evidence under fallback policy."""
    gateway = FakeGateway()
    knowledge_base = FakeKnowledgeBase(fts_error=FtsSearchFallbackError())
    retriever = LocalRetriever(
        retrieval_settings(RAG_ALLOW_FTS_FALLBACK=True),
        gateway_client=gateway,
        knowledge_base=knowledge_base,
    )

    result = await retriever.retrieve("first")

    assert result.retrieval_mode == "vector"
    assert result.fts_candidate_count == 0
    assert result.hits[0].matched_by == "vector"


@pytest.mark.asyncio
async def test_recoverable_fts_failure_fails_when_fallback_is_disabled() -> None:
    """Operators can require FTS availability instead of accepting a degraded result."""
    retriever = LocalRetriever(
        retrieval_settings(RAG_ALLOW_FTS_FALLBACK=False),
        gateway_client=FakeGateway(),
        knowledge_base=FakeKnowledgeBase(fts_error=FtsSearchFallbackError()),
    )

    with pytest.raises(Exception) as error:
        await retriever.retrieve("first")

    assert error.value.code == "rag_retrieval_unavailable"


@pytest.mark.asyncio
async def test_concurrent_retrievals_do_not_share_candidates_or_results() -> None:
    """Separate async calls keep their own embedding and local candidate state."""
    retriever = LocalRetriever(
        retrieval_settings(RAG_ENABLE_FTS=False),
        gateway_client=FakeGateway(),
        knowledge_base=FakeKnowledgeBase(),
    )

    first, second = await asyncio.gather(
        retriever.retrieve("first"),
        retriever.retrieve("second"),
    )

    assert [hit.chunk_id for hit in first.hits] == ["chunk-1"]
    assert [hit.chunk_id for hit in second.hits] == ["chunk-2"]
    assert first.retrieval_mode == second.retrieval_mode == "vector"
