"""Pure ranking and selection functions for local vector and FTS candidates."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from typing import Any

from local_rag_app.config import Settings
from local_rag_app.errors import (
    LocalRagError,
    missing_retrieval_query_error,
    rag_knowledge_base_unavailable_error,
    rag_retrieval_unavailable_error,
    retrieval_query_too_long_error,
)
from local_rag_app.gateway_client import ModelGatewayClient
from local_rag_app.knowledge_base import (
    ChunkMetadata,
    FtsSearchCandidate,
    FtsSearchFallbackError,
    KnowledgeBase,
    KnowledgeBaseLoadError,
    KnowledgeBaseQueryError,
    VectorSearchHit,
)
from local_rag_app.logging_config import get_request_id
from local_rag_app.retrieval_models import RankedCandidate, RetrievalHit, RetrievalResult
from local_rag_app.schemas import ChatCompletionRequest


MAX_RETRIEVAL_QUERY_LENGTH = 4096


class RetrievalFusionError(ValueError):
    """Raised when internally inconsistent candidates cannot be safely fused."""


class LocalRetriever:
    """Orchestrate one query embedding and local hybrid retrieval without HTTP concerns."""

    def __init__(
        self,
        settings: Settings,
        *,
        gateway_client: ModelGatewayClient | Any | None = None,
        knowledge_base: KnowledgeBase | Any | None = None,
    ) -> None:
        self._settings = settings
        self._gateway_client = gateway_client or ModelGatewayClient(settings)
        self._knowledge_base = knowledge_base or KnowledgeBase(settings)

    async def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve structured local evidence for one non-empty, bounded query."""
        normalized_query = validate_retrieval_query(query)
        embedding = await self._gateway_client.create_embedding(
            normalized_query,
            request_id=get_request_id(),
        )
        try:
            return await asyncio.to_thread(
                self._retrieve_locally,
                normalized_query,
                embedding,
            )
        except LocalRagError:
            raise
        except KnowledgeBaseLoadError as exc:
            raise rag_knowledge_base_unavailable_error() from exc
        except (KnowledgeBaseQueryError, RetrievalFusionError) as exc:
            raise rag_retrieval_unavailable_error() from exc

    def _retrieve_locally(
        self,
        query: str,
        embedding: list[float],
    ) -> RetrievalResult:
        """Run synchronous local work in a worker thread after embedding succeeds."""
        self._knowledge_base.load()
        vector_hits = self._knowledge_base.search_vector(
            embedding,
            self._settings.rag_vector_top_k,
        )
        fts_candidates: list[FtsSearchCandidate] = []
        if self._settings.rag_enable_fts:
            try:
                fts_candidates = self._knowledge_base.search_fts(
                    query,
                    self._settings.rag_fts_top_k,
                )
            except FtsSearchFallbackError:
                if not self._settings.rag_allow_fts_fallback:
                    raise KnowledgeBaseQueryError()

        fused = fuse_candidates(
            vector_hits,
            fts_candidates,
            rrf_k=self._settings.rag_rrf_k,
            vector_weight=self._settings.rag_vector_weight,
            fts_weight=self._settings.rag_fts_weight,
        )
        metadata_by_chunk_id: dict[str, ChunkMetadata]
        if fused:
            metadata_by_chunk_id = self._knowledge_base.load_chunk_metadata(
                [candidate.chunk_id for candidate in fused]
            )
        else:
            metadata_by_chunk_id = {}
        selected = select_final_candidates(
            fused,
            document_ids={
                chunk_id: metadata.doc_id
                for chunk_id, metadata in metadata_by_chunk_id.items()
            },
            final_top_k=self._settings.rag_final_top_k,
            max_chunks_per_doc=self._settings.rag_max_chunks_per_doc,
        )
        return RetrievalResult(
            hits=[
                _retrieval_hit_from_candidate(
                    rank=index,
                    candidate=candidate,
                    metadata=metadata_by_chunk_id[candidate.chunk_id],
                )
                for index, candidate in enumerate(selected, start=1)
            ],
            candidate_count=len(fused),
            vector_candidate_count=len(vector_hits),
            fts_candidate_count=len(fts_candidates),
            embedding_model=self._settings.upstream_embedding_model,
            embedding_dim=self._settings.rag_embedding_dim,
            retrieval_mode="hybrid" if fts_candidates else "vector",
        )


def validate_retrieval_query(query: str) -> str:
    """Strip one retrieval query without silently truncating its semantic content."""
    if not isinstance(query, str):
        raise missing_retrieval_query_error()
    normalized = query.strip()
    if not normalized:
        raise missing_retrieval_query_error()
    if len(normalized) > MAX_RETRIEVAL_QUERY_LENGTH:
        raise retrieval_query_too_long_error()
    return normalized


def extract_latest_user_query(request: ChatCompletionRequest) -> str:
    """Return the last non-empty user message without embedding the whole dialogue."""
    for message in reversed(request.messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    raise missing_retrieval_query_error()


def fuse_candidates(
    vector_candidates: Sequence[VectorSearchHit],
    fts_candidates: Sequence[FtsSearchCandidate],
    *,
    rrf_k: int,
    vector_weight: float,
    fts_weight: float,
) -> list[RankedCandidate]:
    """Fuse unique chunk candidates with weighted reciprocal-rank fusion.

    Only rank contributes to the RRF score.  FAISS scores are retained solely
    for later diagnostics and stable tie-breaking; SQLite BM25 scores are not
    mixed into the result because their scale and direction differ.
    """
    _validate_rrf_settings(rrf_k, vector_weight, fts_weight)

    combined: dict[str, dict[str, int | float | None]] = {}
    vector_chunk_ids: set[str] = set()
    vector_ranks: set[int] = set()
    for candidate in vector_candidates:
        _validate_vector_candidate(candidate, vector_chunk_ids, vector_ranks)
        combined[candidate.chunk_id] = {
            "vector_id": candidate.vector_id,
            "vector_score": candidate.vector_score,
            "vector_rank": candidate.vector_rank,
            "fts_rank": None,
        }

    fts_chunk_ids: set[str] = set()
    fts_ranks: set[int] = set()
    for candidate in fts_candidates:
        _validate_fts_candidate(candidate, fts_chunk_ids, fts_ranks)
        existing = combined.setdefault(
            candidate.chunk_id,
            {
                "vector_id": None,
                "vector_score": None,
                "vector_rank": None,
                "fts_rank": None,
            },
        )
        existing["fts_rank"] = candidate.fts_rank

    fused: list[RankedCandidate] = []
    for chunk_id, values in combined.items():
        vector_rank = _optional_positive_int(values["vector_rank"])
        fts_rank = _optional_positive_int(values["fts_rank"])
        score = 0.0
        if vector_rank is not None:
            score += vector_weight / (rrf_k + vector_rank)
        if fts_rank is not None:
            score += fts_weight / (rrf_k + fts_rank)
        if not isfinite(score) or score < 0:
            raise RetrievalFusionError("RRF score is invalid")

        matched_by = (
            "both"
            if vector_rank is not None and fts_rank is not None
            else "vector"
            if vector_rank is not None
            else "fts"
        )
        fused.append(
            RankedCandidate(
                chunk_id=chunk_id,
                vector_id=_optional_int(values["vector_id"]),
                vector_score=_optional_finite_float(values["vector_score"]),
                vector_rank=vector_rank,
                fts_rank=fts_rank,
                final_score=score,
                matched_by=matched_by,
            )
        )

    return sorted(fused, key=_candidate_sort_key)


def select_final_candidates(
    candidates: Sequence[RankedCandidate],
    *,
    document_ids: Mapping[str, str],
    final_top_k: int,
    max_chunks_per_doc: int,
) -> list[RankedCandidate]:
    """Keep ranked candidates while enforcing the configured per-document cap."""
    if (
        isinstance(final_top_k, bool)
        or not isinstance(final_top_k, Integral)
        or final_top_k <= 0
    ):
        raise RetrievalFusionError("final_top_k must be a positive integer")
    if (
        isinstance(max_chunks_per_doc, bool)
        or not isinstance(max_chunks_per_doc, Integral)
        or max_chunks_per_doc <= 0
    ):
        raise RetrievalFusionError("max_chunks_per_doc must be a positive integer")

    selected: list[RankedCandidate] = []
    seen_chunk_ids: set[str] = set()
    document_counts: dict[str, int] = {}
    for candidate in candidates:
        _validate_ranked_candidate(candidate)
        if candidate.chunk_id in seen_chunk_ids:
            raise RetrievalFusionError("duplicate fused chunk ID")
        seen_chunk_ids.add(candidate.chunk_id)
        doc_id = document_ids.get(candidate.chunk_id)
        if not isinstance(doc_id, str) or not doc_id:
            raise RetrievalFusionError("candidate document mapping is missing")
        if document_counts.get(doc_id, 0) >= max_chunks_per_doc:
            continue

        selected.append(candidate)
        document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
        if len(selected) == final_top_k:
            break
    return selected


def _retrieval_hit_from_candidate(
    *,
    rank: int,
    candidate: RankedCandidate,
    metadata: ChunkMetadata,
) -> RetrievalHit:
    """Combine selected rank data with one complete source record for later RAG stages."""
    if metadata.chunk_id != candidate.chunk_id:
        raise RetrievalFusionError("selected chunk metadata does not match its candidate")
    if candidate.vector_id is not None and metadata.vector_id != candidate.vector_id:
        raise RetrievalFusionError("selected vector ID does not match chunk metadata")
    return RetrievalHit(
        rank=rank,
        chunk_id=metadata.chunk_id,
        doc_id=metadata.doc_id,
        chunk_text=metadata.chunk_text,
        title=metadata.title,
        doc_title=metadata.doc_title,
        section_path=metadata.section_path,
        article_no=metadata.article_no,
        article_range=metadata.article_range,
        relative_path=metadata.relative_path,
        paragraph_start=metadata.paragraph_start,
        paragraph_end=metadata.paragraph_end,
        vector_id=metadata.vector_id,
        vector_score=candidate.vector_score,
        vector_rank=candidate.vector_rank,
        fts_rank=candidate.fts_rank,
        final_score=candidate.final_score,
        matched_by=candidate.matched_by,
    )


def _validate_rrf_settings(
    rrf_k: int,
    vector_weight: float,
    fts_weight: float,
) -> None:
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, Integral) or rrf_k <= 0:
        raise RetrievalFusionError("rrf_k must be a positive integer")
    for weight in (vector_weight, fts_weight):
        if isinstance(weight, bool) or not isinstance(weight, Real) or not isfinite(weight):
            raise RetrievalFusionError("RRF weights must be finite numbers")
        if weight < 0:
            raise RetrievalFusionError("RRF weights cannot be negative")
    if vector_weight + fts_weight <= 0:
        raise RetrievalFusionError("at least one RRF weight must be positive")


def _validate_vector_candidate(
    candidate: VectorSearchHit,
    seen_chunk_ids: set[str],
    seen_ranks: set[int],
) -> None:
    if not isinstance(candidate, VectorSearchHit):
        raise RetrievalFusionError("vector candidate has an invalid type")
    if (
        not isinstance(candidate.chunk_id, str)
        or not candidate.chunk_id
        or candidate.chunk_id in seen_chunk_ids
    ):
        raise RetrievalFusionError("duplicate vector chunk ID")
    if (
        isinstance(candidate.vector_rank, bool)
        or not isinstance(candidate.vector_rank, Integral)
        or candidate.vector_rank <= 0
        or candidate.vector_rank in seen_ranks
    ):
        raise RetrievalFusionError("vector ranks must be unique positive integers")
    if (
        isinstance(candidate.vector_score, bool)
        or not isinstance(candidate.vector_score, Real)
        or not isfinite(candidate.vector_score)
    ):
        raise RetrievalFusionError("vector score is invalid")
    seen_chunk_ids.add(candidate.chunk_id)
    seen_ranks.add(candidate.vector_rank)


def _validate_fts_candidate(
    candidate: FtsSearchCandidate,
    seen_chunk_ids: set[str],
    seen_ranks: set[int],
) -> None:
    if not isinstance(candidate, FtsSearchCandidate):
        raise RetrievalFusionError("FTS candidate has an invalid type")
    if (
        not isinstance(candidate.chunk_id, str)
        or not candidate.chunk_id
        or candidate.chunk_id in seen_chunk_ids
    ):
        raise RetrievalFusionError("duplicate FTS chunk ID")
    if (
        isinstance(candidate.fts_rank, bool)
        or not isinstance(candidate.fts_rank, Integral)
        or candidate.fts_rank <= 0
        or candidate.fts_rank in seen_ranks
    ):
        raise RetrievalFusionError("FTS ranks must be unique positive integers")
    if (
        isinstance(candidate.bm25_score, bool)
        or not isinstance(candidate.bm25_score, Real)
        or not isfinite(candidate.bm25_score)
    ):
        raise RetrievalFusionError("BM25 score is invalid")
    seen_chunk_ids.add(candidate.chunk_id)
    seen_ranks.add(candidate.fts_rank)


def _candidate_sort_key(candidate: RankedCandidate) -> tuple[float, int, float, str]:
    vector_score_key = (
        -candidate.vector_score if candidate.vector_score is not None else float("inf")
    )
    return (
        -candidate.final_score,
        0 if candidate.matched_by == "both" else 1,
        vector_score_key,
        candidate.chunk_id,
    )


def _validate_ranked_candidate(candidate: RankedCandidate) -> None:
    if not isinstance(candidate, RankedCandidate):
        raise RetrievalFusionError("fused candidate has an invalid type")
    if (
        not isinstance(candidate.chunk_id, str)
        or not candidate.chunk_id
        or isinstance(candidate.final_score, bool)
        or not isinstance(candidate.final_score, Real)
        or not isfinite(candidate.final_score)
        or candidate.final_score < 0
    ):
        raise RetrievalFusionError("fused candidate is invalid")


def _optional_positive_int(value: int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise RetrievalFusionError("candidate rank is invalid")
    return int(value)


def _optional_int(value: int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RetrievalFusionError("vector ID is invalid")
    return int(value)


def _optional_finite_float(value: int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise RetrievalFusionError("vector score is invalid")
    return float(value)
