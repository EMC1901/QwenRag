"""Strict-model tests for stage-1 retrieval inputs and outputs."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from local_rag_app.retrieval_models import (
    RankedCandidate,
    RetrievalHit,
    RetrievalResult,
)


def valid_hit() -> dict:
    """Return one complete, synthetic retrieval hit with no customer data."""
    return {
        "rank": 1,
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "chunk_text": "Synthetic knowledge-base text.",
        "title": "Synthetic title",
        "doc_title": "Synthetic document",
        "section_path": "Chapter 1",
        "article_no": "Article 1",
        "article_range": None,
        "relative_path": "fixtures/synthetic.docx",
        "paragraph_start": 1,
        "paragraph_end": 2,
        "vector_id": 10,
        "vector_score": 0.8,
        "vector_rank": 1,
        "fts_rank": 2,
        "final_score": 0.015,
        "matched_by": "both",
    }


def test_retrieval_models_accept_documented_shape() -> None:
    """Later RAG stages receive a stable and fully traceable result contract."""
    hit = RetrievalHit.model_validate(valid_hit())
    result = RetrievalResult(
        hits=[hit],
        candidate_count=2,
        vector_candidate_count=2,
        fts_candidate_count=1,
        embedding_model="qwen3-embedding-0.6b",
        embedding_dim=1024,
        retrieval_mode="hybrid",
    )

    assert result.hits[0].chunk_id == "chunk-1"
    assert result.hits[0].matched_by == "both"
    assert result.retrieval_mode == "hybrid"


def test_retrieval_hit_rejects_unknown_fields() -> None:
    """Accidental runtime fields must not silently change the handoff contract."""
    payload = {**valid_hit(), "private_debug_value": "must not pass"}

    with pytest.raises(ValidationError, match="private_debug_value"):
        RetrievalHit.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", "1"),
        ("matched_by", "semantic"),
        ("final_score", float("nan")),
        ("relative_path", ""),
    ],
)
def test_retrieval_hit_rejects_invalid_types_and_values(field: str, value) -> None:
    """Ranks, scores, source labels, and traceability fields are strict."""
    payload = {**valid_hit(), field: value}

    with pytest.raises(ValidationError):
        RetrievalHit.model_validate(payload)


def test_retrieval_result_rejects_unknown_mode_and_fields() -> None:
    """Only vector and hybrid modes belong to the stage-1 contract."""
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate(
            {
                "hits": [],
                "candidate_count": 0,
                "vector_candidate_count": 0,
                "fts_candidate_count": 0,
                "embedding_model": "embed-test",
                "embedding_dim": 3,
                "retrieval_mode": "keyword",
                "debug": True,
            }
        )


def test_ranked_candidate_is_immutable() -> None:
    """Fusion stages cannot accidentally mutate a candidate shared by another list."""
    candidate = RankedCandidate(chunk_id="chunk-1", vector_rank=1)

    with pytest.raises(FrozenInstanceError):
        candidate.final_score = 1.0  # type: ignore[misc]
