"""Strict data contracts shared by the local retrieval implementation."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalHit(BaseModel):
    """One ranked knowledge-base chunk with its traceable source metadata."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)
    title: str | None = None
    doc_title: str | None = None
    section_path: str | None = None
    article_no: str | None = None
    article_range: str | None = None
    relative_path: str = Field(min_length=1)
    paragraph_start: int | None = None
    paragraph_end: int | None = None
    vector_id: int | None = None
    vector_score: float | None = None
    vector_rank: int | None = Field(default=None, ge=1)
    fts_rank: int | None = Field(default=None, ge=1)
    final_score: float = Field(ge=0)
    matched_by: Literal["vector", "fts", "both"]


class RetrievalResult(BaseModel):
    """Structured local-retrieval output consumed by later RAG stages."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    hits: list[RetrievalHit]
    candidate_count: int = Field(ge=0)
    vector_candidate_count: int = Field(ge=0)
    fts_candidate_count: int = Field(ge=0)
    embedding_model: str = Field(min_length=1)
    embedding_dim: int = Field(gt=0)
    retrieval_mode: Literal["vector", "hybrid"]


@dataclass(frozen=True)
class RankedCandidate:
    """Internal rank information before full chunk metadata is loaded."""

    chunk_id: str
    vector_id: int | None = None
    vector_score: float | None = None
    vector_rank: int | None = None
    fts_rank: int | None = None
    final_score: float = 0.0
    matched_by: Literal["vector", "fts", "both"] = "vector"
