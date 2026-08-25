"""Build and query the FAISS vector index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class FaissUnavailableError(RuntimeError):
    """Raised when faiss is not installed in the active Python environment."""


@dataclass
class VectorSearchResult:
    """One FAISS search hit."""

    score: float
    vector_id: int | None = None
    chunk_id: str | None = None


@dataclass
class EmbeddingLoadResult:
    """Vectors loaded from the stage 8 JSONL file."""

    vectors: np.ndarray
    vector_ids: np.ndarray
    chunk_ids: list[str]
    source_line_count: int


def _require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise FaissUnavailableError(
            "FAISS is not installed. Install faiss-cpu in the active environment "
            "before running stage 9."
        ) from exc
    return faiss


def count_jsonl_records(path: Path) -> int:
    """Count non-empty records in a JSONL file without parsing the JSON."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _read_json_object(line: str, line_no: int) -> dict[str, Any]:
    try:
        item = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    if not isinstance(item, dict):
        raise ValueError(f"JSONL line {line_no} is not an object")
    return item


def load_embedding_jsonl(
    path: Path,
    *,
    expected_dim: int | None = None,
    limit: int | None = None,
    expected_count: int | None = None,
) -> EmbeddingLoadResult:
    """Load stage 8 vectors into a float32 matrix."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding vector file does not exist: {path}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if expected_count is not None and expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")

    if expected_count is None:
        expected_count = count_jsonl_records(path)
    capacity = expected_count if limit is None else min(expected_count, limit)
    dim = expected_dim or 0
    vectors: np.ndarray | None = None
    vector_ids: list[int] = []
    chunk_ids: list[str] = []
    seen_vector_ids: set[int] = set()
    source_line_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(vector_ids) >= limit:
                break
            if not line.strip():
                continue
            source_line_count += 1

            item = _read_json_object(line, line_no)
            raw_vector_id = item.get("vector_id")
            raw_chunk_id = item.get("chunk_id")
            raw_vector = item.get("vector")

            if not isinstance(raw_vector_id, int):
                raise ValueError(f"Missing or invalid vector_id at line {line_no}")
            if raw_vector_id in seen_vector_ids:
                raise ValueError(f"Duplicate vector_id {raw_vector_id} at line {line_no}")
            if not isinstance(raw_chunk_id, str) or not raw_chunk_id:
                raise ValueError(f"Missing or invalid chunk_id at line {line_no}")
            if not isinstance(raw_vector, list):
                raise ValueError(f"Missing or invalid vector at line {line_no}")

            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.ndim != 1:
                raise ValueError(f"Vector at line {line_no} must be one-dimensional")
            if expected_dim is not None and vector.shape[0] != expected_dim:
                raise ValueError(
                    f"Vector dim mismatch at line {line_no}: "
                    f"expected={expected_dim}, actual={vector.shape[0]}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"Vector at line {line_no} contains non-finite values")

            if vectors is None:
                dim = int(vector.shape[0])
                vectors = np.empty((capacity, dim), dtype=np.float32)
            elif vector.shape[0] != dim:
                raise ValueError(
                    f"Vector dim mismatch at line {line_no}: expected={dim}, "
                    f"actual={vector.shape[0]}"
                )

            row_index = len(vector_ids)
            if row_index >= capacity:
                raise ValueError(
                    "Embedding vector file contains more records than expected_count"
                )
            vectors[row_index] = vector
            vector_ids.append(raw_vector_id)
            chunk_ids.append(raw_chunk_id)
            seen_vector_ids.add(raw_vector_id)

    if vectors is None:
        vectors = np.empty((0, dim), dtype=np.float32)
    else:
        vectors = vectors[: len(vector_ids)]

    return EmbeddingLoadResult(
        vectors=vectors,
        vector_ids=np.asarray(vector_ids, dtype=np.int64),
        chunk_ids=chunk_ids,
        source_line_count=source_line_count,
    )


def build_faiss_index(
    vectors,
    vector_ids: list[int] | np.ndarray,
    output_path: Path,
    *,
    metric: str = "inner_product",
) -> int:
    """Build and save a FAISS IndexIDMap2 index. Returns the vector count."""
    faiss = _require_faiss()

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if matrix.shape[0] == 0:
        raise ValueError("vectors must not be empty")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain non-finite values")

    ids = np.asarray(vector_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("vector_ids must be one-dimensional")
    if ids.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"vector count and vector_id count mismatch: "
            f"{matrix.shape[0]} != {ids.shape[0]}"
        )
    if len(set(ids.tolist())) != ids.shape[0]:
        raise ValueError("vector_ids contain duplicates")

    dim = int(matrix.shape[1])
    metric = metric.lower()
    if metric in {"inner_product", "ip"}:
        base_index = faiss.IndexFlatIP(dim)
    elif metric in {"l2", "euclidean"}:
        base_index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported FAISS metric: {metric}")

    index = faiss.IndexIDMap2(base_index)
    index.add_with_ids(matrix, ids)

    write_faiss_index(index, output_path)
    return int(index.ntotal)


def write_faiss_index(index, index_path: Path) -> None:
    """Persist an index through Python file I/O.

    FAISS' Windows file helpers use a narrow-character path API in some wheels,
    which breaks for normal Chinese user directories.  Serialising in memory
    leaves path handling to Python's Unicode-aware ``Path`` implementation.
    """
    faiss = _require_faiss()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep compatibility with minimal FAISS test doubles and older FAISS
    # bindings.  Production Windows wheels provide serialize_index, which is
    # required for Unicode-safe paths.
    if not hasattr(faiss, "serialize_index"):
        faiss.write_index(index, str(index_path))
        return
    serialized = faiss.serialize_index(index)
    index_path.write_bytes(bytes(serialized))


def load_faiss_index(index_path: Path):
    """Load a FAISS index through Python file I/O (Unicode-path safe)."""
    faiss = _require_faiss()
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index does not exist: {index_path}")
    if not hasattr(faiss, "deserialize_index"):
        return faiss.read_index(str(index_path))
    serialized = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
    return faiss.deserialize_index(serialized)


def search_faiss(
    query_vector,
    top_k: int = 10,
    *,
    index=None,
    index_path: Path | None = None,
) -> list[VectorSearchResult]:
    """Search an IndexIDMap2 index and return vector_id based hits."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if index is None:
        if index_path is None:
            raise ValueError("Either index or index_path must be provided")
        index = load_faiss_index(index_path)

    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or query.shape[0] != 1:
        raise ValueError("query_vector must be a one-dimensional vector")

    scores, ids = index.search(query, top_k)
    results: list[VectorSearchResult] = []
    for score, vector_id in zip(scores[0], ids[0]):
        vector_id_int = int(vector_id)
        if vector_id_int < 0:
            continue
        results.append(VectorSearchResult(score=float(score), vector_id=vector_id_int))
    return results
