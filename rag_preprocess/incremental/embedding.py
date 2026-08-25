"""Stage 10: file-isolated, resumable embedding artefacts."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import re
from typing import Callable, Iterable, Sequence

from rag_preprocess.embedding_client import EmbeddingResult, embed_batch, normalize_embedding, validate_embedding
from .persistence import atomic_write_text, read_checkpoint, write_checkpoint

VECTOR_SCHEMA_VERSION = 1
_RETRY_DELAYS = (1.0, 3.0, 9.0)

class EmbeddingPreflightError(RuntimeError): pass
class EmbeddingFileError(RuntimeError): pass

Embedder = Callable[..., list[EmbeddingResult]]

def preflight_embedding(settings, *, embedder: Embedder = embed_batch) -> None:
    """Reject a task before any file vector artefact is written."""
    _validate_index_metadata(settings)
    results = embedder(["incremental embedding preflight"], model=settings.embedding_model, batch_size=1, base_url=settings.embedding_base_url, max_retries=0, api_key=settings.embedding_api_key)
    if len(results) != 1 or not results[0].success or not results[0].vector:
        raise EmbeddingPreflightError("EMBEDDING_SERVICE_UNAVAILABLE")
    if not validate_embedding(results[0].vector, settings.embedding_dim):
        raise EmbeddingPreflightError("EMBEDDING_DIMENSION_INVALID")
    normalized = normalize_embedding(results[0].vector)
    if not _is_unit_vector(normalized): raise EmbeddingPreflightError("EMBEDDING_NORMALIZATION_INVALID")

def embed_file_chunks(settings, version_id: str, chunks: Sequence[object], vectors_dir: Path, *, embedder: Embedder = embed_batch, sleeper: Callable[[float], None] = __import__('time').sleep, jitter: Callable[[], float] = random.random) -> Path:
    """Write one file only after every chunk has a valid normalized vector."""
    vectors_dir.mkdir(parents=True, exist_ok=True)
    vector_path = vectors_dir / f"{version_id}.jsonl"
    checkpoint_path = vectors_dir / f"{version_id}.checkpoint.json"
    chunk_ids = [str(chunk.chunk_id) for chunk in chunks]
    if not chunk_ids or len(chunk_ids) != len(set(chunk_ids)): raise EmbeddingFileError("CHUNK_IDS_INVALID")
    if _valid_resume(vector_path, checkpoint_path, chunk_ids, settings): return vector_path
    records: list[dict[str, object]] = []
    for offset in range(0, len(chunks), settings.embedding_batch_size):
        batch = list(chunks[offset:offset + settings.embedding_batch_size])
        vectors = _embed_with_retry(settings, [str(chunk.chunk_text_for_embedding) for chunk in batch], embedder, sleeper, jitter)
        for chunk, vector in zip(batch, vectors, strict=True):
            records.append({"schema_version": VECTOR_SCHEMA_VERSION, "chunk_id": str(chunk.chunk_id), "model": settings.embedding_model, "dim": settings.embedding_dim, "normalized": True, "vector": vector})
        _write_records(vector_path, records)
        write_checkpoint(checkpoint_path, {"schema_version": VECTOR_SCHEMA_VERSION, "version_id": version_id, "chunk_ids": [record["chunk_id"] for record in records], "completed_chunk_count": len(records)})
    return vector_path

def _embed_with_retry(settings, texts: list[str], embedder: Embedder, sleeper, jitter) -> list[list[float]]:
    last_code = "EMBEDDING_REQUEST_FAILED"
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            results = embedder(texts, model=settings.embedding_model, batch_size=len(texts), base_url=settings.embedding_base_url, max_retries=0, api_key=settings.embedding_api_key)
            if len(results) != len(texts): raise EmbeddingFileError("EMBEDDING_RESPONSE_COUNT_INVALID")
            if all(item.success and item.vector and validate_embedding(item.vector, settings.embedding_dim) for item in results):
                return [normalize_embedding(item.vector or []) for item in results]
            messages = " ".join(item.error_message or "" for item in results)
            if _is_non_retryable(messages): raise EmbeddingFileError("EMBEDDING_RESPONSE_INVALID")
            last_code = "EMBEDDING_REQUEST_FAILED"
        except EmbeddingFileError: raise
        except Exception as exc:
            if _is_non_retryable(str(exc)): raise EmbeddingFileError("EMBEDDING_REQUEST_INVALID") from exc
        if attempt < len(_RETRY_DELAYS): sleeper(_RETRY_DELAYS[attempt] + jitter() * 0.2)
    raise EmbeddingFileError(last_code)

def _write_records(path: Path, records: Iterable[dict[str, object]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    atomic_write_text(path, text)

def _valid_resume(vector_path: Path, checkpoint_path: Path, expected_ids: list[str], settings) -> bool:
    try:
        checkpoint = read_checkpoint(checkpoint_path)
        if checkpoint.get("schema_version") != VECTOR_SCHEMA_VERSION or checkpoint.get("chunk_ids") != expected_ids: return False
        records = [json.loads(line) for line in vector_path.read_text(encoding="utf-8").splitlines() if line]
        if [record.get("chunk_id") for record in records] != expected_ids: return False
        return all(record.get("model") == settings.embedding_model and record.get("dim") == settings.embedding_dim and record.get("normalized") is True and validate_embedding(record.get("vector", []), settings.embedding_dim) and _is_unit_vector(record["vector"]) for record in records)
    except (OSError, ValueError, TypeError, json.JSONDecodeError): return False

def _validate_index_metadata(settings) -> None:
    metadata = settings.knowledge_base_root / "vector_index" / "index.meta.json"
    if not metadata.exists(): return
    try: raw=json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise EmbeddingPreflightError("INDEX_METADATA_INVALID") from exc
    if raw.get("embedding_model") != settings.embedding_model or raw.get("embedding_revision", "legacy-unknown") != settings.embedding_revision or raw.get("embedding_dim") != settings.embedding_dim or raw.get("vector_normalized") is not True or raw.get("vector_metric") != "inner_product": raise EmbeddingPreflightError("EMBEDDING_INDEX_CONFIGURATION_MISMATCH")

def _is_unit_vector(vector: Sequence[float]) -> bool:
    norm=math.sqrt(sum(float(value)*float(value) for value in vector)); return math.isfinite(norm) and abs(norm-1.0) < 1e-4

def _is_non_retryable(message: str) -> bool:
    return bool(re.search(r"\b4\d\d\b", message)) or "dimension" in message.lower() or "response" in message.lower()
