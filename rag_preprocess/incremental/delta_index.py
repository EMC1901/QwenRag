"""Stage 12: build, validate and publish one immutable Delta package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Callable

from rag_preprocess.faiss_builder import build_faiss_index, load_embedding_jsonl, load_faiss_index

from .manifest import LayerSpec, ManifestError, load_manifest, publish_delta, sha256_file
from .persistence import atomic_write_text


class DeltaIndexError(RuntimeError):
    """Raised when a Delta cannot be indexed, validated or published."""


def build_delta_indexes(
    delta_root: Path,
    *,
    embedding_dim: int,
    embedding_model: str,
    embedding_revision: str = "legacy-unknown",
    faiss_builder: Callable[..., int] = build_faiss_index,
) -> dict[str, object]:
    """Build only this Delta's FTS and FAISS assets; Base remains untouched."""

    database = delta_root / "delta.db"
    vector_dir = delta_root / "vector_index"
    embeddings = vector_dir / "embeddings.jsonl"
    if not database.is_file() or not embeddings.is_file():
        raise DeltaIndexError("DELTA_ASSET_MISSING")
    with sqlite3.connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DELETE FROM chunk_fts")
            connection.execute(
                """INSERT INTO chunk_fts(chunk_id,title,section_path,article_no,chunk_text)
                   SELECT chunk_id,COALESCE(title,''),COALESCE(section_path,''),
                          COALESCE(article_range,article_no,''),chunk_text
                   FROM chunks
                   WHERE embedding_status='success'"""
            )
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks WHERE embedding_status='success'").fetchone()[0])
            fts_count = int(connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0])
            if chunk_count <= 0 or chunk_count != fts_count:
                raise DeltaIndexError("DELTA_FTS_COUNT_MISMATCH")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    loaded = load_embedding_jsonl(embeddings, expected_dim=embedding_dim, expected_count=chunk_count)
    if loaded.source_line_count != chunk_count:
        raise DeltaIndexError("DELTA_JSONL_COUNT_MISMATCH")
    temporary_index = vector_dir / ".index.faiss.building"
    try:
        vector_count = faiss_builder(loaded.vectors, loaded.vector_ids, temporary_index)
        if vector_count != chunk_count:
            raise DeltaIndexError("DELTA_FAISS_COUNT_MISMATCH")
        os.replace(temporary_index, vector_dir / "index.faiss")
    finally:
        temporary_index.unlink(missing_ok=True)

    delta_meta = _load_json_object(delta_root / "delta.meta.json")
    metadata = {
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "embedding_dim": embedding_dim,
        "vector_metric": "inner_product",
        "vector_normalized": True,
        "vector_count": vector_count,
        "source_vector_file_line_count": loaded.source_line_count,
        "db_total_chunks": chunk_count,
        "db_embedding_success_count": chunk_count,
        "db_success_with_vector_id_count": chunk_count,
        "index_type": "faiss",
        "faiss_factory": "IndexIDMap2(IndexFlatIP)",
        "is_partial_embedding_index": True,
        "delta_id": delta_meta.get("delta_id"),
        "task_id": delta_meta.get("task_id"),
        "parent_manifest_revision": delta_meta.get("parent_manifest_revision"),
    }
    atomic_write_text(vector_dir / "index.meta.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    _write_tombstones_json(delta_root)
    delta_meta.update(
        {
            "fts_count": fts_count,
            "vector_count": vector_count,
            "asset_sha256": {
                "delta.db": sha256_file(database),
                "embeddings.jsonl": sha256_file(embeddings),
                "index.faiss": sha256_file(vector_dir / "index.faiss"),
            },
            "validation_status": "pending_stage_12_validation",
        }
    )
    atomic_write_text(delta_root / "delta.meta.json", json.dumps(delta_meta, ensure_ascii=False, indent=2) + "\n")
    return metadata


def validate_delta_package(
    delta_root: Path,
    *,
    embedding_dim: int,
    embedding_model: str,
    embedding_revision: str = "legacy-unknown",
    faiss_loader: Callable[[Path], Any] = load_faiss_index,
) -> dict[str, object]:
    """Validate Delta SQLite/FTS/JSONL/meta/FAISS consistency before publish."""

    database = delta_root / "delta.db"
    vector_dir = delta_root / "vector_index"
    index_path = vector_dir / "index.faiss"
    index_meta_path = vector_dir / "index.meta.json"
    for path in (database, vector_dir / "embeddings.jsonl", index_path, index_meta_path, delta_root / "delta.meta.json", delta_root / "tombstones.json"):
        if not path.is_file():
            raise DeltaIndexError("DELTA_ASSET_MISSING")
    index_meta = _load_json_object(index_meta_path)
    delta_meta = _load_json_object(delta_root / "delta.meta.json")
    if (
        index_meta.get("embedding_model") != embedding_model
        or index_meta.get("embedding_revision", "legacy-unknown") != embedding_revision
        or index_meta.get("embedding_dim") != embedding_dim
        or index_meta.get("vector_metric") != "inner_product"
        or index_meta.get("vector_normalized") is not True
        or index_meta.get("is_partial_embedding_index") is not True
    ):
        raise DeltaIndexError("DELTA_INDEX_METADATA_INVALID")
    with sqlite3.connect(database) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks WHERE embedding_status='success'").fetchone()[0])
        fts_count = int(connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0])
        duplicate_vector_ids = int(connection.execute("SELECT COUNT(*) FROM (SELECT vector_id FROM chunks WHERE vector_id IS NOT NULL GROUP BY vector_id HAVING COUNT(*)>1)").fetchone()[0])
    loaded = load_embedding_jsonl(vector_dir / "embeddings.jsonl", expected_dim=embedding_dim, expected_count=chunk_count)
    index = faiss_loader(index_path)
    index_count = getattr(index, "ntotal", None)
    index_dim = getattr(index, "d", None)
    if (
        quick_check != "ok"
        or fk_errors
        or chunk_count <= 0
        or fts_count != chunk_count
        or duplicate_vector_ids
        or loaded.source_line_count != chunk_count
        or index_count != chunk_count
        or index_dim != embedding_dim
        or index_meta.get("vector_count") != chunk_count
        or delta_meta.get("chunk_count") != chunk_count
        or delta_meta.get("vector_count") != chunk_count
    ):
        raise DeltaIndexError("DELTA_INCONSISTENT")
    delta_meta["validation_status"] = "passed"
    atomic_write_text(delta_root / "delta.meta.json", json.dumps(delta_meta, ensure_ascii=False, indent=2) + "\n")
    return delta_meta


def publish_delta_package(settings, delta_root: Path, *, expected_revision: int) -> object:
    """Move a validated Delta into its immutable directory, then atomically publish it."""

    metadata = _load_json_object(delta_root / "delta.meta.json")
    if metadata.get("validation_status") != "passed":
        raise DeltaIndexError("DELTA_NOT_VALIDATED")
    delta_id = metadata.get("delta_id")
    if not isinstance(delta_id, str) or not delta_id:
        raise DeltaIndexError("DELTA_METADATA_INVALID")
    settings.deltas_dir.mkdir(parents=True, exist_ok=True)
    final_dir = settings.deltas_dir / delta_id
    if final_dir.exists():
        if not final_dir.is_dir() or sha256_file(final_dir / "delta.meta.json") != sha256_file(delta_root / "delta.meta.json"):
            raise DeltaIndexError("DELTA_FINAL_DIRECTORY_CONFLICT")
    else:
        try:
            os.replace(delta_root, final_dir)
        except PermissionError:
            # On some Windows ACL combinations ``ReplaceFile`` semantics for a
            # new directory are denied even though an atomic same-volume rename
            # is permitted.  The target is verified absent above.
            try:
                os.rename(delta_root, final_dir)
            except PermissionError:
                # The manifest remains unchanged until this complete small
                # package copy has finished, so an incomplete destination is
                # never visible to the runtime.  Keep staging for recovery.
                shutil.copytree(delta_root, final_dir)
    layer = LayerSpec(
        delta_id,
        final_dir.relative_to(settings.knowledge_base_root).as_posix(),
        sha256_file(final_dir / "delta.meta.json"),
        str(metadata.get("created_at") or ""),
    )
    next_vector_id = _next_vector_id(final_dir / "delta.db")
    current = load_manifest(settings.knowledge_base_root, settings.manifest_path)
    if current is not None:
        next_vector_id = max(next_vector_id, current.next_vector_id)
    try:
        return publish_delta(
            settings.knowledge_base_root,
            settings.manifest_path,
            layer,
            expected_revision=expected_revision,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            next_vector_id=next_vector_id,
        )
    except ManifestError as exc:
        raise DeltaIndexError(str(exc)) from exc


def _write_tombstones_json(delta_root: Path) -> None:
    with sqlite3.connect(delta_root / "delta.db") as connection:
        rows = connection.execute(
            "SELECT entity_type,entity_id,superseded_by_version_id FROM delta_tombstones ORDER BY entity_type,entity_id"
        ).fetchall()
    payload = {
        "schema_version": 1,
        "tombstones": [
            {"entity_type": row[0], "entity_id": row[1], "superseded_by_version_id": row[2]}
            for row in rows
        ],
    }
    atomic_write_text(delta_root / "tombstones.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeltaIndexError("DELTA_METADATA_INVALID") from exc
    if not isinstance(value, dict):
        raise DeltaIndexError("DELTA_METADATA_INVALID")
    return value


def _next_vector_id(database: Path) -> int:
    with sqlite3.connect(database) as connection:
        value = connection.execute("SELECT MAX(vector_id) FROM chunks").fetchone()[0]
    return int(value) + 1 if value is not None else 0
