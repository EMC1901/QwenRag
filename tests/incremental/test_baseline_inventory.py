"""Tests for the read-only stage-0 baseline recorder."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from tools.record_incremental_baseline import collect_incremental_baseline


def test_baseline_recorder_reads_counts_and_optional_hashes_without_mutation(
    tmp_path: Path,
) -> None:
    """Baseline data comes from assets and preserves their bytes unchanged."""
    root = tmp_path / "fixture-kb"
    vector_dir = root / "vector_index"
    vector_dir.mkdir(parents=True)
    database = root / "metadata.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (doc_id TEXT);
            CREATE TABLE chunks (chunk_id TEXT);
            CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id);
            INSERT INTO documents VALUES ('doc-1');
            INSERT INTO documents VALUES ('doc-2');
            INSERT INTO chunks VALUES ('chunk-1');
            INSERT INTO chunks VALUES ('chunk-2');
            INSERT INTO chunks VALUES ('chunk-3');
            INSERT INTO chunk_fts VALUES ('chunk-1');
            INSERT INTO chunk_fts VALUES ('chunk-2');
            INSERT INTO chunk_fts VALUES ('chunk-3');
            """
        )
    (vector_dir / "embeddings.jsonl").write_text("a\n\nb\n", encoding="utf-8")
    (vector_dir / "embeddings.meta.json").write_text("{}", encoding="utf-8")
    (vector_dir / "index.faiss").write_bytes(b"fixture-index")
    (vector_dir / "index.meta.json").write_text(
        json.dumps(
            {
                "embedding_model": "fixture",
                "embedding_dim": 3,
                "vector_metric": "inner_product",
                "vector_normalized": True,
                "vector_count": 3,
                "source_vector_file_line_count": 2,
                "db_total_chunks": 3,
                "db_embedding_success_count": 3,
                "is_partial_embedding_index": False,
            }
        ),
        encoding="utf-8",
    )
    original_database_bytes = database.read_bytes()

    report = collect_incremental_baseline(
        root,
        include_sha256=True,
        count_jsonl_records=True,
    )

    assert report["knowledge_base_root"] == "fixture-kb"
    assert report["database_counts"] == {
        "documents": 2,
        "chunks": 3,
        "chunk_fts": 3,
    }
    assert report["vector_metadata"]["embedding_dim"] == 3
    assert report["jsonl_record_count"] == 2
    assert report["jsonl_record_count_source"] == "streamed_jsonl"
    assert report["assets"]["metadata_db"]["sha256"]
    assert database.read_bytes() == original_database_bytes


def test_baseline_recorder_uses_declared_jsonl_count_when_scan_is_not_requested(
    tmp_path: Path,
) -> None:
    """Fast development baseline avoids scanning multi-gigabyte JSONL files."""
    root = tmp_path / "fixture-kb"
    vector_dir = root / "vector_index"
    vector_dir.mkdir(parents=True)
    (vector_dir / "embeddings.jsonl").write_text("only-one-line\n", encoding="utf-8")
    (vector_dir / "index.meta.json").write_text(
        json.dumps({"source_vector_file_line_count": 999}), encoding="utf-8"
    )

    report = collect_incremental_baseline(root)

    assert report["jsonl_record_count"] == 999
    assert report["jsonl_record_count_source"] == "index_meta"
