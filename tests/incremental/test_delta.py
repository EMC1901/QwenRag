"""Stage-11 Delta database and vector package tests."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from rag_preprocess.incremental.delta import (
    DeltaError,
    build_task_delta,
    create_task_delta,
    initial_next_vector_id,
    prepare_delta_vectors,
    resolve_active_document,
    validate_delta_database,
    write_delta_embeddings,
    write_delta_file,
    write_delta_metadata,
)
from tests.incremental.fixtures import EMBEDDING_DIM, EMBEDDING_MODEL, create_small_knowledge_base


def _base(destination: Path) -> Path:
    """Stage 11 needs legacy SQLite metadata, not a real FAISS runtime."""

    def write_placeholder_index(_vectors, vector_ids, target: Path) -> int:
        target.write_bytes(b"stage-11-fixture-index")
        return len(vector_ids)

    return create_small_knowledge_base(destination, faiss_index_builder=write_placeholder_index)


def _row(*, name: str, doc_id: str, version_id: str, digest: str, action: str) -> dict[str, object]:
    return {
        "file_name": name,
        "logical_name_key": name.casefold(),
        "sha256": digest,
        "extension": ".txt",
        "size": 12,
        "doc_id": doc_id,
        "version_id": version_id,
        "action": action,
    }


def _write_file_intermediates(work: Path, version_id: str, chunk_id: str) -> None:
    (work / "vectors").mkdir(parents=True, exist_ok=True)
    (work / f"{version_id}.parsed.json").write_text(
        json.dumps(
            {
                "document": {
                    "title": "测试资料",
                    "parse_method": "txt",
                    "page_count": None,
                    "paragraph_count": 1,
                    "table_row_count": 0,
                    "warnings": [],
                    "blocks": [
                        {
                            "block_index": 0,
                            "block_type": "paragraph",
                            "text": "测试资料正文",
                            "paragraph_index": 1,
                            "table_index": None,
                            "row_index": None,
                            "style_name": None,
                            "page_number": None,
                            "ocr_confidence": None,
                            "quality_status": "ok",
                            "source_locator": "paragraph:1",
                        }
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (work / f"{version_id}.chunks.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": 0,
                        "chunk_text": "测试资料正文",
                        "chunk_text_for_embedding": "测试资料正文",
                        "title": "测试资料",
                        "section_path": None,
                        "article_no": None,
                        "article_range": None,
                        "paragraph_start": 1,
                        "paragraph_end": 1,
                        "token_count": 6,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (work / "vectors" / f"{version_id}.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": chunk_id,
                "model": EMBEDDING_MODEL,
                "dim": EMBEDDING_DIM,
                "normalized": True,
                "vector": [1.0, 0.0, 0.0],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_task_delta_is_empty_small_database_and_never_copies_base(tmp_path: Path) -> None:
    base = _base(tmp_path / "base")
    before = (base / "metadata.db").read_bytes()

    delta = create_task_delta(tmp_path / "work", "ingest-1")

    assert (delta / "delta.db").is_file()
    assert not (delta / "metadata.db").exists()
    assert not (delta / "vector_index" / "embeddings.jsonl").exists()
    assert not (delta / "vector_index" / "index.faiss").exists()
    assert (base / "metadata.db").read_bytes() == before
    assert initial_next_vector_id(base / "metadata.db") == 3


def test_new_file_creates_only_delta_records_and_task_vectors(tmp_path: Path) -> None:
    base = _base(tmp_path / "base")
    before = (base / "metadata.db").read_bytes()
    work = tmp_path / "work" / "ingest-new"
    row = _row(name="新增.txt", doc_id="new-doc", version_id="ver-new", digest="a" * 64, action="NEW")
    _write_file_intermediates(work, "ver-new", "new-chunk")
    delta = create_task_delta(tmp_path / "work", "ingest-new")

    prepared = prepare_delta_vectors(
        delta, row, work, first_vector_id=3, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL
    )
    result = write_delta_file(delta, row, work, prepared.vector_ids_by_chunk, task_id="ingest-new", prior=None)
    count = write_delta_embeddings(delta, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)
    metadata = write_delta_metadata(
        delta,
        delta_id="delta-ingest-new",
        task_id="ingest-new",
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
        parent_manifest_revision=None,
    )

    assert result.chunk_count == 1
    assert count == 1
    assert validate_delta_database(delta) == {
        "document_count": 1,
        "chunk_count": 1,
        "version_count": 1,
        "tombstone_count": 0,
    }
    vector = json.loads((delta / "vector_index" / "embeddings.jsonl").read_text(encoding="utf-8"))
    assert vector["vector_id"] == 3
    assert metadata["validation_status"] == "pending_stage_12"
    assert (base / "metadata.db").read_bytes() == before


def test_update_writes_version_chunk_and_vector_tombstones_without_mutating_base(tmp_path: Path) -> None:
    base = _base(tmp_path / "base")
    before = (base / "metadata.db").read_bytes()
    work = tmp_path / "work" / "ingest-update"
    row = _row(
        name="存量制度乙.txt",
        doc_id="fixture-doc-2",
        version_id="ver-update",
        digest="b" * 64,
        action="UPDATE",
    )
    _write_file_intermediates(work, "ver-update", "updated-chunk")
    prior = resolve_active_document(row["logical_name_key"], base_database=base / "metadata.db")
    assert prior is not None
    assert prior.chunk_ids == ("fixture-chunk-2",)
    assert prior.vector_ids == (1,)
    delta = create_task_delta(tmp_path / "work", "ingest-update")
    prepared = prepare_delta_vectors(
        delta, row, work, first_vector_id=3, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL
    )
    write_delta_file(delta, row, work, prepared.vector_ids_by_chunk, task_id="ingest-update", prior=prior)
    write_delta_embeddings(delta, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL)

    with sqlite3.connect(delta / "delta.db") as connection:
        tombstones = set(connection.execute("SELECT entity_type, entity_id FROM delta_tombstones"))
        identity = connection.execute(
            "SELECT doc_id, active_version_id FROM document_identities WHERE logical_name_key=?",
            ("存量制度乙.txt",),
        ).fetchone()
    assert ("doc_version", prior.version_id) in tombstones
    assert ("chunk", "fixture-chunk-2") in tombstones
    assert ("vector", "1") in tombstones
    assert identity == ("fixture-doc-2", "ver-update")
    assert (base / "metadata.db").read_bytes() == before


def test_failed_file_transaction_does_not_leave_identity_or_version(tmp_path: Path) -> None:
    base = _base(tmp_path / "base")
    work = tmp_path / "work" / "ingest-failed"
    row = _row(name="失败.txt", doc_id="failed-doc", version_id="ver-failed", digest="c" * 64, action="NEW")
    _write_file_intermediates(work, "ver-failed", "failed-chunk")
    delta = create_task_delta(tmp_path / "work", "ingest-failed")

    with pytest.raises(DeltaError, match="DELTA_VECTOR_MAPPING_INVALID"):
        write_delta_file(delta, row, work, {}, task_id="ingest-failed", prior=None)

    with sqlite3.connect(delta / "delta.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM document_identities").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 0


def test_task_delta_builder_resumes_committed_file_without_duplicate_vectors(tmp_path: Path) -> None:
    base = _base(tmp_path / "base")
    work_root = tmp_path / "work"
    task_id = "ingest-resume"
    row = _row(name="恢复.txt", doc_id="resume-doc", version_id="ver-resume", digest="d" * 64, action="NEW")
    _write_file_intermediates(work_root / task_id, "ver-resume", "resume-chunk")

    first_root, first_results, first_metadata = build_task_delta(
        work_root,
        task_id,
        [row],
        base_database=base / "metadata.db",
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
    )
    second_root, second_results, second_metadata = build_task_delta(
        work_root,
        task_id,
        [row],
        base_database=base / "metadata.db",
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
    )

    assert first_root == second_root
    assert first_results["ver-resume"].already_written is False
    assert second_results["ver-resume"].already_written is True
    assert first_metadata["vector_count"] == second_metadata["vector_count"] == 1
    assert _line_count(first_root / "vector_index" / "embeddings.jsonl") == 1


def _line_count(path: Path) -> int:
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line])
