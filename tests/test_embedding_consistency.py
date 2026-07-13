"""阶段 8 SQLite/JSONL 一致性检查的回归测试。"""

import json
from pathlib import Path

from rag_preprocess.database import connect_db, init_db, now_iso
from tools.check_embedding_consistency import check_embedding_consistency


def _prepare_db(tmp_path: Path, statuses: list[tuple[str, int | None, str | None]]) -> Path:
    db_path = tmp_path / "metadata.db"
    conn = connect_db(db_path)
    init_db(conn)
    conn.execute("PRAGMA foreign_keys=OFF")
    for index, (chunk_id, vector_id, status) in enumerate(statuses):
        conn.execute(
            """INSERT INTO chunks
               (chunk_id, doc_id, chunk_index, chunk_text, chunk_text_for_embedding,
                created_at, vector_id, embedding_status)
               VALUES (?, 'doc', ?, 'text', 'text', ?, ?, ?)""",
            (chunk_id, index, now_iso(), vector_id, status),
        )
    conn.commit()
    conn.close()
    return db_path


def _record(vector_id: int, chunk_id: str, vector: list[float] | None = None) -> dict:
    return {
        "vector_id": vector_id,
        "chunk_id": chunk_id,
        "model": "qwen3-embedding-0.6b",
        "dim": 1024,
        "normalized": True,
        "vector": vector if vector is not None else [0.0] * 1024,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_partial_but_consistent_requires_complete(tmp_path: Path):
    db = _prepare_db(tmp_path, [("c1", 0, "success"), ("c2", None, None)])
    vectors = tmp_path / "embeddings.jsonl"
    _write_jsonl(vectors, [_record(0, "c1")])

    report = check_embedding_consistency(db, vectors, mode="quick")

    assert report.is_consistent
    assert report.exit_code(require_complete=False) == 0
    assert report.exit_code(require_complete=True) == 2


def test_complete_full_check_is_ok(tmp_path: Path):
    db = _prepare_db(tmp_path, [("c1", 0, "success"), ("c2", 1, "success")])
    vectors = tmp_path / "embeddings.jsonl"
    _write_jsonl(vectors, [_record(0, "c1"), _record(1, "c2")])

    report = check_embedding_consistency(db, vectors, mode="full")

    assert report.is_complete
    assert report.exit_code(require_complete=True) == 0


def test_db_success_more_than_jsonl_is_inconsistent(tmp_path: Path):
    db = _prepare_db(tmp_path, [("c1", 0, "success"), ("c2", 1, "success")])
    vectors = tmp_path / "embeddings.jsonl"
    _write_jsonl(vectors, [_record(0, "c1")])

    assert check_embedding_consistency(db, vectors).exit_code(False) == 1


def test_extra_jsonl_record_is_inconsistent(tmp_path: Path):
    db = _prepare_db(tmp_path, [("c1", 0, "success")])
    vectors = tmp_path / "embeddings.jsonl"
    _write_jsonl(vectors, [_record(0, "c1"), _record(1, "c2")])

    assert check_embedding_consistency(db, vectors).exit_code(False) == 1


def test_full_check_rejects_duplicate_vector_and_chunk_ids(tmp_path: Path):
    db = _prepare_db(tmp_path, [("c1", 0, "success"), ("c2", 1, "success")])
    vectors = tmp_path / "embeddings.jsonl"
    _write_jsonl(vectors, [_record(0, "c1"), _record(0, "c1")])

    assert check_embedding_consistency(db, vectors, mode="full").exit_code(False) == 1


def test_check_rejects_duplicate_vector_id_in_sqlite(tmp_path: Path):
    db = _prepare_db(tmp_path, [("c1", 0, "success"), ("c2", 0, "success")])
    vectors = tmp_path / "embeddings.jsonl"
    _write_jsonl(vectors, [_record(0, "c1"), _record(0, "c2")])

    assert check_embedding_consistency(db, vectors, mode="full").exit_code(False) == 1


def test_full_check_rejects_truncated_last_json_line(tmp_path: Path):
    db = _prepare_db(tmp_path, [("c1", 0, "success")])
    vectors = tmp_path / "embeddings.jsonl"
    vectors.write_text('{"vector_id": 0', encoding="utf-8")

    assert check_embedding_consistency(db, vectors, mode="full").exit_code(False) == 1


def test_full_check_rejects_wrong_dimension_and_nonfinite_values(tmp_path: Path):
    db = _prepare_db(tmp_path, [("c1", 0, "success")])
    vectors = tmp_path / "embeddings.jsonl"
    wrong = _record(0, "c1", [float("nan")] * 1024)
    wrong["dim"] = 3
    _write_jsonl(vectors, [wrong])

    assert check_embedding_consistency(db, vectors, mode="full").exit_code(False) == 1
