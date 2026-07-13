"""阶段 8 的 CLI 参数、致命失败和批次落盘行为。"""

import logging

import pytest

from rag_preprocess.database import connect_db, init_db, now_iso
from rag_preprocess.embedding_client import EmbeddingResult
from rag_preprocess.config import Config
from scripts import build_kb
from tools.check_embedding_consistency import check_embedding_consistency


def _db_with_chunks(tmp_path, count: int):
    output_dir = tmp_path / "rag_data"
    output_dir.mkdir()
    db_path = output_dir / "metadata.db"
    conn = connect_db(db_path)
    init_db(conn)
    conn.execute("PRAGMA foreign_keys=OFF")
    for index in range(count):
        conn.execute(
            """INSERT INTO chunks
               (chunk_id, doc_id, chunk_index, chunk_text, chunk_text_for_embedding, created_at)
               VALUES (?, 'doc', ?, 'text', 'text', ?)""",
            (f"c{index}", index, now_iso()),
        )
    conn.commit()
    conn.close()
    return output_dir, db_path


def test_batch_size_override_is_passed_to_config(monkeypatch):
    captured = []
    monkeypatch.setattr(build_kb, "run_stage_embed", lambda config, logger: captured.append(config))
    monkeypatch.setattr(build_kb.sys, "argv", ["build_kb.py", "--stage", "embed", "--embedding-batch-size", "64"])

    build_kb.main()

    assert captured[0].embedding_batch_size == 64


def test_invalid_batch_size_exits_nonzero(monkeypatch):
    monkeypatch.setattr(build_kb.sys, "argv", ["build_kb.py", "--stage", "embed", "--embedding-batch-size", "0"])

    with pytest.raises(SystemExit) as exc_info:
        build_kb.main()

    assert exc_info.value.code != 0


def test_preflight_failure_is_fatal_and_does_not_reset_progress(tmp_path, monkeypatch):
    output_dir, db_path = _db_with_chunks(tmp_path, 1)
    monkeypatch.setattr(
        "rag_preprocess.embedding_client.embed_batch",
        lambda *args, **kwargs: [EmbeddingResult(success=False, error_message="service down")],
    )
    config = Config(output_dir=output_dir, db_path=db_path)

    with pytest.raises(build_kb.StageExecutionError):
        build_kb.run_stage_embed(config, logging.getLogger("test"))

    conn = connect_db(db_path)
    assert conn.execute("SELECT embedding_status FROM chunks").fetchone()[0] is None
    conn.close()


def test_no_chunks_is_a_successful_noop(tmp_path):
    output_dir, db_path = _db_with_chunks(tmp_path, 0)

    build_kb.run_stage_embed(Config(output_dir=output_dir, db_path=db_path), logging.getLogger("test"))


def test_batch_jsonl_is_durable_before_sqlite_mapping(tmp_path, monkeypatch):
    output_dir, db_path = _db_with_chunks(tmp_path, 2)
    responses = [
        [EmbeddingResult(success=True, vector=[1.0] * 1024)],
        [
            EmbeddingResult(success=True, vector=[1.0] * 1024),
            EmbeddingResult(success=True, vector=[2.0] * 1024),
        ],
    ]
    monkeypatch.setattr(
        "rag_preprocess.embedding_client.embed_batch",
        lambda *args, **kwargs: responses.pop(0),
    )
    config = Config(output_dir=output_dir, db_path=db_path, embedding_batch_size=2)

    build_kb.run_stage_embed(config, logging.getLogger("test"))

    report = check_embedding_consistency(
        db_path,
        output_dir / "vector_index" / "embeddings.jsonl",
        mode="full",
    )
    assert report.is_complete
