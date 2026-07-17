"""Tests for the stage-0 disposable knowledge-base fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from local_rag_app.config import Settings
from local_rag_app.knowledge_base import KnowledgeBase
from tests.incremental.fixtures import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    create_small_knowledge_base,
    temporary_small_knowledge_base,
)


class _FixtureIndex:
    d = EMBEDDING_DIM
    ntotal = 3


def _write_test_index(_vectors: np.ndarray, ids: list[int], output: Path) -> int:
    """Avoid requiring optional faiss-cpu for this fast fixture contract test."""
    output.write_bytes(b"fixture-faiss-placeholder")
    return len(ids)


def _settings(root: Path) -> Settings:
    return Settings(
        _env_file=None,
        RAG_KNOWLEDGE_BASE_DIR=root,
        UPSTREAM_EMBEDDING_MODEL=EMBEDDING_MODEL,
        RAG_EMBEDDING_DIM=EMBEDDING_DIM,
    )


def test_small_fixture_has_three_unique_files_and_consistent_assets(tmp_path: Path) -> None:
    """Normal fixture is isolated, has unique full filenames, and has all assets."""
    root = create_small_knowledge_base(
        tmp_path / "small-kb",
        faiss_index_builder=_write_test_index,
    )

    with sqlite3.connect(root / "metadata.db") as connection:
        rows = connection.execute(
            "SELECT file_name FROM source_files ORDER BY source_file_id"
        ).fetchall()
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == 3
    names = [row[0] for row in rows]
    assert len({name.casefold() for name in names}) == 3

    vectors = [json.loads(line) for line in (root / "vector_index/embeddings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [vector["vector_id"] for vector in vectors] == [0, 1, 2]
    assert all(vector["dim"] == EMBEDDING_DIM for vector in vectors)


def test_fixture_is_loadable_by_current_knowledge_base_contract(tmp_path: Path) -> None:
    """Fixture metadata, SQLite mapping, and FAISS contract load together."""
    root = create_small_knowledge_base(
        tmp_path / "small-kb",
        faiss_index_builder=_write_test_index,
    )
    knowledge_base = KnowledgeBase(_settings(root), faiss_loader=lambda _path: _FixtureIndex())

    knowledge_base.load()

    assert knowledge_base.is_ready
    assert knowledge_base.index_metadata.embedding_dim == EMBEDDING_DIM
    assert knowledge_base.index_metadata.vector_count == 3


def test_duplicate_fixture_intentionally_violates_case_insensitive_file_name_rule(
    tmp_path: Path,
) -> None:
    """Later identity checks have a deterministic duplicate-name fixture."""
    root = create_small_knowledge_base(
        tmp_path / "duplicate-kb",
        duplicate_logical_name=True,
        faiss_index_builder=_write_test_index,
    )
    with sqlite3.connect(root / "metadata.db") as connection:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT file_name FROM source_files ORDER BY source_file_id"
            )
        ]

    assert len(names) == 3
    assert len({name.casefold() for name in names}) == 2


def test_fixture_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    """Fixture construction cannot accidentally overwrite any test asset."""
    destination = tmp_path / "already-exists"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        create_small_knowledge_base(destination, faiss_index_builder=_write_test_index)


def test_temporary_fixture_is_removed_after_context_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One context-managed command creates and then destroys only synthetic assets."""
    monkeypatch.setattr("tests.incremental.fixtures.tempfile.tempdir", str(tmp_path))

    with temporary_small_knowledge_base(faiss_index_builder=_write_test_index) as root:
        assert root.is_dir()
        assert (root / "metadata.db").is_file()

    assert not root.exists()
