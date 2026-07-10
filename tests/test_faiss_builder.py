import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rag_preprocess.faiss_builder import (
    build_faiss_index,
    load_embedding_jsonl,
    search_faiss,
)


TEST_DIR = Path("rag_data/test_stage9_tmp")


def _test_path(name: str) -> Path:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DIR / name


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_embedding_jsonl_valid():
    path = _test_path("embeddings_valid.jsonl")
    write_jsonl(
        path,
        [
            {"vector_id": 10, "chunk_id": "c1", "vector": [1.0, 0.0, 0.0]},
            {"vector_id": 11, "chunk_id": "c2", "vector": [0.0, 1.0, 0.0]},
        ],
    )

    loaded = load_embedding_jsonl(path, expected_dim=3)

    assert loaded.source_line_count == 2
    assert loaded.vectors.shape == (2, 3)
    assert loaded.vectors.dtype == np.float32
    assert loaded.vector_ids.tolist() == [10, 11]
    assert loaded.chunk_ids == ["c1", "c2"]


def test_load_embedding_jsonl_rejects_duplicate_vector_id():
    path = _test_path("embeddings_duplicate.jsonl")
    write_jsonl(
        path,
        [
            {"vector_id": 10, "chunk_id": "c1", "vector": [1.0, 0.0]},
            {"vector_id": 10, "chunk_id": "c2", "vector": [0.0, 1.0]},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate vector_id"):
        load_embedding_jsonl(path, expected_dim=2)


def test_load_embedding_jsonl_rejects_dim_mismatch():
    path = _test_path("embeddings_dim_mismatch.jsonl")
    write_jsonl(
        path,
        [{"vector_id": 10, "chunk_id": "c1", "vector": [1.0, 0.0]}],
    )

    with pytest.raises(ValueError, match="Vector dim mismatch"):
        load_embedding_jsonl(path, expected_dim=3)


def test_build_faiss_index_uses_id_map(monkeypatch):
    captured = {}

    class FakeBaseIndex:
        def __init__(self, dim):
            self.dim = dim

    class FakeIndexIDMap2:
        def __init__(self, base_index):
            self.base_index = base_index
            self.ntotal = 0

        def add_with_ids(self, matrix, ids):
            captured["matrix_dtype"] = matrix.dtype
            captured["ids"] = ids.tolist()
            captured["dim"] = self.base_index.dim
            self.ntotal = len(ids)

    def fake_write_index(index, path):
        Path(path).write_text(f"ntotal={index.ntotal}", encoding="utf-8")

    fake_faiss = types.SimpleNamespace(
        IndexFlatIP=FakeBaseIndex,
        IndexFlatL2=FakeBaseIndex,
        IndexIDMap2=FakeIndexIDMap2,
        write_index=fake_write_index,
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

    output = _test_path("index.faiss")
    count = build_faiss_index(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        [100, 101],
        output,
    )

    assert count == 2
    assert output.read_text(encoding="utf-8") == "ntotal=2"
    assert captured == {
        "matrix_dtype": np.dtype("float32"),
        "ids": [100, 101],
        "dim": 2,
    }


def test_search_faiss_returns_vector_ids():
    class FakeIndex:
        def search(self, query, top_k):
            assert query.shape == (1, 2)
            assert top_k == 2
            return (
                np.array([[0.9, 0.1]], dtype=np.float32),
                np.array([[100, -1]], dtype=np.int64),
            )

    results = search_faiss([1.0, 0.0], top_k=2, index=FakeIndex())

    assert len(results) == 1
    assert results[0].vector_id == 100
    assert results[0].score == pytest.approx(0.9)
