"""Stage-12 Delta FTS/FAISS validation and manifest publication tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from local_rag_app.config import Settings
from local_rag_app.knowledge_base import KnowledgeBase
from rag_preprocess.incremental.delta import build_task_delta
from rag_preprocess.incremental.delta_index import build_delta_indexes, publish_delta_package, validate_delta_package
from rag_preprocess.incremental.manifest import load_manifest
from rag_preprocess.incremental.settings import load_incremental_settings
from tests.incremental.fixtures import EMBEDDING_DIM, EMBEDDING_MODEL, create_small_knowledge_base


class _Index:
    d = EMBEDDING_DIM
    ntotal = 1


class _SearchIndex:
    def __init__(self, *, vector_count: int, scores: list[float], ids: list[int]) -> None:
        self.d = EMBEDDING_DIM
        self.ntotal = vector_count
        self._scores = np.asarray([scores], dtype=np.float32)
        self._ids = np.asarray([ids], dtype=np.int64)

    def search(self, _query, _top_k):
        return self._scores, self._ids


def _create_base(destination: Path) -> Path:
    def write_placeholder(_vectors, vector_ids, target: Path) -> int:
        target.write_bytes(b"base-index")
        return len(vector_ids)

    return create_small_knowledge_base(destination, faiss_index_builder=write_placeholder)


def _write_ready_file(work: Path) -> dict[str, object]:
    version_id = "ver-update"
    (work / "vectors").mkdir(parents=True, exist_ok=True)
    (work / f"{version_id}.parsed.json").write_text(
        json.dumps(
            {
                "document": {
                    "title": "更新资料",
                    "parse_method": "txt",
                    "page_count": None,
                    "paragraph_count": 1,
                    "table_row_count": 0,
                    "warnings": [],
                    "blocks": [{"block_index": 0, "block_type": "paragraph", "text": "更新后的测试正文", "paragraph_index": 1}],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (work / f"{version_id}.chunks.json").write_text(
        json.dumps(
            {
                "chunks": [{"chunk_id": "updated-chunk", "chunk_index": 0, "chunk_text": "更新后的测试正文", "chunk_text_for_embedding": "更新后的测试正文", "title": "更新资料", "token_count": 8}]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (work / "vectors" / f"{version_id}.jsonl").write_text(
        json.dumps({"chunk_id": "updated-chunk", "model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM, "normalized": True, "vector": [1.0, 0.0, 0.0]}) + "\n",
        encoding="utf-8",
    )
    return {
        "file_name": "存量制度乙.txt",
        "logical_name_key": "存量制度乙.txt",
        "sha256": "e" * 64,
        "extension": ".txt",
        "size": 20,
        "doc_id": "fixture-doc-2",
        "version_id": version_id,
    }


def test_delta_indexes_and_manifest_publish_do_not_copy_base(tmp_path: Path) -> None:
    base = _create_base(tmp_path / "base")
    base_before = (base / "metadata.db").read_bytes()
    work_root = base / "incremental" / "work"
    task_id = "ingest-stage12"
    row = _write_ready_file(work_root / task_id)
    settings = load_incremental_settings(
        project_root=tmp_path,
        environ={
            "INCREMENTAL_KB_ROOT": "base",
            "INCREMENTAL_WORK_DIR": "base/incremental/work",
            "OCR_MODEL_DIR": "models/ocr",
            "EMBEDDING_MODEL": EMBEDDING_MODEL,
            "EMBEDDING_DIM": str(EMBEDDING_DIM),
        },
    )
    delta, _results, _meta = build_task_delta(
        work_root,
        task_id,
        [row],
        base_database=base / "metadata.db",
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
    )

    build_delta_indexes(
        delta,
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        faiss_builder=lambda _vectors, vector_ids, target: (target.write_bytes(b"delta-index"), len(vector_ids))[1],
    )
    validate_delta_package(
        delta,
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        faiss_loader=lambda _path: _Index(),
    )
    manifest = publish_delta_package(settings, delta, expected_revision=0)

    published = base / "kb_deltas" / f"delta-{task_id}"
    assert published.is_dir()
    assert (published / "vector_index" / "index.faiss").is_file()
    assert (published / "tombstones.json").is_file()
    assert manifest.revision == 1
    loaded = load_manifest(base)
    assert loaded is not None and [layer.layer_id for layer in loaded.deltas] == [f"delta-{task_id}"]
    assert (base / "metadata.db").read_bytes() == base_before
    assert not (published / "metadata.db").exists()


def test_manifest_runtime_queries_base_and_delta_and_filters_updated_vector(tmp_path: Path) -> None:
    base = _create_base(tmp_path / "base")
    work_root = base / "incremental" / "work"
    task_id = "ingest-runtime"
    row = _write_ready_file(work_root / task_id)
    incremental_settings = load_incremental_settings(
        project_root=tmp_path,
        environ={
            "INCREMENTAL_KB_ROOT": "base",
            "INCREMENTAL_WORK_DIR": "base/incremental/work",
            "OCR_MODEL_DIR": "models/ocr",
            "EMBEDDING_MODEL": EMBEDDING_MODEL,
            "EMBEDDING_DIM": str(EMBEDDING_DIM),
        },
    )
    delta, _results, _meta = build_task_delta(
        work_root,
        task_id,
        [row],
        base_database=base / "metadata.db",
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
    )
    build_delta_indexes(
        delta,
        embedding_dim=EMBEDDING_DIM,
        embedding_model=EMBEDDING_MODEL,
        faiss_builder=lambda _vectors, vector_ids, target: (target.write_bytes(b"delta-index"), len(vector_ids))[1],
    )
    validate_delta_package(delta, embedding_dim=EMBEDDING_DIM, embedding_model=EMBEDDING_MODEL, faiss_loader=lambda _path: _Index())
    publish_delta_package(incremental_settings, delta, expected_revision=0)

    def load_index(path: Path):
        if "kb_deltas" in path.parts:
            return _SearchIndex(vector_count=1, scores=[0.95], ids=[3])
        return _SearchIndex(vector_count=3, scores=[0.90, 0.80, float("-inf")], ids=[1, 0, -1])

    runtime_settings = Settings(
        _env_file=None,
        RAG_KNOWLEDGE_BASE_DIR=base,
        UPSTREAM_EMBEDDING_MODEL=EMBEDDING_MODEL,
        RAG_EMBEDDING_DIM=EMBEDDING_DIM,
    )
    knowledge_base = KnowledgeBase(runtime_settings, faiss_loader=load_index)
    knowledge_base.load()

    candidates = knowledge_base.search_vector_candidates([1.0, 0.0, 0.0], top_k=3)
    hits = knowledge_base.load_vector_hits(candidates)

    assert [candidate.vector_id for candidate in candidates] == [3, 0]
    assert [hit.chunk_id for hit in hits] == ["updated-chunk", "fixture-chunk-1"]
    assert "fixture-chunk-2" not in {
        candidate.chunk_id for candidate in knowledge_base.search_fts("存量制度乙", top_k=3)
    }
