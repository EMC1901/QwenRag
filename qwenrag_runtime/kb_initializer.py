"""Create a valid, immutable zero-vector Base knowledge base for first install."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import uuid

from rag_preprocess.database import init_db
from rag_preprocess.faiss_builder import write_faiss_index


class KnowledgeBaseInitializationError(RuntimeError):
    """Raised without modifying an existing customer knowledge base."""


def initialize_empty_knowledge_base(
    root: Path,
    *,
    embedding_model: str,
    embedding_revision: str,
    embedding_dimension: int,
    allow_empty_workbench: bool = False,
) -> Path:
    """Atomically create an empty Base; existing data is never overwritten."""
    if root.exists() and any(root.iterdir()) and not (
        allow_empty_workbench and _is_empty_workbench_root(root)
    ):
        raise KnowledgeBaseInitializationError("目标知识库目录已存在，拒绝覆盖")
    if not embedding_model.strip() or not embedding_revision.strip() or embedding_dimension <= 0:
        raise KnowledgeBaseInitializationError("空知识库配置无效")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Keep the transient name ASCII.  Some Windows toolchains reject a rename
    # of a dot-prefixed Unicode staging directory even though the final user
    # data directory may (and commonly will) contain Chinese characters.
    staging = parent / f".qwenrag-kb-initializing-{uuid.uuid4().hex}"
    try:
        vector_dir = staging / "vector_index"
        vector_dir.mkdir(parents=True)
        database = staging / "metadata.db"
        with closing(sqlite3.connect(database)) as connection:
            init_db(connection)
        _write_empty_faiss(vector_dir / "index.faiss", embedding_dimension)
        metadata = {
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "embedding_dim": embedding_dimension,
            "vector_metric": "inner_product",
            "vector_normalized": True,
            "vector_count": 0,
            "is_partial_embedding_index": False,
            "index_type": "faiss",
            "faiss_factory": "IndexIDMap2(IndexFlatIP)",
        }
        (vector_dir / "index.meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "revision": 0,
            "base": {"generation_id": "empty-base", "relative_path": "."},
            "deltas": [],
            "next_vector_id": 0,
            "embedding_model": embedding_model,
            "embedding_revision": embedding_revision,
            "embedding_dim": embedding_dimension,
            "vector_metric": "inner_product",
            "vector_normalized": True,
        }
        (staging / "knowledge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if root.exists():
            # Runtime/config initialization legitimately creates the empty
            # customer-visible workbench before a KB choice is made. It is the
            # only existing content this installer-only mode may remove.
            shutil.rmtree(root / "workbench")
            root.rmdir()
        # Prefer a same-volume rename.  ``shutil.move`` delegates to that
        # operation and has a Windows-safe copy fallback for endpoint security
        # products that transiently reject a Unicode directory rename.
        shutil.move(str(staging), str(root))
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, KnowledgeBaseInitializationError):
            raise
        raise KnowledgeBaseInitializationError("无法创建空知识库") from exc
    return root


def _write_empty_faiss(path: Path, dimension: int) -> None:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise KnowledgeBaseInitializationError("当前运行时缺少 FAISS") from exc
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    write_faiss_index(index, path)


def _is_empty_workbench_root(root: Path) -> bool:
    """Accept only the empty workbench created by RuntimePaths on first install."""
    workbench = root / "workbench"
    if not root.is_dir() or not workbench.is_dir():
        return False
    if any(item != workbench for item in root.iterdir()):
        return False
    return not any(item.is_file() for item in workbench.rglob("*"))
