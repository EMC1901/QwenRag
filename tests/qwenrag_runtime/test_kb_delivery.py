from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from local_rag_app.knowledge_base import KnowledgeBase
from qwenrag_runtime.kb_initializer import KnowledgeBaseInitializationError, initialize_empty_knowledge_base
from qwenrag_runtime.kb_snapshot import KnowledgeBaseSnapshotError, create_kb_snapshot
from qwenrag_runtime.runtime_lock import RuntimeLock


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ENTRY_POINT = ROOT / "packaging" / "stage_kb_snapshot.py"


def _settings(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        rag_knowledge_base_dir=root,
        upstream_embedding_model="embed-test",
        upstream_embedding_revision="artifact-test",
        rag_embedding_dim=3,
        rag_allow_partial_index=False,
        rag_enable_fts=True,
    )


def test_empty_knowledge_base_loads_and_returns_no_hits(tmp_path: Path) -> None:
    root = initialize_empty_knowledge_base(
        tmp_path / "空 知识库", embedding_model="embed-test", embedding_revision="artifact-test", embedding_dimension=3
    )
    knowledge_base = KnowledgeBase(_settings(root))

    knowledge_base.load()

    assert knowledge_base.index_metadata.vector_count == 0
    assert knowledge_base.search_vector([1, 0, 0], 5) == []
    assert knowledge_base.search_fts("任何内容", 5) == []


def test_empty_knowledge_base_never_overwrites_existing_customer_data(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    root.mkdir()
    (root / "customer-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(KnowledgeBaseInitializationError):
        initialize_empty_knowledge_base(root, embedding_model="embed", embedding_revision="rev", embedding_dimension=3)

    assert (root / "customer-file.txt").read_text(encoding="utf-8") == "keep"


def test_empty_knowledge_base_can_replace_only_the_first_install_workbench(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "workbench" / "01_请把新资料放这里").mkdir(parents=True)
    (root / "workbench" / "02_查看处理结果").mkdir()
    (root / "workbench" / "03_已处理资料归档").mkdir()

    initialize_empty_knowledge_base(
        root,
        embedding_model="embed-test",
        embedding_revision="artifact-test",
        embedding_dimension=3,
        allow_empty_workbench=True,
    )

    assert (root / "metadata.db").is_file()
    assert not (root / "workbench").exists()


def test_empty_knowledge_base_does_not_remove_workbench_with_customer_files(tmp_path: Path) -> None:
    root = tmp_path / "data"
    incoming = root / "workbench" / "01_请把新资料放这里"
    incoming.mkdir(parents=True)
    (incoming / "customer.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(KnowledgeBaseInitializationError):
        initialize_empty_knowledge_base(
            root,
            embedding_model="embed-test",
            embedding_revision="artifact-test",
            embedding_dimension=3,
            allow_empty_workbench=True,
        )


def test_snapshot_uses_sqlite_backup_and_excludes_wal_shm(tmp_path: Path) -> None:
    source = initialize_empty_knowledge_base(
        tmp_path / "source", embedding_model="embed-test", embedding_revision="artifact-test", embedding_dimension=3
    )
    (source / "metadata.db-wal").write_bytes(b"must-not-copy")
    (source / "metadata.db-shm").write_bytes(b"must-not-copy")

    destination = create_kb_snapshot(source, tmp_path / "snapshot", version="test-1")

    assert (destination / "SHA256SUMS.txt").is_file()
    assert not list(destination.rglob("*.db-wal"))
    assert not list(destination.rglob("*.db-shm"))
    knowledge_base = KnowledgeBase(_settings(destination))
    knowledge_base.load()
    assert knowledge_base.index_metadata.vector_count == 0


def test_snapshot_can_stamp_a_delivery_embedding_revision(tmp_path: Path) -> None:
    source = initialize_empty_knowledge_base(
        tmp_path / "source", embedding_model="embed-test", embedding_revision="legacy-build", embedding_dimension=3
    )

    destination = create_kb_snapshot(
        source,
        tmp_path / "snapshot",
        version="test-1",
        embedding_revision="delivery-artifact-v1",
    )

    metadata = __import__("json").loads((destination / "vector_index" / "index.meta.json").read_text(encoding="utf-8"))
    snapshot = __import__("json").loads((destination / "snapshot.json").read_text(encoding="utf-8"))
    assert metadata["embedding_revision"] == "delivery-artifact-v1"
    assert snapshot["embedding_contract"]["embedding_revision"] == "delivery-artifact-v1"


def test_snapshot_refuses_a_knowledge_base_in_use_by_rag(tmp_path: Path) -> None:
    source = initialize_empty_knowledge_base(
        tmp_path / "source", embedding_model="embed-test", embedding_revision="artifact-test", embedding_dimension=3
    )
    lock = RuntimeLock(tmp_path / "runtime" / "locks" / "rag.lock", mode="rag")
    lock.acquire()
    try:
        with pytest.raises(KnowledgeBaseSnapshotError):
            create_kb_snapshot(source, tmp_path / "snapshot", version="test-1")
    finally:
        lock.release()


def test_snapshot_entry_point_runs_without_pythonpath(tmp_path: Path) -> None:
    source = initialize_empty_knowledge_base(
        tmp_path / "source", embedding_model="embed-test", embedding_revision="artifact-test", embedding_dimension=3
    )
    destination = tmp_path / "snapshot"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(SNAPSHOT_ENTRY_POINT), "--source", str(source), "--destination", str(destination), "--version", "test-1"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (destination / "snapshot.json").is_file()
