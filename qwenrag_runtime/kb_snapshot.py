"""Create a consistent, hash-verified knowledge-base snapshot without WAL files."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import uuid

from rag_preprocess.faiss_builder import load_faiss_index
from rag_preprocess.incremental.manifest import layer_directory, load_manifest
from qwenrag_runtime.runtime_lock import is_runtime_lock_held


class KnowledgeBaseSnapshotError(RuntimeError):
    """Raised when a source cannot be safely snapshotted."""


def create_kb_snapshot(
    source_root: Path,
    destination: Path,
    *,
    version: str,
    embedding_revision: str | None = None,
    runtime_lock_path: Path | None = None,
) -> Path:
    """Backup SQLite consistently, copy immutable assets, verify, then atomically publish."""
    if destination.exists() and any(destination.iterdir()):
        raise KnowledgeBaseSnapshotError("快照目标已存在，拒绝覆盖")
    lock_path = runtime_lock_path or source_root.parent / "runtime" / "locks" / "rag.lock"
    if is_runtime_lock_held(lock_path):
        raise KnowledgeBaseSnapshotError("QwenRAG 正在运行，请停止后再创建知识库快照")
    manifest = load_manifest(source_root)
    layers = [(source_root, False)] if manifest is None else [(layer_directory(source_root, manifest.base), False), *[(layer_directory(source_root, item), True) for item in manifest.deltas]]
    source_files = _source_assets(source_root, layers)
    before = {_relative(source_root, item): _fingerprint(item) for item in source_files}
    # Keep temporary directories ASCII for broad Windows filesystem/toolchain
    # compatibility; the published snapshot name may still contain Unicode.
    staging = destination.parent / f".qwenrag-kb-snapshot-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True)
        for layer_root, is_delta in layers:
            relative_root = _relative(source_root, layer_root)
            output_root = staging / relative_root
            output_root.mkdir(parents=True, exist_ok=True)
            _backup_database(layer_root / ("delta.db" if is_delta else "metadata.db"), output_root / ("delta.db" if is_delta else "metadata.db"))
            # Keep every published delta asset.  The running RAG reads the
            # SQLite/FAISS pair, while the JSONL and tombstone files preserve
            # the complete published Delta for later diagnostics or rebuilds.
            for name in (
                "delta.meta.json",
                "tombstones.json",
                "vector_index/embeddings.jsonl",
            ) if is_delta else ():
                target = output_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(layer_root / name, target)
            vector_source, vector_target = layer_root / "vector_index", output_root / "vector_index"
            vector_target.mkdir(exist_ok=True)
            for name in ("index.faiss", "index.meta.json"):
                shutil.copy2(vector_source / name, vector_target / name)
        if manifest is not None:
            shutil.copy2(source_root / "knowledge_manifest.json", staging / "knowledge_manifest.json")
        after = {_relative(source_root, item): _fingerprint(item) for item in source_files}
        if before != after:
            raise KnowledgeBaseSnapshotError("快照期间源知识库发生变化")
        contract = _apply_embedding_contract(staging, embedding_revision)
        _verify_snapshot(staging)
        _write_hash_manifest(staging, version, contract)
        if destination.exists():
            destination.rmdir()
        # Prefer a same-volume rename, with a Windows-safe fallback for an
        # endpoint security product briefly locking a Unicode target name.
        shutil.move(str(staging), str(destination))
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, KnowledgeBaseSnapshotError):
            raise
        raise KnowledgeBaseSnapshotError("创建知识库快照失败") from exc
    return destination


def _source_assets(source_root: Path, layers: list[tuple[Path, bool]]) -> list[Path]:
    files: list[Path] = []
    for layer, is_delta in layers:
        files.extend([layer / ("delta.db" if is_delta else "metadata.db"), layer / "vector_index" / "index.faiss", layer / "vector_index" / "index.meta.json"])
        if is_delta:
            files.extend(
                (
                    layer / "delta.meta.json",
                    layer / "tombstones.json",
                    layer / "vector_index" / "embeddings.jsonl",
                )
            )
    manifest = source_root / "knowledge_manifest.json"
    if manifest.exists():
        files.append(manifest)
    if any(not item.is_file() for item in files):
        raise KnowledgeBaseSnapshotError("源知识库资产不完整")
    return files


def _backup_database(source: Path, destination: Path) -> None:
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as reader, closing(sqlite3.connect(destination)) as writer:
        reader.backup(writer)
    with closing(sqlite3.connect(destination)) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok" or connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise KnowledgeBaseSnapshotError("快照数据库一致性校验失败")


def _verify_snapshot(root: Path) -> None:
    for database in root.rglob("*.db"):
        with closing(sqlite3.connect(database)) as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM chunks WHERE embedding_status='success' AND vector_id IS NOT NULL").fetchone()[0])
        metadata = json.loads((database.parent / "vector_index" / "index.meta.json").read_text(encoding="utf-8"))
        index = load_faiss_index(database.parent / "vector_index" / "index.faiss")
        if count != metadata.get("vector_count") or getattr(index, "ntotal", None) != count or getattr(index, "d", None) != metadata.get("embedding_dim"):
            raise KnowledgeBaseSnapshotError("快照向量资产不一致")


def _apply_embedding_contract(root: Path, override_revision: str | None) -> dict[str, object]:
    """Make a legacy source explicit about the embedding artifact it requires."""
    base_metadata_path = root / "vector_index" / "index.meta.json"
    try:
        base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseSnapshotError("快照缺少可读的向量索引元数据") from exc
    model = base_metadata.get("embedding_model")
    dimension = base_metadata.get("embedding_dim")
    normalized = base_metadata.get("vector_normalized")
    metric = base_metadata.get("vector_metric")
    source_revision = base_metadata.get("embedding_revision", "legacy-unknown")
    revision = (override_revision or source_revision).strip() if isinstance(override_revision or source_revision, str) else ""
    if not isinstance(model, str) or not model.strip() or isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0 or normalized is not True or metric != "inner_product" or not revision or revision == "legacy-unknown":
        raise KnowledgeBaseSnapshotError("初始知识库缺少可交付的 Embedding 制品标识")

    metadata_paths = list(root.rglob("vector_index/index.meta.json"))
    for path in metadata_paths:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if raw.get("embedding_model") != model or raw.get("embedding_dim") != dimension or raw.get("vector_normalized") is not True or raw.get("vector_metric") != "inner_product":
            raise KnowledgeBaseSnapshotError("初始知识库各层 Embedding 契约不一致")
        raw["embedding_revision"] = revision
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = root / "knowledge_manifest.json"
    if manifest_path.is_file():
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        raw_manifest["embedding_revision"] = revision
        for layer in raw_manifest.get("deltas", []):
            if not isinstance(layer, dict) or not isinstance(layer.get("relative_path"), str):
                raise KnowledgeBaseSnapshotError("初始知识库清单无效")
            delta_meta = root / layer["relative_path"] / "delta.meta.json"
            raw_delta = json.loads(delta_meta.read_text(encoding="utf-8-sig"))
            raw_delta["embedding_revision"] = revision
            delta_meta.write_text(json.dumps(raw_delta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            layer["meta_sha256"] = _sha256(delta_meta)
        manifest_path.write_text(json.dumps(raw_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "embedding_model": model,
        "embedding_revision": revision,
        "embedding_dimension": dimension,
        "vector_normalized": True,
        "vector_metric": "inner_product",
    }


def _write_hash_manifest(root: Path, version: str, contract: dict[str, object]) -> None:
    entries = []
    for item in sorted(path for path in root.rglob("*") if path.is_file()):
        if item.name in {"SHA256SUMS.txt", "snapshot.json"}:
            continue
        entries.append(f"{_sha256(item)}  {item.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    (root / "snapshot.json").write_text(json.dumps({"version": version, "created_at": datetime.now(timezone.utc).isoformat(), "embedding_contract": contract}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> Path:
    return path.resolve().relative_to(root.resolve())


def _fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns
