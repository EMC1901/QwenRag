#!/usr/bin/env python3
"""Read and record the knowledge-base baseline without modifying its assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


_ASSET_PATHS = {
    "metadata_db": Path("metadata.db"),
    "embeddings_jsonl": Path("vector_index") / "embeddings.jsonl",
    "embeddings_meta": Path("vector_index") / "embeddings.meta.json",
    "index_faiss": Path("vector_index") / "index.faiss",
    "index_meta": Path("vector_index") / "index.meta.json",
}


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 so large assets never enter memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def collect_incremental_baseline(
    knowledge_base_root: Path,
    *,
    include_sha256: bool = False,
    count_jsonl_records: bool = False,
) -> dict[str, Any]:
    """Collect sizes, database counts, and declared vector metadata read-only.

    Exact JSONL line counting and SHA-256 are opt-in because both may scan a
    multi-gigabyte delivery asset.  Without those options the report retains
    the count declared by ``index.meta.json`` and marks its source explicitly.
    """
    root = knowledge_base_root.resolve(strict=False)
    report: dict[str, Any] = {
        "knowledge_base_root": root.name,
        "assets": {},
        "database_counts": {},
        "vector_metadata": {},
    }
    for name, relative_path in _ASSET_PATHS.items():
        path = root / relative_path
        asset: dict[str, Any] = {"relative_path": relative_path.as_posix(), "exists": path.is_file()}
        if path.is_file():
            asset["size_bytes"] = path.stat().st_size
            if include_sha256:
                asset["sha256"] = sha256_file(path)
        report["assets"][name] = asset

    metadata_db = root / _ASSET_PATHS["metadata_db"]
    if metadata_db.is_file():
        report["database_counts"] = _read_database_counts(metadata_db)

    index_meta_path = root / _ASSET_PATHS["index_meta"]
    if index_meta_path.is_file():
        index_meta = _read_json_object(index_meta_path)
        report["vector_metadata"] = {
            key: index_meta.get(key)
            for key in (
                "embedding_model",
                "embedding_dim",
                "vector_metric",
                "vector_normalized",
                "vector_count",
                "source_vector_file_line_count",
                "db_total_chunks",
                "db_embedding_success_count",
                "is_partial_embedding_index",
            )
        }

    jsonl_path = root / _ASSET_PATHS["embeddings_jsonl"]
    if jsonl_path.is_file():
        if count_jsonl_records:
            report["jsonl_record_count"] = _count_nonempty_lines(jsonl_path)
            report["jsonl_record_count_source"] = "streamed_jsonl"
        else:
            report["jsonl_record_count"] = report["vector_metadata"].get(
                "source_vector_file_line_count"
            )
            report["jsonl_record_count_source"] = "index_meta"
    return report


def _read_database_counts(path: Path) -> dict[str, int]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        counts = {}
        for table_name in ("documents", "chunks", "chunk_fts"):
            try:
                counts[table_name] = int(
                    connection.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
                )
            except sqlite3.OperationalError:
                counts[table_name] = -1
        return counts
    finally:
        connection.close()


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path.name}")
    return value


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="只读记录增量入库开发基线")
    parser.add_argument("--kb-root", default="rag_data", help="知识库根目录")
    parser.add_argument("--sha256", action="store_true", help="流式计算所有资产 SHA-256（耗时）")
    parser.add_argument(
        "--count-jsonl",
        action="store_true",
        help="流式计数 JSONL 非空行（大型文件耗时）",
    )
    args = parser.parse_args()
    report = collect_incremental_baseline(
        Path(args.kb_root),
        include_sha256=args.sha256,
        count_jsonl_records=args.count_jsonl,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
