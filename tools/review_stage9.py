#!/usr/bin/env python3
"""Review stage 9 FAISS index build results."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_preprocess.faiss_builder import (  # noqa: E402
    FaissUnavailableError,
    load_faiss_index,
    search_faiss,
)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_first_vector(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="检查阶段 9 FAISS 索引结果")
    parser.add_argument("--db", default="rag_data/metadata.db", help="数据库路径")
    parser.add_argument(
        "--vector-file",
        default="rag_data/vector_index/embeddings.jsonl",
        help="阶段 8 向量 JSONL 文件路径",
    )
    parser.add_argument(
        "--index",
        default="rag_data/vector_index/index.faiss",
        help="阶段 9 FAISS 索引路径",
    )
    parser.add_argument(
        "--meta",
        default="rag_data/vector_index/index.meta.json",
        help="阶段 9 索引元数据路径",
    )
    parser.add_argument("--sample", type=int, default=5, help="显示样例数量")
    parser.add_argument(
        "--skip-vector-count",
        action="store_true",
        help="跳过向量 JSONL 行数统计，适合超大文件快速检查",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    vector_path = Path(args.vector_file)
    index_path = Path(args.index)
    meta_path = Path(args.meta)

    issues: list[str] = []
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        issues.append(f"索引元数据不存在: {meta_path}")

    if not db_path.exists():
        print(f"[FAIL] 数据库不存在: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db_success = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'success'"
    ).fetchone()[0]
    db_mapped = conn.execute(
        """SELECT COUNT(*) FROM chunks
           WHERE embedding_status = 'success' AND vector_id IS NOT NULL"""
    ).fetchone()[0]

    vector_lines = None
    if not args.skip_vector_count:
        vector_lines = count_jsonl(vector_path)

    faiss_count = None
    top1_vector_id = None
    index = None
    if not index_path.exists():
        issues.append(f"FAISS 索引不存在: {index_path}")
    else:
        try:
            index = load_faiss_index(index_path)
            faiss_count = int(index.ntotal)
        except FaissUnavailableError as exc:
            issues.append(str(exc))
        except Exception as exc:
            issues.append(f"FAISS 索引加载失败: {exc}")

    if faiss_count is not None:
        if meta.get("vector_count") is not None and faiss_count != int(meta["vector_count"]):
            issues.append("FAISS 向量数与 index.meta.json 的 vector_count 不一致")
        if meta.get("build_limit") is None and faiss_count != db_mapped:
            issues.append("FAISS 向量数与数据库 success/vector_id 数量不一致")

    if vector_lines is not None and meta.get("build_limit") is None:
        if vector_lines != db_mapped:
            issues.append("向量文件行数与数据库 success/vector_id 数量不一致")

    first = read_first_vector(vector_path)
    if index is not None and first and isinstance(first.get("vector"), list):
        hits = search_faiss(first["vector"], top_k=1, index=index)
        if hits:
            top1_vector_id = hits[0].vector_id
            if top1_vector_id != first.get("vector_id"):
                issues.append(
                    "用向量文件第一条做 top1 自检时，返回的 vector_id 与自身不一致"
                )
        else:
            issues.append("用向量文件第一条做 top1 自检时没有返回结果")

    print(f"数据库: {db_path.resolve()}")
    print(f"向量文件: {vector_path.resolve()}")
    print(f"FAISS 索引: {index_path.resolve()}")
    print(f"索引元数据: {meta_path.resolve()}")
    print()
    print("=" * 70)
    print("  汇总")
    print("=" * 70)
    print(f"  chunks 总数:               {total_chunks}")
    print(f"  embedding success:         {db_success}")
    print(f"  success 且有 vector_id:    {db_mapped}")
    if vector_lines is not None:
        print(f"  向量文件行数:              {vector_lines}")
    else:
        print("  向量文件行数:              已跳过")
    print(f"  FAISS 向量数:              {faiss_count}")
    if meta:
        print(f"  模型:                      {meta.get('embedding_model')}")
        print(f"  维度:                      {meta.get('embedding_dim')}")
        print(f"  度量:                      {meta.get('vector_metric')}")
        print(f"  是否部分索引:              {meta.get('is_partial_embedding_index')}")
        print(f"  build_limit:               {meta.get('build_limit')}")
    print(f"  top1 自检 vector_id:       {top1_vector_id}")
    print()
    print(f"  状态: {'OK' if not issues else 'ISSUE: ' + '; '.join(issues)}")

    rows = conn.execute(
        """SELECT chunk_id, doc_id, chunk_index, vector_id, title, article_range
           FROM chunks
           WHERE embedding_status = 'success' AND vector_id IS NOT NULL
           ORDER BY vector_id
           LIMIT ?""",
        (args.sample,),
    ).fetchall()

    if rows:
        print()
        print("=" * 70)
        print("  样例")
        print("=" * 70)
        for row in rows:
            print(
                f"  vector_id={row['vector_id']} chunk_index={row['chunk_index']} "
                f"title={row['title']} range={row['article_range']}"
            )

    conn.close()


if __name__ == "__main__":
    main()
