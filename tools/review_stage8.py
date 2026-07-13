#!/usr/bin/env python3
"""检查阶段8 embedding 向量化结果。"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="检查阶段8 embedding 结果")
    parser.add_argument("--db", default="rag_data/metadata.db", help="数据库路径")
    parser.add_argument(
        "--vector-file",
        default="rag_data/vector_index/embeddings.jsonl",
        help="阶段8向量 JSONL 文件路径",
    )
    parser.add_argument(
        "--meta",
        default="rag_data/vector_index/embeddings.meta.json",
        help="阶段8向量元数据 JSON 文件路径",
    )
    parser.add_argument("--sample", type=int, default=5, help="显示样例数量")
    args = parser.parse_args()

    db_path = Path(args.db)
    vector_path = Path(args.vector_file)
    meta_path = Path(args.meta)

    if not db_path.exists() or not vector_path.exists():
        missing = db_path if not db_path.exists() else vector_path
        print(f"[FAIL] 必需文件不存在: {missing}")
        sys.exit(2)
    if not db_path.is_file():
        print(f"[FAIL] 数据库不存在: {db_path}")
        sys.exit(2)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    success = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'success'"
    ).fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'failed'"
    ).fetchone()[0]
    mapped = conn.execute(
        """SELECT COUNT(*) FROM chunks
           WHERE embedding_status = 'success' AND vector_id IS NOT NULL"""
    ).fetchone()[0]
    missing_vector_id = conn.execute(
        """SELECT COUNT(*) FROM chunks
           WHERE embedding_status = 'success' AND vector_id IS NULL"""
    ).fetchone()[0]

    vector_lines = count_jsonl(vector_path)
    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    print(f"数据库: {db_path.resolve()}")
    print(f"向量文件: {vector_path.resolve()}")
    print(f"元数据文件: {meta_path.resolve()}")
    print()
    print("=" * 70)
    print("  汇总")
    print("=" * 70)
    print(f"  chunks 总数:             {total_chunks}")
    print(f"  embedding success:       {success}")
    print(f"  embedding failed:        {failed}")
    print(f"  success 且有 vector_id:  {mapped}")
    print(f"  success 缺 vector_id:    {missing_vector_id}")
    print(f"  向量文件行数:            {vector_lines}")
    if meta:
        print(f"  模型:                    {meta.get('embedding_model')}")
        print(f"  维度:                    {meta.get('embedding_dim')}")
        print(f"  本次成功:                {meta.get('success_count')}")
        print(f"  本次失败:                {meta.get('failed_count')}")

    issues: list[str] = []
    if success != mapped:
        issues.append("success 数量与 vector_id 映射数量不一致")
    if vector_lines != mapped:
        issues.append("向量文件行数与 success/vector_id 数量不一致")
    if missing_vector_id:
        issues.append("存在 success 但 vector_id 为空的 chunk")

    print()
    print(f"  状态: {'OK' if not issues else 'ISSUE: ' + '; '.join(issues)}")

    rows = conn.execute(
        """SELECT chunk_id, doc_id, chunk_index, vector_id, embedding_status,
                  title, article_range
           FROM chunks
           WHERE embedding_status IS NOT NULL
           ORDER BY doc_id, chunk_index
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
                f"  vector_id={row['vector_id']} status={row['embedding_status']} "
                f"chunk_index={row['chunk_index']} title={row['title']} "
                f"range={row['article_range']}"
            )

    conn.close()
    sys.exit(0 if not issues else 1)


if __name__ == "__main__":
    main()
