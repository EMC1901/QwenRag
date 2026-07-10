#!/usr/bin/env python3
"""随机抽检阶段7 chunk 切分结果。"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTICLE_REF_RE = re.compile(r"第[一二三四五六七八九十百千万零〇两]+条")


def _fetch_sample(
    conn: sqlite3.Connection,
    count: int,
    doc_limit: int | None,
) -> tuple[list[sqlite3.Row], int]:
    if doc_limit is None:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        rows = conn.execute(
            """SELECT c.chunk_id, c.doc_id, c.chunk_index, c.title,
                      c.section_path, c.article_no, c.article_range, c.token_count,
                      c.chunk_text, c.chunk_text_for_embedding,
                      d.relative_path
               FROM chunks c
               LEFT JOIN documents d ON c.doc_id = d.doc_id
               ORDER BY RANDOM()
               LIMIT ?""",
            (count,),
        ).fetchall()
        return rows, total

    total = conn.execute(
        """
        WITH limited_docs AS (
            SELECT DISTINCT sb.doc_id
            FROM structured_blocks sb
            ORDER BY sb.doc_id
            LIMIT ?
        )
        SELECT COUNT(*)
        FROM chunks c
        INNER JOIN limited_docs ld ON c.doc_id = ld.doc_id
        """,
        (doc_limit,),
    ).fetchone()[0]

    rows = conn.execute(
        """
        WITH limited_docs AS (
            SELECT DISTINCT sb.doc_id
            FROM structured_blocks sb
            ORDER BY sb.doc_id
            LIMIT ?
        )
        SELECT c.chunk_id, c.doc_id, c.chunk_index, c.title,
               c.section_path, c.article_no, c.article_range, c.token_count,
               c.chunk_text, c.chunk_text_for_embedding,
               d.relative_path
        FROM chunks c
        INNER JOIN limited_docs ld ON c.doc_id = ld.doc_id
        LEFT JOIN documents d ON c.doc_id = d.doc_id
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (doc_limit, count),
    ).fetchall()
    return rows, total


def _review_row(row: sqlite3.Row, max_tokens: int) -> list[str]:
    issues: list[str] = []
    chunk_text = row["chunk_text"] or ""
    embedding_text = row["chunk_text_for_embedding"] or ""

    if not chunk_text.strip():
        issues.append("chunk_text 为空")
    if not embedding_text.strip():
        issues.append("chunk_text_for_embedding 为空")
    if row["token_count"] is None or row["token_count"] <= 0:
        issues.append("token_count 异常")
    elif row["token_count"] > max_tokens:
        issues.append(f"token_count 超过 {max_tokens}")

    if row["title"] and f"法规标题：{row['title']}" not in embedding_text:
        issues.append("embedding 文本缺少法规标题")
    if row["section_path"] and "章节路径：" not in embedding_text:
        issues.append("embedding 文本缺少章节路径")
    if row["article_no"] and "条号：" not in embedding_text:
        issues.append("embedding 文本缺少条号")
    if row["article_range"] and "条文范围：" not in embedding_text:
        issues.append("embedding 文本缺少条文范围")
    if not row["article_range"] and ARTICLE_REF_RE.search(chunk_text):
        issues.append("正文含条号引用但 article_range 为空")
    if "正文：" not in embedding_text and (
        row["title"] or row["section_path"] or row["article_no"] or row["article_range"]
    ):
        issues.append("embedding 文本缺少正文标记")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="抽检阶段7 chunk 切分结果")
    parser.add_argument("--count", type=int, default=10, help="抽样数量 (默认 10)")
    parser.add_argument(
        "--doc-limit",
        type=int,
        default=None,
        help="只从阶段7前 N 篇范围内抽样，例如 500",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="用于抽检的最大 token 阈值 (默认 1200)",
    )
    parser.add_argument("--db", default="rag_data/metadata.db", help="数据库路径")
    args = parser.parse_args()

    if args.count <= 0:
        print("[FAIL] --count 必须大于 0")
        sys.exit(1)
    if args.doc_limit is not None and args.doc_limit <= 0:
        print("[FAIL] --doc-limit 必须大于 0")
        sys.exit(1)
    if args.max_tokens <= 0:
        print("[FAIL] --max-tokens 必须大于 0")
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[FAIL] 数据库不存在: {db_path}")
        print("请先运行: python scripts/build_kb.py --stage chunk --limit N")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows, total = _fetch_sample(conn, args.count, args.doc_limit)
    if total == 0:
        print("没有 chunks 数据。")
        print("请先运行: python scripts/build_kb.py --stage chunk --limit N")
        conn.close()
        sys.exit(1)

    print(f"数据库: {db_path.resolve()}")
    if args.doc_limit is None:
        print(f"已有 chunks: {total} | 抽样: {len(rows)}")
    else:
        print(f"抽样范围: 阶段7前 {args.doc_limit} 篇内的 {total} 个 chunks")
        print(f"随机抽样: {len(rows)}")

    results: list[list[str]] = []
    for i, row in enumerate(rows, 1):
        issues = _review_row(row, args.max_tokens)
        results.append(issues)

        print(f"\n{'='*70}")
        print(f"  [{i}/{len(rows)}] {row['title'] or '(无标题)'}")
        print(f"{'='*70}")
        print(f"  文件: {row['relative_path'] or '?'}")
        print(f"  doc_id: {row['doc_id']}")
        print(f"  chunk_index: {row['chunk_index']}")
        print(f"  token_count: {row['token_count']}")
        print(f"  section_path: {row['section_path'] or 'null'}")
        print(f"  article_no: {row['article_no'] or 'null'}")
        print(f"  article_range: {row['article_range'] or 'null'}")
        print(f"  状态: {'OK' if not issues else 'ISSUE: ' + '; '.join(issues)}")
        print()
        text = row["chunk_text"] or ""
        if len(text) > 1200:
            text = text[:1200] + "\n... <truncated>"
        print(text)

    issue_docs = sum(1 for r in results if r)
    issue_count = sum(len(r) for r in results)
    print(f"\n{'='*70}")
    print("  汇总")
    print(f"{'='*70}")
    print(f"  审查 chunks: {len(rows)} | 有问题: {issue_docs} | 问题项: {issue_count}")

    conn.close()


if __name__ == "__main__":
    main()
