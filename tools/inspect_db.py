#!/usr/bin/env python3
"""检查 SQLite 数据库的表结构、行数和完整性。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_preprocess.database import connect_db


def inspect(db_path: Path) -> None:
    conn = connect_db(db_path)

    # 列出所有用户可见表（排除 FTS 内部表）
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '%fts_%' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [row[0] for row in cursor]

    print(f"数据库: {db_path.resolve()}")
    print(f"用户表数量: {len(tables)}")
    print()

    total_cols = 0
    for table in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
        except Exception:
            count = "N/A"

        cols = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
        total_cols += len(cols)
        col_str = ", ".join(c[1] for c in cols[:6])
        if len(cols) > 6:
            col_str += f" ... (+{len(cols) - 6})"

        print(f"  [{table}]  {count} 行  |  {col_str}")

    print()
    print("─" * 50)
    print("完整性检查")
    print("─" * 50)

    # 检查: law_records 中 matched_source_file_id 的 FK 有效性
    orphan_records = conn.execute(
        """SELECT COUNT(*) FROM law_records lr
           WHERE lr.matched_source_file_id IS NOT NULL
           AND lr.matched_source_file_id NOT IN
               (SELECT source_file_id FROM source_files)"""
    ).fetchone()[0]
    print(f"  law_records 孤立外键 (matched_source_file_id): {orphan_records}")

    # 检查: documents 中 source_file_id 的 FK 有效性
    orphan_docs = conn.execute(
        """SELECT COUNT(*) FROM documents d
           WHERE d.source_file_id NOT IN
               (SELECT source_file_id FROM source_files)"""
    ).fetchone()[0]
    print(f"  documents 孤立外键 (source_file_id): {orphan_docs}")

    # 匹配概览
    matched = conn.execute(
        "SELECT COUNT(*) FROM law_records WHERE matched_source_file_id IS NOT NULL"
    ).fetchone()[0]
    total_records = conn.execute("SELECT COUNT(*) FROM law_records").fetchone()[0]
    total_files = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
    if total_records > 0:
        print(f"  records 匹配率: {matched}/{total_records} = {matched/total_records*100:.1f}%")
    if total_files > 0:
        matched_files = conn.execute(
            "SELECT COUNT(DISTINCT matched_source_file_id) FROM law_records "
            "WHERE matched_source_file_id IS NOT NULL"
        ).fetchone()[0]
        print(f"  files 被引用率: {matched_files}/{total_files} = {matched_files/total_files*100:.1f}%")

    # 文档状态
    doc_stats = conn.execute(
        "SELECT parse_status, COUNT(*) FROM documents GROUP BY parse_status"
    ).fetchall()
    if doc_stats:
        parts = [f'{r[0] or "null"}={r[1]}' for r in doc_stats]
        print(f"  documents 解析状态: {', '.join(parts)}")

    # chunk 统计
    chunk_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if chunk_total > 0:
        chunk_embed_ok = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'success'"
        ).fetchone()[0]
        print(f"  chunks embedding 成功率: {chunk_embed_ok}/{chunk_total}")

    # 构建错误统计
    err_count = conn.execute("SELECT COUNT(*) FROM build_errors").fetchone()[0]
    if err_count > 0:
        err_by_stage = conn.execute(
            "SELECT stage, COUNT(*) FROM build_errors GROUP BY stage"
        ).fetchall()
        print(f"  build_errors: {err_count} 条 ({', '.join(f'{r[0]}={r[1]}' for r in err_by_stage)})")

    conn.close()
    print("─" * 50)


if __name__ == "__main__":
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rag_data/metadata.db")
    if not db_path.exists():
        print(f"错误: 数据库文件不存在: {db_path}")
        sys.exit(1)
    inspect(db_path)
