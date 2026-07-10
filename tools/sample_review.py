#!/usr/bin/env python3
"""从已解析文档中随机抽样，输出全文供人工审查。

用法：
  python scripts/sample_review.py [--count 15] [--with-tables]
"""

import argparse
import random
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_preprocess.text_cleaner import is_noise_line


# ── 乱码检测 ──────────────────────────────────────────────────

# 常见乱码字符（CP1252 误解释为 Latin-1 等产生的符号）
GARBLED_CHARS = {
    "�",                # Unicode replacement character
    "Â", "Ã", "Ä",      # 常见 UTF-8 被错误解码的产物
    "â", "ã", "ä",
}

# 可疑符号比例阈值
GARBLED_RATIO_THRESHOLD = 0.05


def detect_garbled(text: str) -> tuple[bool, int, float]:
    """检测文本中的乱码。

    Returns:
        (has_garbled, garbled_count, garbled_ratio)
    """
    total = len(text)
    if total == 0:
        return False, 0, 0.0

    garbled_count = sum(1 for ch in text if ch in GARBLED_CHARS or ord(ch) == 0xFFFD)
    ratio = garbled_count / total
    return ratio > GARBLED_RATIO_THRESHOLD, garbled_count, ratio


# ── 空行检测 ──────────────────────────────────────────────────

def check_empty_lines(text: str) -> tuple[int, int]:
    """检测空行和噪声行比例。

    Returns:
        (total_lines, noise_or_empty_lines)
    """
    lines = text.split("\n")
    total = len(lines)
    noisy = sum(1 for line in lines if is_noise_line(line))
    return total, noisy


# ── 主逻辑 ────────────────────────────────────────────────────

def review_documents(conn: sqlite3.Connection, doc_ids: list[str]) -> None:
    """抽样审查多个文档。"""
    for idx, doc_id in enumerate(doc_ids, 1):
        # 取 document 记录
        doc = conn.execute(
            "SELECT doc_id, title, relative_path, extension, file_size, parse_status "
            "FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()

        if not doc:
            print(f"\n{'='*70}\n  [{idx}] doc_id={doc_id[:20]}... --- 未找到!\n{'='*70}")
            continue

        title = doc["title"] or "(无标题)"
        path = doc["relative_path"]
        parse_status = doc["parse_status"]

        # 取所有块
        blocks = conn.execute(
            "SELECT block_index, block_type, text FROM parsed_blocks "
            "WHERE doc_id = ? ORDER BY block_index",
            (doc_id,),
        ).fetchall()

        full_text = "\n".join(b["text"] for b in blocks)
        para_count = sum(1 for b in blocks if b["block_type"] == "paragraph")
        table_count = sum(1 for b in blocks if b["block_type"] == "table_row")

        # ── 检查项 ──
        title_ok = bool(doc["title"])
        content_ok = len(full_text) > 100
        has_garbled, gb_count, gb_ratio = detect_garbled(full_text)
        total_lines, noisy_lines = check_empty_lines(full_text)
        empty_line_ratio = noisy_lines / total_lines if total_lines else 0
        table_blocks_present = table_count > 0

        # 检查条文顺序：找 "第X条" 出现的顺序
        import re
        article_matches = re.findall(r"第[一二三四五六七八九十百千零〇]+条", full_text)
        articles_unique = list(dict.fromkeys(article_matches))  # 去重保序

        # ── 输出 ──
        print(f"\n{'='*70}")
        print(f"  [{idx}/{len(doc_ids)}] {title}")
        print(f"{'='*70}")
        print(f"  文件: {path}")
        print(f"  大小: {doc['file_size']:,} bytes | 状态: {parse_status}")
        print(f"  块数: {len(blocks)} (段落={para_count}, 表格行={table_count})")
        print(f"  总字符数: {len(full_text):,}")
        print()

        # 检查清单
        checks = [
            ("标题存在", title_ok, f"[v] {title[:60]}" if title_ok else "[x] 缺失"),
            ("正文完整", content_ok, f"[v] {len(full_text):,} 字符" if content_ok else "[x] 过短"),
            ("无明显乱码", not has_garbled,
             f"[v] 0 可疑字符" if not has_garbled else f"[x] {gb_count} 可疑字符 ({gb_ratio:.1%})"),
            ("空行控制", empty_line_ratio < 0.3,
             f"[v] 噪声行 {noisy_lines}/{total_lines} ({empty_line_ratio:.1%})"
             if empty_line_ratio < 0.3
             else f"[!] 噪声行 {noisy_lines}/{total_lines} ({empty_line_ratio:.1%})"),
            ("条文识别", len(articles_unique) > 0,
             f"[v] 识别到 {len(articles_unique)} 个条号"
             if articles_unique else "[!] 未识别到条号"),
            ("表格保留", True,
             f"[v] {table_count} 行表格文本" if table_blocks_present else "--- 本文档无表格"),
        ]
        all_pass = True
        for check_name, ok, detail in checks:
            if ok:
                status = "[OK]"
            elif "[!]" in detail:
                status = "[WARN]"
            else:
                status = "[FAIL]"
                all_pass = False
            print(f"  {status} {check_name}: {detail}")
        if all_pass:
            print(f"  [PASS] 综合判定: 通过")
        else:
            print(f"  [ISSUE] 综合判定: 有问题需处理")

        # ── 正文预览 ──
        # 显示前 10 个段落 + 条文编号分布
        print(f"\n  ── 正文预览（前 15 行）──")
        para_blocks = [b for b in blocks if b["block_type"] == "paragraph"]
        for b in para_blocks[:15]:
            text = b["text"][:120]
            print(f"  #{b['block_index']:>4d} | {text}")

        # 条文顺序抽样
        if articles_unique:
            print(f"\n  ── 条文顺序（前 20 条）──")
            shown = articles_unique[:20]
            if len(articles_unique) > 20:
                shown_str = " → ".join(shown) + f" ... (+{len(articles_unique) - 20})"
            else:
                shown_str = " → ".join(shown)
            print(f"  {shown_str}")

        # 表格抽样
        table_blocks = [b for b in blocks if b["block_type"] == "table_row"]
        if table_blocks:
            print(f"\n  ── 表格内容（前 5 行）──")
            for b in table_blocks[:5]:
                print(f"  #{b['block_index']:>4d} | {b['text'][:150]}")


def main():
    parser = argparse.ArgumentParser(description="抽样审查解析结果")
    parser.add_argument("--count", type=int, default=15, help="抽样数量 (默认 15)")
    parser.add_argument("--with-tables", action="store_true", help="优先抽含表格的文档")
    parser.add_argument("--db", default="rag_data/metadata.db", help="数据库路径")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"错误: 数据库不存在 {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 获取成功解析的文档 ID 列表
    if args.with_tables:
        # 优先选含表格的
        doc_rows = conn.execute(
            """SELECT DISTINCT d.doc_id
               FROM documents d
               INNER JOIN parsed_blocks pb ON d.doc_id = pb.doc_id
               WHERE d.parse_status = 'success' AND pb.block_type = 'table_row'
               LIMIT ?"""
        , (args.count * 2,)).fetchall()
    else:
        doc_rows = conn.execute(
            "SELECT doc_id FROM documents WHERE parse_status = 'success'"
        ).fetchall()

    total = len(doc_rows)
    if total == 0:
        print("没有已成功解析的文档。")
        sys.exit(1)

    sample_size = min(args.count, total)
    sampled = random.sample(doc_rows, sample_size)
    doc_ids = [r["doc_id"] for r in sampled]

    print(f"数据库: {db_path.resolve()}")
    print(f"已解析文档总数: {total}")
    print(f"随机抽样: {sample_size} 篇")
    if args.with_tables:
        print("(已过滤: 仅含表格的文档)")

    review_documents(conn, doc_ids)

    conn.close()
    print(f"\n{'='*70}")
    print("审查完成。")


if __name__ == "__main__":
    main()
