#!/usr/bin/env python3
"""从 structured_blocks 中随机抽样，审查阶段6清洗与结构识别质量。

检查项：
  1. clean_text 正常无乱码
  2. detected_level 能识别 chapter / section / article
  3. section_path 随章节变化
  4. article_no 能识别 "第十条"
  5. is_noise 未将正文误判为噪声

用法：
  python tools/review_stage6.py             # 默认抽查 3 篇
  python tools/review_stage6.py --count 5   # 抽查 5 篇
  python tools/review_stage6.py --count 10 --doc-limit 500
                                            # 只在阶段6前500篇范围内抽查
"""

import argparse
import random
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 乱码检测 ──────────────────────────────────────────────────

GARBLED_CHARS = {
    "�",
    "Â", "Ã", "Ä",
    "â", "ã", "ä",
}


def detect_garbled(text: str) -> tuple[bool, int, float]:
    if not text:
        return False, 0, 0.0
    count = sum(1 for ch in text if ch in GARBLED_CHARS)
    ratio = count / len(text)
    return ratio > 0.03, count, ratio


# ── 单文档审查 ────────────────────────────────────────────────

def review_one_doc(conn: sqlite3.Connection, doc_id: str, index: int, total: int):
    doc = conn.execute(
        "SELECT doc_id, title, relative_path, parse_status FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    title = doc["title"] or "(无标题)" if doc else "(无标题)"
    path = doc["relative_path"] if doc else "?"

    sbs = conn.execute(
        """SELECT structured_block_id, block_index, block_type,
                  raw_text, clean_text, detected_level,
                  section_path, article_no, is_noise
           FROM structured_blocks
           WHERE doc_id = ? ORDER BY block_index""",
        (doc_id,),
    ).fetchall()

    if not sbs:
        return None

    # ── 质量检查 ──
    issues: list[str] = []

    # 1. 乱码
    garbled_blocks = 0
    for sb in sbs:
        _, gb_count, _ = detect_garbled(sb["clean_text"] or "")
        if gb_count > 0:
            garbled_blocks += 1
    garbled_ok = garbled_blocks == 0

    # 2. detected_level
    levels_found = set()
    for sb in sbs:
        if sb["detected_level"]:
            levels_found.add(sb["detected_level"])
    has_chapter = "章" in levels_found
    has_section = "节" in levels_found
    has_article = "条" in levels_found

    # 3. section_path 变化
    distinct_paths = set()
    for sb in sbs:
        if sb["section_path"]:
            distinct_paths.add(sb["section_path"])
    path_changes = len(distinct_paths) >= 2

    # 4. article_no
    article_nos = list(dict.fromkeys(
        sb["article_no"] for sb in sbs if sb["article_no"]
    ))
    has_article_no = len(article_nos) > 0

    # 5. is_noise 误判
    structure_keywords = ["第", "章", "节", "条", "款", "项", "编"]
    noise_blocks = [sb for sb in sbs if sb["is_noise"]]
    false_noise = [
        sb for sb in noise_blocks
        if any(kw in (sb["clean_text"] or sb["raw_text"] or "") for kw in structure_keywords)
    ]
    noise_ok = len(false_noise) == 0

    # 累计问题
    if not garbled_ok:
        issues.append(f"{garbled_blocks} 个 block 含乱码字符")
    if not has_chapter and not has_section and not has_article:
        issues.append("未识别到任何法规层级")
    if not has_article_no and has_article:
        issues.append("检测到条层级但未提取到条号")
    if not noise_ok:
        issues.append(f"{len(false_noise)} 个正文被误判为噪声")

    # ── 输出头部 ──
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  [{index}/{total}] {title}")
    print(f"{sep}")
    print(f"  文件: {path}")
    print(f"  blocks: {len(sbs)} | 层级: {', '.join(sorted(levels_found)) if levels_found else '无'}"
          f" | 条号: {len(article_nos)} 个"
          f" | 路径变化: {'是' if path_changes else '否'}")
    print()

    # 检查清单（紧凑）
    checks = [
        ("clean_text 无乱码", garbled_ok),
        ("detected_level 识别", has_chapter or has_section or has_article),
        ("article_no 识别", has_article_no),
        ("is_noise 无误判", noise_ok),
    ]
    statuses = []
    for name, ok in checks:
        statuses.append(f"[{'v' if ok else 'x'}] {name}")
    print(f"  {' | '.join(statuses)}")
    if issues:
        print(f"  [ISSUE] {'; '.join(issues)}")

    # ── 正文展示：逐 block 字段展开 ──
    print(f"\n  {'─'*66}")

    for sb in sbs:
        text = sb["clean_text"] or sb["raw_text"] or ""
        level = sb["detected_level"] or "null"
        path = sb["section_path"] or "null"
        art = sb["article_no"] or "null"
        noise_mark = "  <NOISE>" if sb["is_noise"] else ""

        print(f"  [{sb['block_index']}]"
              f"{noise_mark}")
        print(f"    clean_text:     {text}")
        print(f"    detected_level: {level}")
        print(f"    section_path:   {path}")
        print(f"    article_no:     {art}")
        print()

    # ── 条号列表 ──
    if article_nos:
        print(f"  ── 条号 ({len(article_nos)} 个) ──")
        print(f"  {' → '.join(article_nos[:20])}")
        print()

    return issues


# ── 汇总 ──────────────────────────────────────────────────────

def print_summary(results: list, doc_count: int):
    print(f"\n{'='*70}")
    print(f"  汇总")
    print(f"{'='*70}")
    pass_count = sum(1 for r in results if not r)
    total_issues = sum(len(r) for r in results if r)
    print(f"  审查: {doc_count} 篇 | 通过: {pass_count} | 问题: {doc_count - pass_count} ({total_issues} 项)")

    if total_issues > 0:
        counts: dict[str, int] = {}
        for r in results:
            if r:
                for issue in r:
                    if "乱码" in issue: k = "乱码"
                    elif "层级" in issue: k = "层级未识别"
                    elif "条号" in issue: k = "条号缺失"
                    elif "噪声" in issue: k = "噪声误判"
                    else: k = issue[:20]
                    counts[k] = counts.get(k, 0) + 1
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")


# ── 主入口 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="审查阶段6结构化结果")
    parser.add_argument("--count", type=int, default=3, help="抽样数量 (默认 3)")
    parser.add_argument(
        "--doc-limit",
        type=int,
        default=None,
        help="只从阶段6按 doc_id 排序取前 N 篇的范围内抽样，例如 500",
    )
    parser.add_argument("--db", default="rag_data/metadata.db", help="数据库路径")
    args = parser.parse_args()

    if args.count <= 0:
        print("[FAIL] --count 必须大于 0")
        sys.exit(1)

    if args.doc_limit is not None and args.doc_limit <= 0:
        print("[FAIL] --doc-limit 必须大于 0")
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[FAIL] 数据库不存在: {db_path}")
        print("  请先运行: python scripts/build_kb.py --stage structure --limit N")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if args.doc_limit is None:
        doc_rows = conn.execute(
            "SELECT DISTINCT doc_id FROM structured_blocks ORDER BY doc_id"
        ).fetchall()
    else:
        # 与 scripts/build_kb.py --stage structure --limit N 的默认/force 范围保持一致：
        # 先按 parsed_blocks.doc_id 排序取前 N 篇，再从其中已结构化的文档里抽样。
        doc_rows = conn.execute(
            """
            WITH limited_docs AS (
                SELECT DISTINCT pb.doc_id
                FROM parsed_blocks pb
                ORDER BY pb.doc_id
                LIMIT ?
            )
            SELECT ld.doc_id
            FROM limited_docs ld
            WHERE EXISTS (
                SELECT 1
                FROM structured_blocks sb
                WHERE sb.doc_id = ld.doc_id
            )
            ORDER BY ld.doc_id
            """,
            (args.doc_limit,),
        ).fetchall()

    total = len(doc_rows)
    if total == 0:
        print("没有 structured_blocks 数据。")
        print("请先运行: python scripts/build_kb.py --stage structure --limit N")
        conn.close()
        sys.exit(1)

    sample_size = min(args.count, total)
    sampled = random.sample(doc_rows, sample_size)
    doc_ids = [r["doc_id"] for r in sampled]

    print(f"数据库: {db_path.resolve()}")
    if args.doc_limit is None:
        print(f"已有文档: {total} | 抽样: {sample_size} 篇")
    else:
        print(f"抽样范围: 阶段6前 {args.doc_limit} 篇中已结构化的 {total} 篇")
        print(f"随机抽样: {sample_size} 篇")

    results = []
    for i, doc_id in enumerate(doc_ids, 1):
        issues = review_one_doc(conn, doc_id, i, sample_size)
        results.append(issues)

    print_summary(results, sample_size)
    conn.close()
    print()


if __name__ == "__main__":
    main()
