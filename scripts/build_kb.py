#!/usr/bin/env python3
"""主入口：一键构建法规知识库。

用法：
  python scripts/build_kb.py --stage scan
  python scripts/build_kb.py --stage records
  python scripts/build_kb.py --stage init_db
  python scripts/build_kb.py --stage parse_docx --limit 100
  python scripts/build_kb.py --stage chunk --limit 100
  python scripts/build_kb.py --stage embed --limit 100
  python scripts/build_kb.py --stage faiss
  python scripts/build_kb.py --stage fts
  python scripts/build_kb.py --stage report
  python scripts/build_kb.py --stage all
"""

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_preprocess.config import Config
from rag_preprocess.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="法规知识库离线构建工具")
    parser.add_argument("--rawdata-dir", default="Rawdata", help="Rawdata 目录路径")
    parser.add_argument("--output-dir", default="rag_data", help="输出目录路径")
    parser.add_argument(
        "--stage",
        choices=[
            "scan", "records", "init_db", "parse_docx", "parse_doc",
            "structure", "chunk", "embed", "faiss", "fts", "report", "all",
        ],
        default="all",
        help="要执行的构建阶段",
    )
    parser.add_argument("--limit", type=int, default=None, help="限制处理文件数量（调试用）")
    parser.add_argument("--resume", action="store_true", help="断点续跑（跳过已完成）")
    parser.add_argument("--force", action="store_true", help="强制全部重建（覆盖已有结果）")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# 阶段 1: 扫描 Rawdata
# ═══════════════════════════════════════════════════════════════

def run_stage_scan(config: Config, logger) -> None:
    """阶段 1: 扫描 Rawdata，识别 Word 文件，写入 source_files 表。"""
    import csv
    from rag_preprocess.scanner import scan_rawdata, format_summary
    from rag_preprocess.database import connect_db, init_db, insert_source_files
    from rag_preprocess.paths import ensure_output_dirs

    logger.info("=" * 50)
    logger.info("阶段 1: 扫描 Rawdata")
    logger.info("=" * 50)

    ensure_output_dirs(config.output_dir)
    logger.info(f"扫描目录: {config.rawdata_dir.resolve()}")
    files, summary = scan_rawdata(config.rawdata_dir)
    logger.info("\n" + format_summary(summary))

    logger.info("初始化数据库...")
    conn = connect_db(config.db_path)
    init_db(conn)
    logger.info(f"写入 {len(files)} 条 source_files 记录...")
    insert_source_files(conn, files)

    count = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
    logger.info(f"source_files 表实际行数: {count}")

    csv_path = config.output_dir / "exports" / "source_files.sample.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample = files[:100]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file_id", "volume", "relative_path", "file_name",
                "extension", "file_size", "file_hash_sha256", "mtime",
                "path_length", "is_word_file",
            ],
        )
        writer.writeheader()
        for sf in sample:
            writer.writerow({
                "source_file_id": sf.source_file_id,
                "volume": sf.volume,
                "relative_path": sf.relative_path,
                "file_name": sf.file_name,
                "extension": sf.extension,
                "file_size": sf.file_size,
                "file_hash_sha256": sf.file_hash_sha256[:16] + "...",
                "mtime": sf.mtime,
                "path_length": sf.path_length,
                "is_word_file": 1 if sf.is_word_file else 0,
            })
    logger.info(f"样本 CSV 已导出: {csv_path} ({len(sample)} 条)")

    conn.close()
    logger.info("阶段 1 完成。")


# ═══════════════════════════════════════════════════════════════
# 阶段 2: 读取 records.json
# ═══════════════════════════════════════════════════════════════

def run_stage_records(config: Config, logger) -> None:
    """阶段 2: 读取 records.json，导入 law_records，匹配 source_files。"""
    from rag_preprocess.records_loader import load_records, match_records_to_files
    from rag_preprocess.database import connect_db, init_db, insert_law_records
    from rag_preprocess.paths import ensure_output_dirs

    logger.info("=" * 50)
    logger.info("阶段 2: 读取 records.json")
    logger.info("=" * 50)

    ensure_output_dirs(config.output_dir)
    conn = connect_db(config.db_path)
    init_db(conn)

    cursor = conn.execute("SELECT relative_path FROM source_files")
    source_paths: set[str] = {row[0] for row in cursor}
    logger.info(f"已加载 {len(source_paths)} 条 source_files 路径")

    vol1_path = config.rawdata_dir / "law-flk-vol1-main" / "records.json"
    vol2_path = config.rawdata_dir / "law-flk-vol2-main" / "records.json"

    all_records: list = []
    for vol, path in [("vol1", vol1_path), ("vol2", vol2_path)]:
        if not path.exists():
            logger.warning(f"records.json 不存在: {path}")
            continue
        logger.info(f"加载 {path} ...")
        records = load_records(path, vol)
        logger.info(f"  {vol}: {len(records)} 条记录")
        all_records.extend(records)

    match_result = match_records_to_files(all_records, source_paths)

    for r in all_records:
        if r.expected_relative_path and r.expected_relative_path in source_paths:
            cursor = conn.execute(
                "SELECT source_file_id FROM source_files WHERE relative_path = ?",
                (r.expected_relative_path,),
            )
            row = cursor.fetchone()
            if row:
                r.matched_source_file_id = row[0]

    logger.info(f"写入 {len(all_records)} 条 law_records ...")
    insert_law_records(conn, all_records)

    count = conn.execute("SELECT COUNT(*) FROM law_records").fetchone()[0]
    logger.info(f"law_records 表实际行数: {count}")

    vol1_count = sum(1 for r in all_records if r.volume == "vol1")
    vol2_count = sum(1 for r in all_records if r.volume == "vol2")
    logger.info("")
    logger.info("=" * 50)
    logger.info("records.json 导入统计")
    logger.info("=" * 50)
    logger.info(f"  vol1_records:             {vol1_count:>8}")
    logger.info(f"  vol2_records:             {vol2_count:>8}")
    logger.info(f"  total_records:            {len(all_records):>8}")
    logger.info(f"  matched (有对应文件):     {match_result.matched:>8}")
    logger.info(f"  records_without_file:     {match_result.records_without_file:>8}")
    logger.info(f"  files_without_record:     {match_result.files_without_record:>8}")
    logger.info("")
    logger.info("  缺字段统计:")
    logger.info(f"    missing_title:          {match_result.missing_title:>8}")
    logger.info(f"    missing_gbrq:           {match_result.missing_gbrq:>8}")
    logger.info(f"    missing_sxrq:           {match_result.missing_sxrq:>8}")
    logger.info(f"    missing_sxx:            {match_result.missing_sxx:>8}")
    logger.info(f"    missing_my_file:        {match_result.missing_my_file:>8}")
    logger.info(f"    missing_bbbs:           {match_result.missing_bbbs:>8}")
    logger.info("=" * 50)

    conn.close()
    logger.info("阶段 2 完成。")


# ═══════════════════════════════════════════════════════════════
# 阶段 3: 初始化数据库
# ═══════════════════════════════════════════════════════════════

def run_stage_init_db(config: Config, logger) -> None:
    """阶段 3: 初始化数据库，创建所有表和索引。"""
    from rag_preprocess.database import connect_db, init_db, get_table_counts
    from rag_preprocess.paths import ensure_dir

    logger.info("=" * 50)
    logger.info("阶段 3: 初始化 SQLite 数据库")
    logger.info("=" * 50)

    ensure_dir(config.output_dir)
    db_path = config.db_path

    if config.force and db_path.exists():
        logger.warning(f"强制重建：删除旧数据库 {db_path}")
        db_path.unlink()

    existed = db_path.exists()
    conn = connect_db(db_path)
    init_db(conn)

    counts = get_table_counts(conn)

    logger.info(f"数据库路径: {db_path.resolve()}")
    logger.info(f"状态: {'已存在，表已补建' if existed else '新建'}")
    logger.info("")
    logger.info("表结构验证:")
    for table, count in counts.items():
        status = "✓" if count >= 0 else "✗"
        logger.info(f"  {status} {table}: {count} 行")

    conn.close()
    logger.info("阶段 3 完成。")


# ═══════════════════════════════════════════════════════════════
# 阶段 4: 解析 docx
# ═══════════════════════════════════════════════════════════════

def run_stage_parse_docx(config: Config, logger) -> None:
    """阶段 4: 解析 .docx 文件，提取段落和表格，写入 documents + parsed_blocks。"""
    import hashlib
    from tqdm import tqdm
    from rag_preprocess.database import (
        connect_db, init_db,
        get_source_files_for_parsing, count_documents_by_status,
        insert_document, insert_parsed_blocks, insert_build_error,
    )
    from rag_preprocess.docx_parser import parse_docx
    from rag_preprocess.paths import ensure_output_dirs

    logger.info("=" * 50)
    logger.info("阶段 4: 解析 .docx 文件")
    logger.info("=" * 50)

    ensure_output_dirs(config.output_dir)
    conn = connect_db(config.db_path)
    init_db(conn)

    # 1. 获取待解析的 .docx 文件列表
    rows = get_source_files_for_parsing(
        conn, extension=".docx", limit=config.limit,
        resume=config.resume, force=config.force,
    )

    # 统计信息
    total_eligible = conn.execute(
        "SELECT COUNT(*) FROM source_files WHERE is_word_file=1 AND extension='.docx'"
    ).fetchone()[0]
    skipped = total_eligible - len(rows)

    if config.resume and skipped > 0:
        logger.info(f"--resume：跳过已处理 {skipped} 个，待处理 {len(rows)} 个")
    elif config.force and skipped > 0:
        logger.info(f"--force：跳过已成功 {skipped} 个，待处理 {len(rows)} 个")
    else:
        logger.info(f"待解析 .docx 文件: {len(rows)} (共 {total_eligible} 个)")

    if not rows:
        logger.info("没有需要解析的文件。")
        conn.close()
        return

    success_count = 0
    fail_count = 0
    empty_count = 0
    # 详细统计累加器
    total_paragraphs = 0
    total_text_chars = 0
    total_table_blocks = 0
    docs_with_tables = 0

    for row in tqdm(rows, desc="解析 docx", unit="file"):
        source_file_id = row["source_file_id"]
        relative_path = row["relative_path"]
        file_path = config.rawdata_dir / relative_path

        # 检查文件是否存在
        if not file_path.exists():
            insert_build_error(
                conn,
                error_id=f"missing:{source_file_id}",
                stage="parse_docx",
                error_type="file_missing",
                error_message=str(file_path),
                source_file_id=source_file_id,
            )
            fail_count += 1
            continue

        try:
            # 2. 解析 docx
            parsed = parse_docx(file_path, doc_id=source_file_id)

            # 3. 计算内容 hash
            all_text = "\n".join(b.text for b in parsed.blocks)
            content_hash = hashlib.sha256(all_text.encode("utf-8")).hexdigest()

            # 4. 构建 record 相关字段
            record_data = {}
            if row["record_id"]:
                record_data = {
                    "record_id": row["record_id"],
                    "title": row["record_title"],
                    "flxz": row["flxz"],
                    "gbrq": row["gbrq"],
                    "sxrq": row["sxrq"],
                    "sxx": row["sxx"],
                    "zdjg_name": row["zdjg_name"],
                }

            # 5. 写入 document 记录
            parse_status = "failed" if parsed.parse_error else ("success" if parsed.blocks else "empty")
            insert_document(
                conn,
                doc_id=source_file_id,
                source_file_id=source_file_id,
                relative_path=relative_path,
                extension=row["extension"],
                file_size=row["file_size"],
                file_hash_sha256=row["file_hash_sha256"],
                content_hash_sha256=content_hash,
                volume=row["volume"],
                parse_status=parse_status,
                parse_error=parsed.parse_error,
                **record_data,
            )

            # 6. 写入 parsed_blocks
            if parsed.blocks:
                insert_parsed_blocks(conn, parsed.blocks, doc_id=source_file_id)

            # 7. 统计
            if parsed.parse_error:
                fail_count += 1
                insert_build_error(
                    conn,
                    error_id=f"parse:{source_file_id}",
                    stage="parse_docx",
                    error_type="parse_error",
                    error_message=parsed.parse_error,
                    source_file_id=source_file_id,
                    doc_id=source_file_id,
                )
            elif not parsed.blocks:
                empty_count += 1
            else:
                success_count += 1
                # 累计详细统计
                para_count = sum(1 for b in parsed.blocks if b.block_type == "paragraph")
                table_count = sum(1 for b in parsed.blocks if b.block_type == "table_row")
                text_len = sum(len(b.text) for b in parsed.blocks)
                total_paragraphs += para_count
                total_text_chars += text_len
                total_table_blocks += table_count
                if table_count > 0:
                    docs_with_tables += 1

        except Exception as e:
            fail_count += 1
            insert_build_error(
                conn,
                error_id=f"except:{source_file_id}",
                stage="parse_docx",
                error_type="exception",
                error_message=str(e),
                source_file_id=source_file_id,
            )
            try:
                insert_document(
                    conn,
                    doc_id=source_file_id,
                    source_file_id=source_file_id,
                    relative_path=relative_path,
                    extension=row["extension"],
                    parse_status="failed",
                    parse_error=str(e),
                )
            except Exception:
                pass

    # 8. 输出统计
    doc_stats = count_documents_by_status(conn)
    block_count = conn.execute("SELECT COUNT(*) FROM parsed_blocks").fetchone()[0]

    logger.info("")
    logger.info("=" * 50)
    logger.info("解析统计")
    logger.info("=" * 50)
    logger.info(f"  处理文件数:          {len(rows):>8}")
    logger.info(f"  成功:                {success_count:>8}")
    logger.info(f"  正文为空:            {empty_count:>8}")
    logger.info(f"  失败:                {fail_count:>8}")
    logger.info(f"  已解析文档 (DB):     {sum(doc_stats.values()):>8}")
    logger.info(f"  已解析块 (DB):       {block_count:>8}")
    if doc_stats:
        logger.info(f"  按状态: {dict(doc_stats)}")

    # 详细统计
    processed = success_count + empty_count
    if processed > 0:
        avg_para = total_paragraphs / processed
        avg_text_len = total_text_chars / processed
    else:
        avg_para = 0
        avg_text_len = 0

    logger.info("")
    logger.info("  详细统计 (本次成功+空文本):")
    logger.info(f"  计入文档数:          {processed:>8}")
    logger.info(f"  平均段落数:          {avg_para:>8.1f}")
    logger.info(f"  平均文本长度(字符):  {avg_text_len:>8.0f}")
    logger.info(f"  表格行块总数:        {total_table_blocks:>8}")
    logger.info(f"  含表格的文档数:      {docs_with_tables:>8}")
    logger.info("=" * 50)

    conn.close()
    logger.info("阶段 4 完成。")


# ═══════════════════════════════════════════════════════════════
# 阶段 6: 文本清洗与法规结构识别
# ═══════════════════════════════════════════════════════════════

def run_stage_structure(config: Config, logger) -> None:
    """阶段 6: 对已解析的 parsed_blocks 进行文本清洗和法规结构识别，
    结果写入 structured_blocks 表。"""
    from tqdm import tqdm
    from rag_preprocess.database import (
        connect_db, init_db,
        insert_structured_blocks, get_structured_blocks_by_doc,
    )
    from rag_preprocess.text_cleaner import clean_text, is_noise_line
    from rag_preprocess.law_structure import build_section_path
    from rag_preprocess.paths import ensure_output_dirs

    logger.info("=" * 50)
    logger.info("阶段 6: 文本清洗与法规结构识别")
    logger.info("=" * 50)

    ensure_output_dirs(config.output_dir)
    conn = connect_db(config.db_path)
    init_db(conn)

    # ── 选取待处理的文档 ──
    total_eligible = conn.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM parsed_blocks"
    ).fetchone()[0]

    if config.force:
        # 只跳过已成功的
        doc_rows = conn.execute("""
            SELECT DISTINCT pb.doc_id FROM parsed_blocks pb
            ORDER BY pb.doc_id
        """).fetchall()
    elif config.resume:
        # 跳过已有 structured_blocks 的文档
        doc_rows = conn.execute("""
            SELECT DISTINCT pb.doc_id
            FROM parsed_blocks pb
            WHERE pb.doc_id NOT IN (
                SELECT DISTINCT doc_id FROM structured_blocks
            )
            ORDER BY pb.doc_id
        """).fetchall()
    else:
        # 全量，不跳过
        doc_rows = conn.execute("""
            SELECT DISTINCT pb.doc_id FROM parsed_blocks pb
            ORDER BY pb.doc_id
        """).fetchall()

    all_doc_ids = [r[0] for r in doc_rows]

    if config.limit is not None:
        doc_ids = all_doc_ids[:config.limit]
    else:
        doc_ids = all_doc_ids

    if not doc_ids:
        logger.info("没有需要处理的文档。")
        conn.close()
        return

    skipped = total_eligible - len(all_doc_ids)

    if config.resume and skipped > 0:
        logger.info(f"--resume：跳过已处理 {skipped} 个，待处理 {len(doc_ids)} 个")
    elif config.force and skipped > 0:
        logger.info(f"--force：全量重处理 {len(doc_ids)} 个")
    else:
        logger.info(f"待处理文档: {len(doc_ids)} (共 {total_eligible} 个)")

    # ── 逐文档处理 ──
    total_blocks = 0
    noise_count = 0
    structure_hits: dict[str, int] = {}

    for doc_id in tqdm(doc_ids, desc="结构化", unit="doc"):
        pb_rows = conn.execute(
            "SELECT * FROM parsed_blocks WHERE doc_id = ? ORDER BY block_index",
            (doc_id,),
        ).fetchall()

        if not pb_rows:
            continue

        # 1. 将 DB 行转为对象
        class _Block:
            pass

        blocks = []
        for i, row in enumerate(pb_rows):
            b = _Block()
            b.text = row["text"] or ""
            b.block_id = row["block_id"] or ""
            b.block_index = i
            b.block_type = row["block_type"] or "paragraph"
            b.paragraph_index = row["paragraph_index"]
            b.table_index = row["table_index"]
            b.row_index = row["row_index"]
            b.style_name = row["style_name"]
            blocks.append(b)

        # 2. 文本清洗
        for b in blocks:
            b.raw_text = b.text
            b.text = clean_text(b.text)
            b._is_noise = is_noise_line(b.text)
            if b._is_noise:
                noise_count += 1

        # 3. 法规结构识别
        structured_blocks = build_section_path(blocks)

        # 4. 补齐 raw/clean/noise 默认值。
        # build_section_path 可能会把一个 parsed block 拆成多个 structured block，
        # 因此不能再用 zip(structured_blocks, blocks)，否则拆分后的子块会错位。
        for sb in structured_blocks:
            if not hasattr(sb, "raw_text"):
                sb.raw_text = sb.text
            if not hasattr(sb, "clean_text"):
                sb.clean_text = sb.text
            if not hasattr(sb, "is_noise"):
                sb.is_noise = False

        # 5. 写入 DB
        insert_structured_blocks(conn, structured_blocks, doc_id)

        # 6. 统计
        total_blocks += len(structured_blocks)
        for sb in structured_blocks:
            if sb.law_level:
                key = sb.law_level.value
                structure_hits[key] = structure_hits.get(key, 0) + 1

    # ── 输出报告 ──
    sb_doc_count = conn.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM structured_blocks"
    ).fetchone()[0]
    sb_total = conn.execute("SELECT COUNT(*) FROM structured_blocks").fetchone()[0]

    logger.info("")
    logger.info("=" * 50)
    logger.info("阶段 6 完成")
    logger.info("=" * 50)
    logger.info(f"  本次处理文档数:     {len(doc_ids):>8}")
    logger.info(f"  DB 中文档数:        {sb_doc_count:>8}")
    logger.info(f"  总 structured_blocks:{sb_total:>8}")
    logger.info(f"  本次 block 数:      {total_blocks:>8}")
    logger.info(f"  噪声行数:           {noise_count:>8}")

    if structure_hits:
        logger.info("")
        logger.info("  法规结构识别统计:")
        for level in ["编", "章", "节", "条", "款", "项", "目"]:
            cnt = structure_hits.get(level, 0)
            if cnt > 0:
                logger.info(f"    {level}: {cnt}")

    # 抽样展示
    logger.info("")
    logger.info("  抽样 (前 3 条有结构信息的记录):")
    shown = 0
    for did in doc_ids[:10]:
        for r in get_structured_blocks_by_doc(conn, did):
            if shown >= 3:
                break
            if r["section_path"] or r["article_no"]:
                logger.info(
                    f"    [{r['detected_level'] or 'plain'}] "
                    f"path={r['section_path']} "
                    f"art={r['article_no']} "
                    f"text={r['raw_text'][:60]}"
                )
                shown += 1
        if shown >= 3:
            break

    conn.close()
    logger.info("阶段 6 结束。")


# ═══════════════════════════════════════════════════════════════
# 阶段 7: chunk 切分
# ═══════════════════════════════════════════════════════════════

def run_stage_chunk(config: Config, logger) -> None:
    """阶段 7: 基于 structured_blocks 生成 chunks，并写入 FTS5。"""
    from tqdm import tqdm

    from rag_preprocess.chunker import (
        ChunkConfig,
        StructuredDocument,
        build_chunks,
    )
    from rag_preprocess.database import (
        clear_chunk_fts_for_docs,
        clear_chunks_for_doc,
        connect_db,
        init_db,
        insert_chunks,
        rebuild_chunk_fts,
    )
    from rag_preprocess.law_structure import LawLevel
    from rag_preprocess.paths import ensure_output_dirs

    logger.info("=" * 50)
    logger.info("阶段 7: chunk 切分")
    logger.info("=" * 50)

    ensure_output_dirs(config.output_dir)
    conn = connect_db(config.db_path)
    init_db(conn)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")

    total_eligible = conn.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM structured_blocks"
    ).fetchone()[0]

    if config.force:
        doc_rows = conn.execute("""
            SELECT DISTINCT sb.doc_id
            FROM structured_blocks sb
            ORDER BY sb.doc_id
        """).fetchall()
    elif config.resume:
        doc_rows = conn.execute("""
            SELECT DISTINCT sb.doc_id
            FROM structured_blocks sb
            WHERE sb.doc_id NOT IN (
                SELECT DISTINCT doc_id FROM chunks
            )
            ORDER BY sb.doc_id
        """).fetchall()
    else:
        doc_rows = conn.execute("""
            SELECT DISTINCT sb.doc_id
            FROM structured_blocks sb
            ORDER BY sb.doc_id
        """).fetchall()

    all_doc_ids = [r[0] for r in doc_rows]
    if config.limit is not None:
        doc_ids = all_doc_ids[:config.limit]
    else:
        doc_ids = all_doc_ids

    if not doc_ids:
        logger.info("没有需要生成 chunk 的文档。")
        conn.close()
        return

    skipped = total_eligible - len(all_doc_ids)
    if config.resume and skipped > 0:
        logger.info(f"--resume：跳过已生成 chunk 的 {skipped} 个，待处理 {len(doc_ids)} 个")
    elif config.force:
        logger.info(f"--force：重建 chunk {len(doc_ids)} 个文档")
    else:
        logger.info(f"待生成 chunk 文档: {len(doc_ids)} (共 {total_eligible} 个)")

    chunk_config = ChunkConfig(
        target_chunk_tokens=config.target_chunk_tokens,
        max_chunk_tokens=config.max_chunk_tokens,
        overlap_tokens=config.overlap_tokens,
    )

    total_chunks = 0
    empty_docs = 0
    docs_without_chunks = 0
    max_token_seen = 0
    over_max_chunks = 0
    batch_size = 500
    pending_writes = 0
    rebuild_all_fts = config.limit is None and not config.resume

    if not rebuild_all_fts:
        logger.info(f"预清理 FTS5 旧记录: 本次 {len(doc_ids)} 个文档")
        clear_chunk_fts_for_docs(conn, doc_ids)

    for doc_id in tqdm(doc_ids, desc="生成 chunk", unit="doc"):
        doc_row = conn.execute(
            """SELECT doc_id, title, relative_path, zdjg_name, gbrq, sxrq, flxz, sxx
               FROM documents
               WHERE doc_id = ?""",
            (doc_id,),
        ).fetchone()

        sb_rows = conn.execute(
            """SELECT structured_block_id, block_id, block_index, block_type,
                      clean_text, raw_text, detected_level, section_path,
                      article_no, is_noise
               FROM structured_blocks
               WHERE doc_id = ?
               ORDER BY block_index""",
            (doc_id,),
        ).fetchall()

        class _Block:
            pass

        blocks = []
        for row in sb_rows:
            text = (row["clean_text"] or row["raw_text"] or "")
            text = text.replace("\u200b", "").replace("\ufeff", "").strip()
            if not text or row["is_noise"]:
                continue

            b = _Block()
            b.block_id = row["block_id"] or row["structured_block_id"]
            b.block_index = row["block_index"]
            b.block_type = row["block_type"] or "paragraph"
            b.text = text
            b.section_path = row["section_path"]
            b.article_no = row["article_no"]
            b.law_level = _law_level_from_db(row["detected_level"], LawLevel)
            blocks.append(b)

        if not blocks:
            empty_docs += 1
            clear_chunks_for_doc(
                conn,
                doc_id,
                clear_fts=False,
                commit=False,
            )
            pending_writes += 1
            if pending_writes >= batch_size:
                conn.commit()
                pending_writes = 0
            continue

        title = None
        relative_path = None
        if doc_row:
            title = doc_row["title"]
            relative_path = doc_row["relative_path"]
        if not title and relative_path:
            title = Path(relative_path).stem

        structured_doc = StructuredDocument(
            doc_id=doc_id,
            title=title,
            zdjg_name=doc_row["zdjg_name"] if doc_row else None,
            gbrq=doc_row["gbrq"] if doc_row else None,
            sxrq=doc_row["sxrq"] if doc_row else None,
            flxz=doc_row["flxz"] if doc_row else None,
            sxx=doc_row["sxx"] if doc_row else None,
            blocks=blocks,
        )

        chunks = build_chunks(structured_doc, chunk_config)
        if not chunks:
            docs_without_chunks += 1
            clear_chunks_for_doc(
                conn,
                doc_id,
                clear_fts=False,
                commit=False,
            )
            pending_writes += 1
            if pending_writes >= batch_size:
                conn.commit()
                pending_writes = 0
            continue

        insert_chunks(
            conn,
            chunks,
            clear_fts=False,
            commit=False,
        )
        pending_writes += 1
        if pending_writes >= batch_size:
            conn.commit()
            pending_writes = 0

        total_chunks += len(chunks)
        for chunk in chunks:
            max_token_seen = max(max_token_seen, chunk.token_count)
            if chunk.token_count > config.max_chunk_tokens:
                over_max_chunks += 1

    if pending_writes > 0:
        conn.commit()

    if rebuild_all_fts:
        logger.info("重建 FTS5 索引: 全量")
        db_fts_total = rebuild_chunk_fts(conn)
    else:
        logger.info(f"重建 FTS5 索引: 本次 {len(doc_ids)} 个文档")
        db_fts_total = rebuild_chunk_fts(conn, doc_ids=doc_ids)

    db_chunk_docs = conn.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM chunks"
    ).fetchone()[0]
    db_chunk_total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    sample_path = _export_chunk_sample(conn, config.output_dir)

    logger.info("")
    logger.info("=" * 50)
    logger.info("阶段 7 完成")
    logger.info("=" * 50)
    logger.info(f"  本次处理文档数:       {len(doc_ids):>8}")
    logger.info(f"  本次生成 chunks:      {total_chunks:>8}")
    logger.info(f"  批量提交大小:         {batch_size:>8}")
    logger.info(f"  FTS5 重建模式:        {'全量' if rebuild_all_fts else '本次文档'}")
    logger.info(f"  DB 中 chunk 文档数:   {db_chunk_docs:>8}")
    logger.info(f"  DB 中 chunks:         {db_chunk_total:>8}")
    logger.info(f"  FTS5 rows:            {db_fts_total:>8}")
    logger.info(f"  空文本文档:           {empty_docs:>8}")
    logger.info(f"  未产出 chunk 文档:    {docs_without_chunks:>8}")
    logger.info(f"  最大 token 估算:      {max_token_seen:>8}")
    logger.info(f"  超过 max 的 chunks:   {over_max_chunks:>8}")
    logger.info(f"  样本已导出:           {sample_path}")

    logger.info("")
    logger.info("  抽样 (前 3 个 chunk):")
    rows = conn.execute(
        """SELECT chunk_index, title, section_path, article_no, article_range,
                  token_count, chunk_text
           FROM chunks
           ORDER BY doc_id, chunk_index
           LIMIT 3"""
    ).fetchall()
    for row in rows:
        logger.info(
            f"    [{row['chunk_index']}] "
            f"title={row['title']} path={row['section_path']} "
            f"art={row['article_no']} range={row['article_range']} "
            f"tokens={row['token_count']} "
            f"text={row['chunk_text'][:80]}"
        )

    conn.close()
    logger.info("阶段 7 结束。")


def _law_level_from_db(label: str | None, law_level_cls):
    """把 DB 中的 detected_level 文本转回 LawLevel。"""
    if not label:
        return None
    try:
        return law_level_cls(label)
    except ValueError:
        return None


def _export_chunk_sample(conn, output_dir: Path, sample_size: int = 100) -> Path:
    """导出 chunk 样本 JSONL，供人工抽检。"""
    import json

    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    sample_path = export_dir / "chunks.sample.jsonl"

    rows = conn.execute(
        """SELECT chunk_id, doc_id, chunk_index, title, section_path, article_no,
                  article_range, token_count, chunk_text, chunk_text_for_embedding
           FROM chunks
           ORDER BY doc_id, chunk_index
           LIMIT ?""",
        (sample_size,),
    ).fetchall()

    with open(sample_path, "w", encoding="utf-8") as f:
        for row in rows:
            item = {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "title": row["title"],
                "section_path": row["section_path"],
                "article_no": row["article_no"],
                "article_range": row["article_range"],
                "token_count": row["token_count"],
                "chunk_text": row["chunk_text"],
                "chunk_text_for_embedding": row["chunk_text_for_embedding"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return sample_path


# ═══════════════════════════════════════════════════════════════
# 阶段 8: embedding 向量化
# ═══════════════════════════════════════════════════════════════

def run_stage_embed(config: Config, logger) -> None:
    """阶段 8: 调用 embedding 服务，把 chunk_text_for_embedding 转成向量。"""
    import json

    from tqdm import tqdm

    from rag_preprocess.database import (
        connect_db,
        init_db,
        insert_build_error,
        reset_chunk_embeddings,
        update_chunk_embedding_status,
    )
    from rag_preprocess.embedding_client import (
        embed_batch,
        get_embedding_base_url,
        normalize_embedding,
        validate_embedding,
    )
    from rag_preprocess.paths import ensure_output_dirs

    logger.info("=" * 50)
    logger.info("阶段 8: embedding 向量化")
    logger.info("=" * 50)

    ensure_output_dirs(config.output_dir)
    conn = connect_db(config.db_path)
    init_db(conn)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if total_chunks == 0:
        logger.info("没有 chunks 数据。请先运行: python scripts/build_kb.py --stage chunk")
        conn.close()
        return

    if config.force:
        rows = conn.execute(
            """SELECT chunk_id, chunk_text_for_embedding
               FROM chunks
               ORDER BY doc_id, chunk_index"""
        ).fetchall()
    elif config.resume:
        rows = conn.execute(
            """SELECT chunk_id, chunk_text_for_embedding
               FROM chunks
               WHERE embedding_status IS NULL OR embedding_status != 'success'
               ORDER BY doc_id, chunk_index"""
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT chunk_id, chunk_text_for_embedding
               FROM chunks
               ORDER BY doc_id, chunk_index"""
        ).fetchall()

    if config.limit is not None:
        rows = rows[:config.limit]

    if not rows:
        logger.info("没有需要向量化的 chunk。")
        conn.close()
        return

    logger.info(f"Embedding 服务: {get_embedding_base_url()}")
    logger.info(f"模型: {config.embedding_model}")
    logger.info(f"期望维度: {config.embedding_dim}")
    logger.info("预检 embedding 服务...")
    preflight_results = embed_batch(
        [rows[0]["chunk_text_for_embedding"] or ""],
        model=config.embedding_model,
        batch_size=1,
    )
    preflight = preflight_results[0] if preflight_results else None
    if preflight is None or not preflight.success or preflight.vector is None:
        message = preflight.error_message if preflight else "embedding 预检无返回"
        logger.error(f"embedding 服务预检失败，不修改数据库: {message}")
        conn.close()
        return
    if not validate_embedding(preflight.vector, config.embedding_dim):
        logger.error(
            "embedding 服务预检维度不匹配，不修改数据库: "
            f"expected={config.embedding_dim}, actual={len(preflight.vector)}"
        )
        conn.close()
        return

    chunk_ids = [row["chunk_id"] for row in rows]
    if not config.resume:
        reset_chunk_embeddings(conn, chunk_ids)

    vector_dir = config.output_dir / "vector_index"
    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_path = vector_dir / "embeddings.jsonl"
    meta_path = vector_dir / "embeddings.meta.json"
    append_mode = config.resume and vector_path.exists()

    if append_mode:
        max_vector_id = conn.execute(
            "SELECT COALESCE(MAX(vector_id), -1) FROM chunks WHERE embedding_status = 'success'"
        ).fetchone()[0]
        next_vector_id = int(max_vector_id) + 1
        file_mode = "a"
    else:
        next_vector_id = 0
        file_mode = "w"

    logger.info(f"待向量化 chunks: {len(rows)} / {total_chunks}")
    logger.info(f"批量大小: {config.embedding_batch_size}")
    logger.info(f"向量文件: {vector_path}")
    logger.info(f"写入模式: {'追加' if append_mode else '覆盖'}")

    success_count = 0
    failed_count = 0
    dim_failed_count = 0
    total_retry_count = 0

    with open(vector_path, file_mode, encoding="utf-8") as f:
        for start in tqdm(
            range(0, len(rows), config.embedding_batch_size),
            desc="embedding",
            unit="batch",
        ):
            batch_rows = rows[start:start + config.embedding_batch_size]
            texts = [row["chunk_text_for_embedding"] or "" for row in batch_rows]
            results = embed_batch(
                texts,
                model=config.embedding_model,
                batch_size=config.embedding_batch_size,
            )

            for row, result in zip(batch_rows, results):
                chunk_id = row["chunk_id"]
                total_retry_count += result.retry_count

                if not result.success or result.vector is None:
                    failed_count += 1
                    message = result.error_message or "embedding failed"
                    update_chunk_embedding_status(
                        conn,
                        chunk_id,
                        "failed",
                        None,
                        commit=False,
                    )
                    insert_build_error(
                        conn,
                        error_id=f"embed:{chunk_id}",
                        stage="embed",
                        error_type="embedding_error",
                        error_message=message[:1000],
                        doc_id=None,
                        commit=False,
                    )
                    continue

                vector = result.vector
                if not validate_embedding(vector, config.embedding_dim):
                    failed_count += 1
                    dim_failed_count += 1
                    update_chunk_embedding_status(
                        conn,
                        chunk_id,
                        "failed",
                        None,
                        commit=False,
                    )
                    insert_build_error(
                        conn,
                        error_id=f"embed_dim:{chunk_id}",
                        stage="embed",
                        error_type="embedding_dim_mismatch",
                        error_message=f"expected={config.embedding_dim}, actual={len(vector)}",
                        doc_id=None,
                        commit=False,
                    )
                    continue

                if config.vector_normalized:
                    vector = normalize_embedding(vector)

                vector_id = next_vector_id
                next_vector_id += 1
                item = {
                    "vector_id": vector_id,
                    "chunk_id": chunk_id,
                    "model": config.embedding_model,
                    "dim": config.embedding_dim,
                    "normalized": config.vector_normalized,
                    "vector": vector,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                update_chunk_embedding_status(
                    conn,
                    chunk_id,
                    "success",
                    vector_id,
                    commit=False,
                )
                success_count += 1

            conn.commit()
            f.flush()

    meta = {
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "vector_normalized": config.vector_normalized,
        "vector_metric": config.vector_metric,
        "vector_file": str(vector_path.as_posix()),
        "selected_chunk_count": len(rows),
        "success_count": success_count,
        "failed_count": failed_count,
        "dim_failed_count": dim_failed_count,
        "total_retry_count": total_retry_count,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    db_success = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'success'"
    ).fetchone()[0]
    db_failed = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'failed'"
    ).fetchone()[0]
    mapped = conn.execute(
        """SELECT COUNT(*) FROM chunks
           WHERE embedding_status = 'success' AND vector_id IS NOT NULL"""
    ).fetchone()[0]

    logger.info("")
    logger.info("=" * 50)
    logger.info("阶段 8 完成")
    logger.info("=" * 50)
    logger.info(f"  本次选择 chunks:      {len(rows):>8}")
    logger.info(f"  本次成功:             {success_count:>8}")
    logger.info(f"  本次失败:             {failed_count:>8}")
    logger.info(f"  维度失败:             {dim_failed_count:>8}")
    logger.info(f"  总重试次数:           {total_retry_count:>8}")
    logger.info(f"  DB success:           {db_success:>8}")
    logger.info(f"  DB failed:            {db_failed:>8}")
    logger.info(f"  success 且有 vector_id:{mapped:>8}")
    logger.info(f"  向量文件:             {vector_path}")
    logger.info(f"  元数据文件:           {meta_path}")

    conn.close()
    logger.info("阶段 8 结束。")


# ═══════════════════════════════════════════════════════════════
# 阶段 9: 构建 FAISS 索引
# ═══════════════════════════════════════════════════════════════

def run_stage_faiss(config: Config, logger) -> None:
    """阶段 9: 从 embeddings.jsonl 构建 FAISS IndexIDMap2 索引。"""
    import json

    from rag_preprocess.database import connect_db, init_db, now_iso
    from rag_preprocess.faiss_builder import (
        FaissUnavailableError,
        build_faiss_index,
        load_embedding_jsonl,
    )
    from rag_preprocess.paths import ensure_output_dirs

    logger.info("=" * 50)
    logger.info("阶段 9: 构建 FAISS 索引")
    logger.info("=" * 50)

    ensure_output_dirs(config.output_dir)
    vector_dir = config.output_dir / "vector_index"
    vector_path = vector_dir / "embeddings.jsonl"
    stage8_meta_path = vector_dir / "embeddings.meta.json"
    index_path = vector_dir / "index.faiss"
    index_meta_path = vector_dir / "index.meta.json"

    if not vector_path.exists():
        logger.error(f"向量文件不存在，请先完成阶段 8: {vector_path}")
        return

    conn = connect_db(config.db_path)
    init_db(conn)

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db_success = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding_status = 'success'"
    ).fetchone()[0]
    db_mapped = conn.execute(
        """SELECT COUNT(*) FROM chunks
           WHERE embedding_status = 'success' AND vector_id IS NOT NULL"""
    ).fetchone()[0]
    db_missing_vector_id = db_success - db_mapped

    if db_mapped == 0:
        logger.error("数据库中没有已成功向量化且带 vector_id 的 chunk，请先运行阶段 8。")
        conn.close()
        return
    if db_missing_vector_id:
        logger.error(f"存在 success 但缺少 vector_id 的 chunk: {db_missing_vector_id}")
        conn.close()
        return
    if db_mapped < total_chunks:
        logger.warning(
            f"当前只完成部分 embedding: {db_mapped} / {total_chunks}。"
            "阶段 9 将基于当前可用向量构建部分索引；全量 embedding 完成后需要重跑阶段 9。"
        )

    logger.info(f"向量文件: {vector_path}")
    logger.info(f"索引文件: {index_path}")
    logger.info(f"期望维度: {config.embedding_dim}")
    logger.info(f"索引度量: {config.vector_metric}")
    logger.info("读取并校验向量文件...")

    try:
        loaded = load_embedding_jsonl(
            vector_path,
            expected_dim=config.embedding_dim,
            limit=config.limit,
            expected_count=db_mapped,
        )
    except Exception as exc:
        logger.error(f"读取向量文件失败，未构建索引: {exc}")
        conn.close()
        return

    vector_count = int(loaded.vectors.shape[0])
    if vector_count == 0:
        logger.error("向量文件为空，未构建索引。")
        conn.close()
        return

    if config.limit is None and loaded.source_line_count != db_mapped:
        logger.error(
            "向量文件行数与数据库 success/vector_id 数量不一致，未构建索引: "
            f"vector_file={loaded.source_line_count}, db_mapped={db_mapped}"
        )
        conn.close()
        return

    logger.info("校验 vector_id/chunk_id 与 SQLite 映射...")
    rows = conn.execute(
        """SELECT vector_id, chunk_id
           FROM chunks
           WHERE embedding_status = 'success' AND vector_id IS NOT NULL"""
    ).fetchall()
    db_vector_to_chunk = {int(row["vector_id"]): row["chunk_id"] for row in rows}
    file_vector_ids = set(int(vector_id) for vector_id in loaded.vector_ids.tolist())

    unknown_ids: list[int] = []
    mismatched_ids: list[int] = []
    for vector_id, chunk_id in zip(loaded.vector_ids.tolist(), loaded.chunk_ids):
        db_chunk_id = db_vector_to_chunk.get(int(vector_id))
        if db_chunk_id is None:
            unknown_ids.append(int(vector_id))
        elif db_chunk_id != chunk_id:
            mismatched_ids.append(int(vector_id))

    missing_ids_count = 0
    if config.limit is None:
        missing_ids_count = len(set(db_vector_to_chunk) - file_vector_ids)

    if unknown_ids or mismatched_ids or missing_ids_count:
        logger.error("向量文件与数据库映射不一致，未构建索引。")
        if unknown_ids:
            logger.error(f"  向量文件中存在数据库没有的 vector_id 示例: {unknown_ids[:5]}")
        if mismatched_ids:
            logger.error(f"  vector_id 对应 chunk_id 不一致示例: {mismatched_ids[:5]}")
        if missing_ids_count:
            logger.error(f"  数据库中有 {missing_ids_count} 个 success vector_id 不在向量文件中")
        conn.close()
        return

    logger.info(f"开始构建 FAISS IndexIDMap2，向量数: {vector_count}")
    try:
        faiss_count = build_faiss_index(
            loaded.vectors,
            loaded.vector_ids,
            index_path,
            metric=config.vector_metric,
        )
    except FaissUnavailableError as exc:
        logger.error(str(exc))
        logger.error("可尝试安装: pip install faiss-cpu")
        logger.error("如果 Windows pip 安装失败，建议用 conda 安装 faiss-cpu。")
        conn.close()
        return
    except Exception as exc:
        logger.error(f"构建 FAISS 索引失败: {exc}")
        conn.close()
        return

    stage8_meta = {}
    if stage8_meta_path.exists():
        with open(stage8_meta_path, "r", encoding="utf-8") as f:
            stage8_meta = json.load(f)

    index_meta = {
        "index_type": "faiss",
        "faiss_factory": "IndexIDMap2(IndexFlatIP)"
        if config.vector_metric == "inner_product"
        else f"IndexIDMap2({config.vector_metric})",
        "embedding_model": stage8_meta.get("embedding_model", config.embedding_model),
        "embedding_dim": config.embedding_dim,
        "vector_metric": config.vector_metric,
        "vector_normalized": config.vector_normalized,
        "index_path": str(index_path.as_posix()),
        "source_vector_file": str(vector_path.as_posix()),
        "vector_count": faiss_count,
        "source_vector_file_line_count": loaded.source_line_count,
        "db_total_chunks": total_chunks,
        "db_embedding_success_count": db_success,
        "db_success_with_vector_id_count": db_mapped,
        "build_limit": config.limit,
        "is_partial_embedding_index": db_mapped < total_chunks,
        "vector_id_min": int(loaded.vector_ids.min()),
        "vector_id_max": int(loaded.vector_ids.max()),
        "created_at": now_iso(),
    }
    with open(index_meta_path, "w", encoding="utf-8") as f:
        json.dump(index_meta, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 50)
    logger.info("阶段 9 完成")
    logger.info("=" * 50)
    logger.info(f"  FAISS 向量数:          {faiss_count:>8}")
    logger.info(f"  DB success/vector_id: {db_mapped:>8}")
    logger.info(f"  chunks 总数:           {total_chunks:>8}")
    logger.info(f"  索引文件:              {index_path}")
    logger.info(f"  元数据文件:            {index_meta_path}")

    conn.close()
    logger.info("阶段 9 结束。")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)

    config = Config(
        rawdata_dir=Path(args.rawdata_dir),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        resume=args.resume,
        force=args.force,
        log_level=args.log_level,
    )

    stage_handlers = {
        "scan": run_stage_scan,
        "records": run_stage_records,
        "init_db": run_stage_init_db,
        "parse_docx": run_stage_parse_docx,
        "structure": run_stage_structure,
        "chunk": run_stage_chunk,
        "embed": run_stage_embed,
        "faiss": run_stage_faiss,
    }

    if args.stage == "all":
        logger.info("全量构建...")
        for stage_name, handler in stage_handlers.items():
            handler(config, logger)
        logger.info("构建完成。")
    else:
        handler = stage_handlers.get(args.stage)
        if handler:
            handler(config, logger)
        else:
            logger.warning(f"阶段 '{args.stage}' 尚未实现")


if __name__ == "__main__":
    main()
