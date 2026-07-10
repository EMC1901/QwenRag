"""创建 SQLite 表、写入文档、chunk、错误日志。

为后续阶段（4-11）预置所有需要的 CRUD 函数。
"""

import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════
# 连接与初始化
# ═══════════════════════════════════════════════════════════════

def connect_db(db_path: Path) -> sqlite3.Connection:
    """连接 SQLite，启用 WAL 模式和外键。"""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """创建所有表（幂等，使用 IF NOT EXISTS）。"""
    conn.executescript(SCHEMA_SQL)
    _ensure_schema_migrations(conn)


def now_iso() -> str:
    """返回当前 UTC 时间 ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# 完整 Schema
# ═══════════════════════════════════════════════════════════════

SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS source_files (
  source_file_id TEXT PRIMARY KEY,
  volume TEXT,
  relative_path TEXT NOT NULL,
  file_name TEXT NOT NULL,
  extension TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  file_hash_sha256 TEXT NOT NULL,
  mtime TEXT,
  path_length INTEGER,
  is_word_file INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS law_records (
  record_id TEXT PRIMARY KEY,
  volume TEXT NOT NULL,
  bbbs TEXT,
  title TEXT,
  gbrq TEXT,
  sxrq TEXT,
  sxx TEXT,
  zdjg_name TEXT,
  flxz TEXT,
  zdjg_code_id TEXT,
  flfg_code_id TEXT,
  my_dir TEXT,
  my_file TEXT,
  my_status TEXT,
  my_time TEXT,
  expected_relative_path TEXT,
  matched_source_file_id TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  source_file_id TEXT NOT NULL,
  record_id TEXT,
  title TEXT,
  flxz TEXT,
  gbrq TEXT,
  sxrq TEXT,
  sxx TEXT,
  zdjg_name TEXT,
  bbbs TEXT,
  flfg_code_id TEXT,
  zdjg_code_id TEXT,
  volume TEXT,
  my_dir TEXT,
  my_file TEXT,
  relative_path TEXT NOT NULL,
  extension TEXT NOT NULL,
  file_size INTEGER,
  file_hash_sha256 TEXT,
  content_hash_sha256 TEXT,
  conversion_status TEXT,
  conversion_error TEXT,
  parse_status TEXT,
  parse_error TEXT,
  converted_file_path TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(source_file_id) REFERENCES source_files(source_file_id),
  FOREIGN KEY(record_id) REFERENCES law_records(record_id)
);

CREATE TABLE IF NOT EXISTS parsed_blocks (
  block_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  block_index INTEGER NOT NULL,
  block_type TEXT NOT NULL,
  text TEXT,
  paragraph_index INTEGER,
  table_index INTEGER,
  row_index INTEGER,
  cell_index INTEGER,
  style_name TEXT,
  detected_level TEXT,
  article_no TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS structured_blocks (
  structured_block_id TEXT PRIMARY KEY,
  block_id TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  block_index INTEGER NOT NULL,
  block_type TEXT NOT NULL,
  raw_text TEXT,
  clean_text TEXT,
  detected_level TEXT,
  section_path TEXT,
  article_no TEXT,
  is_noise INTEGER DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  chunk_text TEXT NOT NULL,
  chunk_text_for_embedding TEXT NOT NULL,
  title TEXT,
  section_path TEXT,
  article_no TEXT,
  article_range TEXT,
  paragraph_start INTEGER,
  paragraph_end INTEGER,
  token_count INTEGER,
  vector_id INTEGER,
  embedding_status TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
  chunk_id UNINDEXED,
  title,
  section_path,
  article_no,
  chunk_text
);

CREATE TABLE IF NOT EXISTS build_errors (
  error_id TEXT PRIMARY KEY,
  source_file_id TEXT,
  doc_id TEXT,
  stage TEXT NOT NULL,
  error_type TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_source_files_volume ON source_files(volume);
CREATE INDEX IF NOT EXISTS idx_source_files_ext ON source_files(extension);
CREATE INDEX IF NOT EXISTS idx_law_records_volume ON law_records(volume);
CREATE INDEX IF NOT EXISTS idx_law_records_matched ON law_records(matched_source_file_id);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_file_id);
CREATE INDEX IF NOT EXISTS idx_documents_record ON documents(record_id);
CREATE INDEX IF NOT EXISTS idx_parsed_blocks_doc ON parsed_blocks(doc_id);
CREATE INDEX IF NOT EXISTS idx_structured_blocks_doc ON structured_blocks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_vector ON chunks(vector_id);
CREATE INDEX IF NOT EXISTS idx_build_errors_stage ON build_errors(stage);
"""


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    """补齐旧数据库缺失的新列。"""
    chunk_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
    }
    if "article_range" not in chunk_columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN article_range TEXT")
    conn.commit()


# ═══════════════════════════════════════════════════════════════
# source_files 写入
# ═══════════════════════════════════════════════════════════════

def insert_source_files(conn: sqlite3.Connection, files: list) -> None:
    """批量写入 source_files（INSERT OR REPLACE）。"""
    rows = []
    for f in files:
        rows.append((
            f.source_file_id, f.volume, f.relative_path, f.file_name,
            f.extension, f.file_size, f.file_hash_sha256, f.mtime,
            f.path_length, 1 if f.is_word_file else 0, now_iso(),
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO source_files
           (source_file_id, volume, relative_path, file_name, extension,
            file_size, file_hash_sha256, mtime, path_length, is_word_file, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════
# law_records 写入
# ═══════════════════════════════════════════════════════════════

def insert_law_records(conn: sqlite3.Connection, records: list) -> None:
    """批量写入 law_records（INSERT OR REPLACE）。"""
    rows = []
    for r in records:
        rows.append((
            r.record_id, r.volume, r.bbbs, r.title, r.gbrq, r.sxrq,
            r.sxx, r.zdjg_name, r.flxz, r.zdjg_code_id, r.flfg_code_id,
            r.my_dir, r.my_file, r.my_status, r.my_time,
            r.expected_relative_path, r.matched_source_file_id, now_iso(),
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO law_records
           (record_id, volume, bbbs, title, gbrq, sxrq, sxx, zdjg_name, flxz,
            zdjg_code_id, flfg_code_id, my_dir, my_file, my_status, my_time,
            expected_relative_path, matched_source_file_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════
# documents 写入（阶段 4/5 使用）
# ═══════════════════════════════════════════════════════════════

def insert_document(
    conn: sqlite3.Connection,
    doc_id: str,
    source_file_id: str,
    relative_path: str,
    extension: str,
    record_id: str | None = None,
    title: str | None = None,
    flxz: str | None = None,
    gbrq: str | None = None,
    sxrq: str | None = None,
    sxx: str | None = None,
    zdjg_name: str | None = None,
    bbbs: str | None = None,
    flfg_code_id: str | None = None,
    zdjg_code_id: str | None = None,
    volume: str | None = None,
    my_dir: str | None = None,
    my_file: str | None = None,
    file_size: int | None = None,
    file_hash_sha256: str | None = None,
    content_hash_sha256: str | None = None,
    conversion_status: str | None = None,
    conversion_error: str | None = None,
    parse_status: str | None = None,
    parse_error: str | None = None,
    converted_file_path: str | None = None,
) -> None:
    """写入或更新单条 document 记录。"""
    conn.execute(
        """INSERT OR REPLACE INTO documents
           (doc_id, source_file_id, record_id, title, flxz, gbrq, sxrq, sxx,
            zdjg_name, bbbs, flfg_code_id, zdjg_code_id, volume, my_dir, my_file,
            relative_path, extension, file_size, file_hash_sha256,
            content_hash_sha256, conversion_status, conversion_error,
            parse_status, parse_error, converted_file_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id, source_file_id, record_id, title, flxz, gbrq, sxrq, sxx,
            zdjg_name, bbbs, flfg_code_id, zdjg_code_id, volume, my_dir, my_file,
            relative_path, extension, file_size, file_hash_sha256,
            content_hash_sha256, conversion_status, conversion_error,
            parse_status, parse_error, converted_file_path, now_iso(),
        ),
    )
    conn.commit()


def get_source_files_for_parsing(
    conn: sqlite3.Connection,
    extension: str = ".docx",
    limit: int | None = None,
    resume: bool = False,
    force: bool = False,
) -> list[sqlite3.Row]:
    """获取待解析的源文件列表。

    三段式控制：
    - 默认（resume=False, force=False）：全量，不跳过任何文件
    - resume=True：只取未处理或处理失败的，跳过已成功
    - force=True：只跳过已成功，失败的重新处理（覆盖 resume）

    返回 source_files 行 + 关联的 law_records 字段。
    """
    if force:
        # 只跳过已成功的，失败的重新处理
        status_filter = "AND (d.parse_status IS NULL OR d.parse_status != 'success')"
    elif resume:
        # 跳过已成功和已失败的
        status_filter = "AND (d.parse_status IS NULL OR d.parse_status NOT IN ('success', 'failed'))"
    else:
        # 全量，不跳过
        status_filter = ""

    query = f"""
        SELECT sf.*, lr.record_id, lr.title AS record_title, lr.flxz,
               lr.gbrq, lr.sxrq, lr.sxx, lr.zdjg_name
        FROM source_files sf
        LEFT JOIN law_records lr ON sf.source_file_id = lr.matched_source_file_id
        LEFT JOIN documents d ON sf.source_file_id = d.doc_id
        WHERE sf.is_word_file = 1
          AND sf.extension = ?
          {status_filter}
        ORDER BY sf.relative_path
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return conn.execute(query, (extension,)).fetchall()


def count_documents_by_status(conn: sqlite3.Connection) -> dict:
    """按 parse_status 统计 documents 数量。"""
    rows = conn.execute(
        "SELECT parse_status, COUNT(*) FROM documents GROUP BY parse_status"
    ).fetchall()
    return {row[0] or "null": row[1] for row in rows}


# ═══════════════════════════════════════════════════════════════
# parsed_blocks 写入（阶段 4/5 使用）
# ═══════════════════════════════════════════════════════════════

def insert_parsed_blocks(
    conn: sqlite3.Connection,
    blocks: list,
    doc_id: str,
) -> None:
    """批量写入 parsed_blocks，同一 doc_id 下先删后插。"""
    # 删除旧数据
    conn.execute("DELETE FROM parsed_blocks WHERE doc_id = ?", (doc_id,))
    rows = []
    for b in blocks:
        rows.append((
            _make_block_id(doc_id, b.block_index),
            doc_id,
            b.block_index,
            b.block_type,
            b.text,
            getattr(b, "paragraph_index", None),
            getattr(b, "table_index", None),
            getattr(b, "row_index", None),
            getattr(b, "cell_index", None),
            getattr(b, "style_name", None),
            getattr(b, "detected_level", None),
            getattr(b, "article_no", None),
            now_iso(),
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO parsed_blocks
           (block_id, doc_id, block_index, block_type, text,
            paragraph_index, table_index, row_index, cell_index,
            style_name, detected_level, article_no, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def _make_block_id(doc_id: str, block_index: int) -> str:
    raw = f"{doc_id}:block:{block_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _make_structured_block_id(doc_id: str, block_index: int) -> str:
    raw = f"{doc_id}:structured_block:{block_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
# structured_blocks 写入（阶段 6 使用）
# ═══════════════════════════════════════════════════════════════

def insert_structured_blocks(
    conn: sqlite3.Connection,
    structured_blocks: list,
    doc_id: str,
) -> None:
    """批量写入 structured_blocks，同一 doc_id 下先删后插。

    Args:
        conn: 数据库连接
        structured_blocks: StructuredBlock 对象列表
        doc_id: 所属文档 ID
    """
    conn.execute("DELETE FROM structured_blocks WHERE doc_id = ?", (doc_id,))
    rows = []
    for output_index, sb in enumerate(structured_blocks):
        raw_text = getattr(sb, "raw_text", getattr(sb, "text", ""))
        clean_text = getattr(sb, "clean_text", getattr(sb, "text", ""))
        is_noise = 1 if getattr(sb, "is_noise", False) else 0
        rows.append((
            _make_structured_block_id(doc_id, output_index),
            getattr(sb, "block_id", "") or _make_block_id(doc_id, sb.block_index),
            doc_id,
            output_index,
            getattr(sb, "block_type", "paragraph"),
            raw_text,
            clean_text,
            sb.law_level.value if sb.law_level else None,                   # detected_level
            sb.section_path,
            sb.article_no,
            is_noise,
            now_iso(),
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO structured_blocks
           (structured_block_id, block_id, doc_id, block_index, block_type,
            raw_text, clean_text, detected_level, section_path, article_no,
            is_noise, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def get_structured_blocks_by_doc(
    conn: sqlite3.Connection,
    doc_id: str,
) -> list[sqlite3.Row]:
    """按 doc_id 查询 structured_blocks，按 block_index 排序。"""
    return conn.execute(
        "SELECT * FROM structured_blocks WHERE doc_id = ? ORDER BY block_index",
        (doc_id,),
    ).fetchall()


# ═══════════════════════════════════════════════════════════════
# chunks 写入（阶段 7 使用）
# ═══════════════════════════════════════════════════════════════

def clear_chunks_for_doc(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    clear_fts: bool = True,
    commit: bool = True,
) -> None:
    """删除某个文档的 chunks 及其 FTS 记录。"""
    if clear_fts:
        conn.execute(
            """DELETE FROM chunk_fts
               WHERE chunk_id IN (
                   SELECT chunk_id FROM chunks WHERE doc_id = ?
               )""",
            (doc_id,),
        )
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    if commit:
        conn.commit()


def insert_chunks(
    conn: sqlite3.Connection,
    chunks: list,
    *,
    clear_fts: bool = True,
    commit: bool = True,
) -> None:
    """批量写入 chunks，同一 doc_id 下先删后插。"""
    if not chunks:
        return
    doc_id = chunks[0].doc_id
    clear_chunks_for_doc(conn, doc_id, clear_fts=clear_fts, commit=False)
    rows = []
    for c in chunks:
        rows.append((
            c.chunk_id,
            c.doc_id,
            c.chunk_index,
            c.chunk_text,
            c.chunk_text_for_embedding,
            c.title,
            c.section_path,
            c.article_no,
            c.article_range,
            c.paragraph_start,
            c.paragraph_end,
            c.token_count,
            c.vector_id,
            c.embedding_status,
            now_iso(),
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO chunks
           (chunk_id, doc_id, chunk_index, chunk_text, chunk_text_for_embedding,
            title, section_path, article_no, article_range, paragraph_start, paragraph_end,
            token_count, vector_id, embedding_status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    if commit:
        conn.commit()


def insert_chunk_fts(conn: sqlite3.Connection, chunks: list) -> None:
    """将 chunk 内容写入 FTS5 索引。"""
    if not chunks:
        return

    conn.executemany(
        "DELETE FROM chunk_fts WHERE chunk_id = ?",
        [(c.chunk_id,) for c in chunks],
    )

    rows = []
    for c in chunks:
        rows.append((
            c.chunk_id,
            c.title or "",
            c.section_path or "",
            c.article_range or c.article_no or "",
            c.chunk_text,
        ))
    conn.executemany(
        "INSERT INTO chunk_fts (chunk_id, title, section_path, article_no, chunk_text) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def clear_chunk_fts_for_docs(
    conn: sqlite3.Connection,
    doc_ids: list[str],
    *,
    commit: bool = True,
) -> None:
    """一次性清理指定文档的旧 FTS 记录。"""
    if not doc_ids:
        return

    conn.execute("DROP TABLE IF EXISTS temp.clear_chunk_fts_doc_ids")
    conn.execute(
        "CREATE TEMP TABLE clear_chunk_fts_doc_ids (doc_id TEXT PRIMARY KEY)"
    )
    conn.executemany(
        "INSERT OR IGNORE INTO clear_chunk_fts_doc_ids (doc_id) VALUES (?)",
        [(doc_id,) for doc_id in doc_ids],
    )
    conn.execute(
        """DELETE FROM chunk_fts
           WHERE chunk_id IN (
               SELECT c.chunk_id
               FROM chunks c
               INNER JOIN clear_chunk_fts_doc_ids d ON c.doc_id = d.doc_id
           )"""
    )
    conn.execute("DROP TABLE IF EXISTS temp.clear_chunk_fts_doc_ids")

    if commit:
        conn.commit()


def rebuild_chunk_fts(
    conn: sqlite3.Connection,
    *,
    doc_ids: list[str] | None = None,
    commit: bool = True,
) -> int:
    """从 chunks 表统一重建 chunk_fts，返回 FTS 总行数。

    doc_ids=None 时重建全库；传入 doc_ids 时只重建这些文档对应的 FTS。
    调用者若只重建部分文档，应先清理这些文档的旧 FTS 记录。
    """
    if doc_ids is None:
        conn.execute("DELETE FROM chunk_fts")
        conn.execute(
            """INSERT INTO chunk_fts (chunk_id, title, section_path, article_no, chunk_text)
               SELECT chunk_id,
                      COALESCE(title, ''),
                      COALESCE(section_path, ''),
                      COALESCE(article_range, article_no, ''),
                      chunk_text
               FROM chunks"""
        )
    else:
        conn.execute("DROP TABLE IF EXISTS temp.rebuild_chunk_doc_ids")
        conn.execute(
            "CREATE TEMP TABLE rebuild_chunk_doc_ids (doc_id TEXT PRIMARY KEY)"
        )
        conn.executemany(
            "INSERT OR IGNORE INTO rebuild_chunk_doc_ids (doc_id) VALUES (?)",
            [(doc_id,) for doc_id in doc_ids],
        )
        conn.execute(
            """DELETE FROM chunk_fts
               WHERE chunk_id IN (
                   SELECT c.chunk_id
                   FROM chunks c
                   INNER JOIN rebuild_chunk_doc_ids d ON c.doc_id = d.doc_id
               )"""
        )
        conn.execute(
            """INSERT INTO chunk_fts (chunk_id, title, section_path, article_no, chunk_text)
               SELECT c.chunk_id,
                      COALESCE(c.title, ''),
                      COALESCE(c.section_path, ''),
                      COALESCE(c.article_range, c.article_no, ''),
                      c.chunk_text
               FROM chunks c
               INNER JOIN rebuild_chunk_doc_ids d ON c.doc_id = d.doc_id"""
        )
        conn.execute("DROP TABLE IF EXISTS temp.rebuild_chunk_doc_ids")

    count = conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
    if commit:
        conn.commit()
    return count


def update_chunk_vector_id(
    conn: sqlite3.Connection,
    chunk_id: str,
    vector_id: int,
    embedding_status: str = "success",
) -> None:
    """回填 chunk 的 vector_id。"""
    conn.execute(
        "UPDATE chunks SET vector_id = ?, embedding_status = ? WHERE chunk_id = ?",
        (vector_id, embedding_status, chunk_id),
    )
    conn.commit()


def update_chunk_embedding_status(
    conn: sqlite3.Connection,
    chunk_id: str,
    embedding_status: str,
    vector_id: int | None = None,
    *,
    commit: bool = True,
) -> None:
    """更新单个 chunk 的 embedding 状态。"""
    conn.execute(
        "UPDATE chunks SET vector_id = ?, embedding_status = ? WHERE chunk_id = ?",
        (vector_id, embedding_status, chunk_id),
    )
    if commit:
        conn.commit()


def reset_chunk_embeddings(
    conn: sqlite3.Connection,
    chunk_ids: list[str] | None = None,
    *,
    commit: bool = True,
) -> None:
    """清空 chunk 的 vector_id 和 embedding_status。"""
    if chunk_ids is None:
        conn.execute("UPDATE chunks SET vector_id = NULL, embedding_status = NULL")
    elif chunk_ids:
        conn.executemany(
            "UPDATE chunks SET vector_id = NULL, embedding_status = NULL WHERE chunk_id = ?",
            [(chunk_id,) for chunk_id in chunk_ids],
        )
    if commit:
        conn.commit()


def get_chunks_for_embedding(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """获取待 embedding 的 chunk 列表。"""
    query = """
        SELECT chunk_id, chunk_text_for_embedding
        FROM chunks
        WHERE embedding_status IS NULL OR embedding_status != 'success'
        ORDER BY doc_id, chunk_index
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return conn.execute(query).fetchall()


# ═══════════════════════════════════════════════════════════════
# build_errors 写入
# ═══════════════════════════════════════════════════════════════

def insert_build_error(
    conn: sqlite3.Connection,
    error_id: str,
    stage: str,
    error_type: str | None = None,
    error_message: str | None = None,
    source_file_id: str | None = None,
    doc_id: str | None = None,
    *,
    commit: bool = True,
) -> None:
    """写入构建错误日志。"""
    conn.execute(
        """INSERT OR REPLACE INTO build_errors
           (error_id, source_file_id, doc_id, stage, error_type, error_message, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (error_id, source_file_id, doc_id, stage, error_type, error_message, now_iso()),
    )
    if commit:
        conn.commit()


# ═══════════════════════════════════════════════════════════════
# 查询辅助
# ═══════════════════════════════════════════════════════════════

def get_table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """返回所有用户表的行数。"""
    user_tables = [
        "source_files", "law_records", "documents", "parsed_blocks",
        "structured_blocks", "chunks", "chunk_fts", "build_errors",
    ]
    counts = {}
    for t in user_tables:
        try:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            cnt = -1  # 表不存在
        counts[t] = cnt
    return counts
