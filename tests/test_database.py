"""测试 database 模块。"""

import sqlite3
from pathlib import Path

import pytest

from rag_preprocess.database import (
    connect_db,
    init_db,
    get_table_counts,
    now_iso,
    _make_block_id,
    _make_structured_block_id,
    insert_source_files,
    insert_document,
    clear_chunk_fts_for_docs,
    clear_chunks_for_doc,
    insert_chunks,
    insert_chunk_fts,
    rebuild_chunk_fts,
    insert_structured_blocks,
    get_source_files_for_parsing,
    get_structured_blocks_by_doc,
)
from rag_preprocess.chunker import Chunk
from rag_preprocess.law_structure import (
    StructuredBlock,
    LawLevel,
)


@pytest.fixture
def tmp_conn(tmp_path: Path):
    """创建临时数据库连接并初始化。"""
    db = tmp_path / "test.db"
    conn = connect_db(db)
    init_db(conn)
    yield conn
    conn.close()


def test_connect_db_creates_file(tmp_path: Path):
    db = tmp_path / "new.db"
    conn = connect_db(db)
    assert db.exists()
    conn.close()


def test_init_db_creates_all_tables(tmp_conn: sqlite3.Connection):
    tables = tmp_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {r[0] for r in tables}
    expected = {"source_files", "law_records", "documents", "parsed_blocks",
                "structured_blocks", "chunks", "chunk_fts", "build_errors"}
    assert expected <= table_names, f"Missing: {expected - table_names}"
    chunk_columns = {
        r[1] for r in tmp_conn.execute("PRAGMA table_info(chunks)").fetchall()
    }
    assert "article_range" in chunk_columns


def test_get_table_counts(tmp_conn: sqlite3.Connection):
    counts = get_table_counts(tmp_conn)
    assert counts["source_files"] == 0
    assert counts["law_records"] == 0
    assert counts["documents"] == 0
    assert counts["structured_blocks"] == 0
    assert counts["chunk_fts"] == 0


def test_now_iso_format():
    ts = now_iso()
    assert "T" in ts
    assert len(ts) >= 20


def test_make_block_id_deterministic():
    b1 = _make_block_id("doc1", 3)
    b2 = _make_block_id("doc1", 3)
    assert b1 == b2
    assert len(b1) == 64


def test_make_block_id_different():
    b1 = _make_block_id("doc1", 1)
    b2 = _make_block_id("doc1", 2)
    assert b1 != b2


def test_make_structured_block_id_deterministic():
    b1 = _make_structured_block_id("doc1", 5)
    b2 = _make_structured_block_id("doc1", 5)
    assert b1 == b2
    assert len(b1) == 64


def test_make_structured_block_id_different():
    b1 = _make_structured_block_id("doc1", 1)
    b2 = _make_structured_block_id("doc1", 2)
    b3 = _make_structured_block_id("doc2", 1)
    assert b1 != b2
    assert b1 != b3


def test_init_db_idempotent(tmp_conn: sqlite3.Connection):
    """init_db 可以安全地重复调用。"""
    init_db(tmp_conn)  # 不应抛出异常
    init_db(tmp_conn)  # 再次调用
    counts = get_table_counts(tmp_conn)
    assert counts["source_files"] == 0


def test_foreign_keys_enabled(tmp_conn: sqlite3.Connection):
    fk = tmp_conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


# ═══════════════════════════════════════════════════════════════
# structured_blocks 表测试
# ═══════════════════════════════════════════════════════════════

class TestStructuredBlocks:
    """测试 structured_blocks 表的 CRUD 操作。"""

    def test_insert_and_query(self, tmp_conn: sqlite3.Connection):
        """写入并查询 structured_blocks。"""
        doc_id = "test_doc_001"
        blocks = [
            StructuredBlock(
                block_id="block_0",
                text="第一章 总则",
                law_level=LawLevel.CHAPTER,
                section_path="第一章 总则",
                block_index=0,
                block_type="paragraph",
            ),
            StructuredBlock(
                block_id="block_1",
                text="第一条 立法目的",
                law_level=LawLevel.ARTICLE,
                section_path="第一章 总则",
                article_no="第一条",
                block_index=1,
                block_type="paragraph",
            ),
            StructuredBlock(
                block_id="block_2",
                text="本法为了规范...",
                law_level=None,
                section_path="第一章 总则",
                article_no=None,
                block_index=2,
                block_type="paragraph",
            ),
        ]

        insert_structured_blocks(tmp_conn, blocks, doc_id)

        rows = get_structured_blocks_by_doc(tmp_conn, doc_id)
        assert len(rows) == 3

        # 验证排序
        assert rows[0]["block_index"] == 0
        assert rows[1]["block_index"] == 1
        assert rows[2]["block_index"] == 2

        # 验证字段内容
        assert rows[0]["raw_text"] == "第一章 总则"
        assert rows[0]["detected_level"] == "章"
        assert rows[0]["section_path"] == "第一章 总则"

        assert rows[1]["raw_text"] == "第一条 立法目的"
        assert rows[1]["detected_level"] == "条"
        assert rows[1]["article_no"] == "第一条"

        assert rows[2]["detected_level"] is None
        assert rows[2]["article_no"] is None

    def test_insert_replaces_old_data(self, tmp_conn: sqlite3.Connection):
        """同一 doc_id 重新写入应替换旧数据（先删后插）。"""
        doc_id = "test_doc_replace"
        blocks_v1 = [
            StructuredBlock(
                block_id="b0", text="旧版本内容", block_index=0,
            ),
        ]
        insert_structured_blocks(tmp_conn, blocks_v1, doc_id)
        assert len(get_structured_blocks_by_doc(tmp_conn, doc_id)) == 1

        blocks_v2 = [
            StructuredBlock(
                block_id="b0", text="新版本内容", block_index=0,
            ),
            StructuredBlock(
                block_id="b1", text="新增内容", block_index=1,
            ),
        ]
        insert_structured_blocks(tmp_conn, blocks_v2, doc_id)
        rows = get_structured_blocks_by_doc(tmp_conn, doc_id)
        assert len(rows) == 2
        assert rows[0]["raw_text"] == "新版本内容"

    def test_empty_blocks(self, tmp_conn: sqlite3.Connection):
        """空列表写入不应崩溃。"""
        insert_structured_blocks(tmp_conn, [], "empty_doc")
        rows = get_structured_blocks_by_doc(tmp_conn, "empty_doc")
        assert len(rows) == 0

    def test_is_noise_default(self, tmp_conn: sqlite3.Connection):
        """is_noise 默认为 0。"""
        doc_id = "test_noise_default"
        blocks = [
            StructuredBlock(block_id="b0", text="正常内容", block_index=0),
        ]
        insert_structured_blocks(tmp_conn, blocks, doc_id)
        rows = get_structured_blocks_by_doc(tmp_conn, doc_id)
        assert rows[0]["is_noise"] == 0

    def test_all_level_types(self, tmp_conn: sqlite3.Connection):
        """验证所有法规层级都能正确存储。"""
        doc_id = "test_all_levels"
        level_cases = [
            ("第一编 总则", LawLevel.PART, "编"),
            ("第一章 基本原则", LawLevel.CHAPTER, "章"),
            ("第一节 适用范围", LawLevel.SECTION, "节"),
            ("第一条 立法目的", LawLevel.ARTICLE, "条"),
            ("（一）具体事项", LawLevel.ITEM, "项"),
        ]
        blocks = [
            StructuredBlock(
                block_id=f"b{i}",
                text=text,
                law_level=level,
                block_index=i,
            )
            for i, (text, level, _) in enumerate(level_cases)
        ]
        insert_structured_blocks(tmp_conn, blocks, doc_id)

        rows = get_structured_blocks_by_doc(tmp_conn, doc_id)
        for row, (_, _, expected_label) in zip(rows, level_cases):
            assert row["detected_level"] == expected_label

    def test_different_docs_isolated(self, tmp_conn: sqlite3.Connection):
        """不同 doc_id 的数据互不影响。"""
        blocks_a = [
            StructuredBlock(block_id="a0", text="文档A内容", block_index=0),
        ]
        blocks_b = [
            StructuredBlock(block_id="b0", text="文档B内容", block_index=0),
        ]
        insert_structured_blocks(tmp_conn, blocks_a, "doc_a")
        insert_structured_blocks(tmp_conn, blocks_b, "doc_b")

        assert len(get_structured_blocks_by_doc(tmp_conn, "doc_a")) == 1
        assert len(get_structured_blocks_by_doc(tmp_conn, "doc_b")) == 1
        assert get_structured_blocks_by_doc(tmp_conn, "doc_a")[0]["raw_text"] == "文档A内容"


# ═══════════════════════════════════════════════════════════════
# chunks / chunk_fts 表测试
# ═══════════════════════════════════════════════════════════════

class TestChunks:
    """测试 chunks 与 chunk_fts 的写入和重建。"""

    def _insert_minimal_document(self, conn: sqlite3.Connection, doc_id: str):
        now = now_iso()
        conn.execute(
            """INSERT OR REPLACE INTO source_files
               (source_file_id, volume, relative_path, file_name, extension,
                file_size, file_hash_sha256, is_word_file, created_at)
               VALUES (?, 'vol1', ?, ?, '.docx', 100, ?, 1, ?)""",
            (doc_id, f"law/{doc_id}.docx", f"{doc_id}.docx", f"hash_{doc_id}", now),
        )
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, source_file_id, title, relative_path, extension, parse_status, created_at)
               VALUES (?, ?, ?, ?, '.docx', 'success', ?)""",
            (doc_id, doc_id, "测试法规", f"law/{doc_id}.docx", now),
        )
        conn.commit()

    def _chunk(
        self,
        doc_id: str,
        chunk_id: str,
        index: int,
        text: str,
        article_no: str | None = None,
        article_range: str | None = None,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_index=index,
            chunk_text=text,
            chunk_text_for_embedding=f"法规标题：测试法规\n\n正文：\n{text}",
            title="测试法规",
            article_no=article_no,
            article_range=article_range,
            token_count=10,
        )

    def test_insert_chunks_replaces_old_chunks_and_fts(self, tmp_conn: sqlite3.Connection):
        """同一 doc_id 重建 chunks 时，旧 chunks 和旧 FTS 记录都应删除。"""
        doc_id = "doc_chunks_replace"
        self._insert_minimal_document(tmp_conn, doc_id)

        old_chunks = [
            self._chunk(doc_id, "old_chunk_1", 0, "旧内容一"),
            self._chunk(doc_id, "old_chunk_2", 1, "旧内容二"),
        ]
        insert_chunks(tmp_conn, old_chunks)
        insert_chunk_fts(tmp_conn, old_chunks)

        assert tmp_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert tmp_conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == 2

        new_chunks = [
            self._chunk(doc_id, "new_chunk_1", 0, "新内容"),
        ]
        insert_chunks(tmp_conn, new_chunks)
        insert_chunk_fts(tmp_conn, new_chunks)

        chunk_ids = [
            r[0] for r in tmp_conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id")
        ]
        fts_ids = [
            r[0] for r in tmp_conn.execute("SELECT chunk_id FROM chunk_fts ORDER BY chunk_id")
        ]
        assert chunk_ids == ["new_chunk_1"]
        assert fts_ids == ["new_chunk_1"]

    def test_insert_chunks_writes_article_range(self, tmp_conn: sqlite3.Connection):
        """chunks.article_range 应正常写入，FTS 条号列优先使用范围。"""
        doc_id = "doc_chunks_article_range"
        self._insert_minimal_document(tmp_conn, doc_id)

        chunks = [
            self._chunk(
                doc_id,
                "range_chunk_1",
                0,
                "第一条 内容。\n\n第二条 内容。",
                article_no="第二条",
                article_range="第一条-第二条",
            )
        ]
        insert_chunks(tmp_conn, chunks)
        insert_chunk_fts(tmp_conn, chunks)

        row = tmp_conn.execute(
            "SELECT article_no, article_range FROM chunks WHERE chunk_id = ?",
            ("range_chunk_1",),
        ).fetchone()
        assert row["article_no"] == "第二条"
        assert row["article_range"] == "第一条-第二条"

        fts_row = tmp_conn.execute(
            "SELECT article_no FROM chunk_fts WHERE chunk_id = ?",
            ("range_chunk_1",),
        ).fetchone()
        assert fts_row["article_no"] == "第一条-第二条"

    def test_clear_chunks_for_doc_removes_chunks_and_fts(self, tmp_conn: sqlite3.Connection):
        """显式清理文档 chunk 时，应同时清理 FTS。"""
        doc_id = "doc_chunks_clear"
        self._insert_minimal_document(tmp_conn, doc_id)
        chunks = [self._chunk(doc_id, "chunk_to_clear", 0, "待清理内容")]
        insert_chunks(tmp_conn, chunks)
        insert_chunk_fts(tmp_conn, chunks)

        clear_chunks_for_doc(tmp_conn, doc_id)

        assert tmp_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert tmp_conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == 0

    def test_rebuild_chunk_fts_for_selected_docs(self, tmp_conn: sqlite3.Connection):
        """部分重建 FTS 时，应只重建指定文档并保留其他文档。"""
        doc_a = "doc_fts_a"
        doc_b = "doc_fts_b"
        self._insert_minimal_document(tmp_conn, doc_a)
        self._insert_minimal_document(tmp_conn, doc_b)

        chunks_a_v1 = [self._chunk(doc_a, "a_old", 0, "文档A旧内容")]
        chunks_b = [self._chunk(doc_b, "b_keep", 0, "文档B保留内容")]
        insert_chunks(tmp_conn, chunks_a_v1)
        insert_chunk_fts(tmp_conn, chunks_a_v1)
        insert_chunks(tmp_conn, chunks_b)
        insert_chunk_fts(tmp_conn, chunks_b)

        chunks_a_v2 = [self._chunk(doc_a, "a_new", 0, "文档A新内容")]
        insert_chunks(tmp_conn, chunks_a_v2)
        fts_count = rebuild_chunk_fts(tmp_conn, doc_ids=[doc_a])

        fts_ids = [
            r[0] for r in tmp_conn.execute("SELECT chunk_id FROM chunk_fts ORDER BY chunk_id")
        ]
        assert fts_count == 2
        assert fts_ids == ["a_new", "b_keep"]

    def test_clear_chunk_fts_for_docs(self, tmp_conn: sqlite3.Connection):
        """应能一次性清理指定文档的 FTS 记录。"""
        doc_a = "doc_clear_fts_a"
        doc_b = "doc_clear_fts_b"
        self._insert_minimal_document(tmp_conn, doc_a)
        self._insert_minimal_document(tmp_conn, doc_b)

        chunks_a = [self._chunk(doc_a, "a_clear", 0, "文档A")]
        chunks_b = [self._chunk(doc_b, "b_keep_after_clear", 0, "文档B")]
        insert_chunks(tmp_conn, chunks_a)
        insert_chunk_fts(tmp_conn, chunks_a)
        insert_chunks(tmp_conn, chunks_b)
        insert_chunk_fts(tmp_conn, chunks_b)

        clear_chunk_fts_for_docs(tmp_conn, [doc_a])

        fts_ids = [
            r[0] for r in tmp_conn.execute("SELECT chunk_id FROM chunk_fts ORDER BY chunk_id")
        ]
        assert fts_ids == ["b_keep_after_clear"]


# ═══════════════════════════════════════════════════════════════
# get_source_files_for_parsing 三段式模式测试
# ═══════════════════════════════════════════════════════════════

class TestGetSourceFilesForParsingModes:
    """测试 resume/force 三段式：默认全量 / --resume 跳过 / --force 重试。"""

    def _setup_test_data(self, conn: sqlite3.Connection):
        """创建 4 个 source_files + 对应 documents，覆盖三种 parse_status。"""
        import hashlib
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        files = [
            ("sf_success", "vol1", "law/success.docx", "success.docx", ".docx", 1024),
            ("sf_failed",  "vol1", "law/failed.docx",  "failed.docx",  ".docx", 2048),
            ("sf_null",    "vol1", "law/null.docx",     "null.docx",    ".docx", 512),
            ("sf_empty",   "vol1", "law/empty.docx",    "empty.docx",   ".docx", 256),
        ]
        for sfid, vol, rp, fn, ext, sz in files:
            conn.execute(
                """INSERT OR REPLACE INTO source_files
                   (source_file_id, volume, relative_path, file_name, extension,
                    file_size, file_hash_sha256, is_word_file, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (sfid, vol, rp, fn, ext, sz, "hash_" + sfid, now),
            )

        # documents: success, failed, null(=没记录), empty(=parse_status='empty')
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, source_file_id, relative_path, extension, parse_status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("sf_success", "sf_success", "law/success.docx", ".docx", "success", now),
        )
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, source_file_id, relative_path, extension, parse_status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("sf_failed", "sf_failed", "law/failed.docx", ".docx", "failed", now),
        )
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, source_file_id, relative_path, extension, parse_status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("sf_empty", "sf_empty", "law/empty.docx", ".docx", "empty", now),
        )
        # sf_null 不写入 documents（模拟未处理）
        conn.commit()

    def test_default_mode_returns_all(self, tmp_conn: sqlite3.Connection):
        """默认模式：不跳过任何文件，全量返回。"""
        self._setup_test_data(tmp_conn)
        rows = get_source_files_for_parsing(
            tmp_conn, extension=".docx", resume=False, force=False,
        )
        ids = {r["source_file_id"] for r in rows}
        assert ids == {"sf_success", "sf_failed", "sf_null", "sf_empty"}

    def test_resume_skips_success_and_failed(self, tmp_conn: sqlite3.Connection):
        """--resume：跳过 success 和 failed，保留 null 和 empty。"""
        self._setup_test_data(tmp_conn)
        rows = get_source_files_for_parsing(
            tmp_conn, extension=".docx", resume=True, force=False,
        )
        ids = {r["source_file_id"] for r in rows}
        assert ids == {"sf_null", "sf_empty"}

    def test_force_skips_only_success(self, tmp_conn: sqlite3.Connection):
        """--force：只跳过 success，failed/empty/null 都重新处理。"""
        self._setup_test_data(tmp_conn)
        rows = get_source_files_for_parsing(
            tmp_conn, extension=".docx", resume=False, force=True,
        )
        ids = {r["source_file_id"] for r in rows}
        assert ids == {"sf_null", "sf_failed", "sf_empty"}

    def test_force_overrides_resume(self, tmp_conn: sqlite3.Connection):
        """--resume --force：force 优先生效，只跳过 success。"""
        self._setup_test_data(tmp_conn)
        rows = get_source_files_for_parsing(
            tmp_conn, extension=".docx", resume=True, force=True,
        )
        ids = {r["source_file_id"] for r in rows}
        assert ids == {"sf_null", "sf_failed", "sf_empty"}

    def test_limit_applies_after_filter(self, tmp_conn: sqlite3.Connection):
        """--limit 在 mode 过滤之后再截断。"""
        self._setup_test_data(tmp_conn)
        rows = get_source_files_for_parsing(
            tmp_conn, extension=".docx", resume=False, force=False, limit=2,
        )
        assert len(rows) == 2

    def test_different_extension_not_returned(self, tmp_conn: sqlite3.Connection):
        """只返回指定扩展名的文件。"""
        self._setup_test_data(tmp_conn)
        rows = get_source_files_for_parsing(
            tmp_conn, extension=".doc", resume=False, force=False,
        )
        assert len(rows) == 0

    def test_no_data_returns_empty(self, tmp_conn: sqlite3.Connection):
        """无数据时不崩溃。"""
        rows = get_source_files_for_parsing(
            tmp_conn, extension=".docx", resume=False, force=False,
        )
        assert rows == []
