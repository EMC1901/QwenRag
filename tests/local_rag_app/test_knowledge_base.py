"""Tests for stage-3 read-only local knowledge-base loading."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from local_rag_app.config import PROJECT_ROOT, Settings
from local_rag_app.knowledge_base import (
    FtsSearchFallbackError,
    KnowledgeBase,
    KnowledgeBaseLoadError,
    KnowledgeBaseQueryError,
    build_fts_query,
    build_fts_query_plan,
)


_TEST_ROOT = PROJECT_ROOT / "rag_data" / "test_stage6_tmp"


class FakeIndex:
    """Small FAISS-shaped object that keeps these tests independent of faiss-cpu."""

    def __init__(self, *, dimension: int = 3, vector_count: int = 3) -> None:
        self.d = dimension
        self.ntotal = vector_count


class FakeFaissLoader:
    """Record load attempts and return one configurable index object."""

    def __init__(self, index: Any | None = None) -> None:
        self.index = index or FakeIndex()
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> Any:
        self.paths.append(path)
        return self.index


class SearchableFakeIndex(FakeIndex):
    """FAISS-shaped index with deterministic vectors, scores, and call capture."""

    def __init__(
        self,
        scores: Any,
        vector_ids: Any,
    ) -> None:
        super().__init__()
        self._scores = np.asarray(scores)
        self._vector_ids = np.asarray(vector_ids)
        self.search_calls: list[tuple[np.ndarray, int]] = []

    def search(self, query: np.ndarray, top_k: int):
        self.search_calls.append((query.copy(), top_k))
        return self._scores, self._vector_ids


@pytest.fixture
def knowledge_base_dir() -> Path:
    """Create a temporary synthetic asset directory inside the writable workspace."""
    _TEST_ROOT.mkdir(parents=True, exist_ok=True)
    root = _TEST_ROOT / f"knowledge-base-{uuid4().hex}"
    root.mkdir()
    _write_assets(root)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _write_assets(root: Path, **metadata_overrides: object) -> None:
    vector_dir = root / "vector_index"
    vector_dir.mkdir(parents=True, exist_ok=True)
    (vector_dir / "index.faiss").write_bytes(b"synthetic-faiss-index")

    metadata = {
        "embedding_model": "embed-test",
        "embedding_dim": 3,
        "vector_metric": "inner_product",
        "vector_normalized": True,
        "vector_count": 3,
        "is_partial_embedding_index": False,
    }
    metadata.update(metadata_overrides)
    (vector_dir / "index.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    _write_database(root / "metadata.db")


def _write_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                relative_path TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                title TEXT,
                section_path TEXT,
                article_no TEXT,
                article_range TEXT,
                paragraph_start INTEGER,
                paragraph_end INTEGER,
                vector_id INTEGER,
                embedding_status TEXT
            );
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                chunk_id UNINDEXED,
                title,
                section_path,
                article_no,
                chunk_text
            );
            """
        )
        connection.executemany(
            "INSERT INTO documents (doc_id, title, relative_path) VALUES (?, ?, ?)",
            [
                ("doc-1", "Synthetic document 1", "fixtures/doc-1.docx"),
                ("doc-2", "Synthetic document 2", "fixtures/doc-2.docx"),
            ],
        )
        chunks = [
            ("chunk-1", "doc-1", 0),
            ("chunk-2", "doc-1", 1),
            ("chunk-3", "doc-2", 2),
        ]
        connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, chunk_text, title, section_path, article_no,
                article_range, paragraph_start, paragraph_end, vector_id,
                embedding_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk_id,
                    doc_id,
                    f"Synthetic text for {chunk_id}",
                    "Synthetic title",
                    "Synthetic section",
                    "Article 1",
                    None,
                    1,
                    1,
                    vector_id,
                    "success",
                )
                for chunk_id, doc_id, vector_id in chunks
            ],
        )
        connection.executemany(
            "INSERT INTO chunk_fts VALUES (?, ?, ?, ?, ?)",
            [
                (chunk_id, "Synthetic title", "Synthetic section", "Article 1", "text")
                for chunk_id, _, _ in chunks
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _settings(root: Path, **overrides: object) -> Settings:
    values = {
        "RAG_KNOWLEDGE_BASE_DIR": root,
        "UPSTREAM_EMBEDDING_MODEL": "embed-test",
        "RAG_EMBEDDING_DIM": 3,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _write_metadata(root: Path, **overrides: object) -> None:
    path = root / "vector_index" / "index.meta.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(overrides)
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_loads_valid_assets_once_and_opens_only_readonly_connections(
    knowledge_base_dir: Path,
) -> None:
    """A consistent delivery asset becomes ready and never needs a write connection."""
    loader = FakeFaissLoader()
    knowledge_base = KnowledgeBase(_settings(knowledge_base_dir), faiss_loader=loader)

    knowledge_base.load()
    knowledge_base.load()

    assert knowledge_base.is_ready is True
    assert knowledge_base.index is loader.index
    assert knowledge_base.index_metadata.embedding_dim == 3
    assert knowledge_base.index_metadata.vector_normalized is True
    assert loader.paths == [knowledge_base_dir / "vector_index" / "index.faiss"]

    with knowledge_base.open_readonly_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE forbidden_write (id INTEGER)")


@pytest.mark.parametrize(
    "relative_path",
    ["metadata.db", "vector_index/index.faiss", "vector_index/index.meta.json"],
)
def test_rejects_missing_required_asset_files(
    knowledge_base_dir: Path,
    relative_path: str,
) -> None:
    """Delivery is blocked when any core asset is absent."""
    (knowledge_base_dir / relative_path).unlink()
    loader = FakeFaissLoader()
    knowledge_base = KnowledgeBase(_settings(knowledge_base_dir), faiss_loader=loader)

    with pytest.raises(KnowledgeBaseLoadError) as error:
        knowledge_base.load()

    assert str(knowledge_base_dir) not in str(error.value)
    assert knowledge_base.is_ready is False
    assert loader.paths == []


def test_rejects_invalid_index_metadata_json(knowledge_base_dir: Path) -> None:
    """The metadata file must be one valid JSON object before SQLite is touched."""
    (knowledge_base_dir / "vector_index" / "index.meta.json").write_text(
        "not-json",
        encoding="utf-8",
    )
    loader = FakeFaissLoader()
    knowledge_base = KnowledgeBase(_settings(knowledge_base_dir), faiss_loader=loader)

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()

    assert knowledge_base.is_ready is False
    assert loader.paths == []


def test_rejects_index_metadata_with_missing_required_field(
    knowledge_base_dir: Path,
) -> None:
    """All fields needed to align the query model and index must be present."""
    path = knowledge_base_dir / "vector_index" / "index.meta.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["vector_count"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    knowledge_base = KnowledgeBase(
        _settings(knowledge_base_dir),
        faiss_loader=FakeFaissLoader(),
    )

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()

    assert knowledge_base.is_ready is False


@pytest.mark.parametrize(
    "metadata_overrides",
    [
        {"embedding_model": "other-model"},
        {"embedding_dim": 4},
        {"vector_metric": "l2"},
        {"vector_normalized": False},
        {"vector_normalized": "true"},
        {"vector_count": 0},
        {"is_partial_embedding_index": "false"},
    ],
)
def test_rejects_index_metadata_that_violates_the_contract(
    knowledge_base_dir: Path,
    metadata_overrides: dict[str, object],
) -> None:
    """Model, dimensions, metric, count, and flags must match a supported index."""
    _write_metadata(knowledge_base_dir, **metadata_overrides)
    knowledge_base = KnowledgeBase(
        _settings(knowledge_base_dir),
        faiss_loader=FakeFaissLoader(),
    )

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()

    assert knowledge_base.is_ready is False


def test_rejects_partial_index_by_default_and_allows_explicit_development_opt_in(
    knowledge_base_dir: Path,
) -> None:
    """Partial embedding output cannot silently become a customer delivery asset."""
    _write_metadata(knowledge_base_dir, is_partial_embedding_index=True)

    with pytest.raises(KnowledgeBaseLoadError):
        KnowledgeBase(
            _settings(knowledge_base_dir),
            faiss_loader=FakeFaissLoader(),
        ).load()

    loader = FakeFaissLoader()
    knowledge_base = KnowledgeBase(
        _settings(knowledge_base_dir, RAG_ALLOW_PARTIAL_INDEX=True),
        faiss_loader=loader,
    )
    knowledge_base.load()

    assert knowledge_base.is_ready is True
    assert knowledge_base.index_metadata.is_partial_embedding_index is True
    assert loader.paths


def test_rejects_missing_required_sqlite_table(knowledge_base_dir: Path) -> None:
    """A database with no document metadata cannot safely return traceable chunks."""
    connection = sqlite3.connect(knowledge_base_dir / "metadata.db")
    try:
        connection.execute("DROP TABLE documents")
        connection.commit()
    finally:
        connection.close()

    knowledge_base = KnowledgeBase(
        _settings(knowledge_base_dir),
        faiss_loader=FakeFaissLoader(),
    )

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()


def test_rejects_missing_required_sqlite_column(knowledge_base_dir: Path) -> None:
    """A missing source-path field prevents later source attribution."""
    connection = sqlite3.connect(knowledge_base_dir / "metadata.db")
    try:
        connection.execute("DROP TABLE documents")
        connection.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT)")
        connection.commit()
    finally:
        connection.close()

    knowledge_base = KnowledgeBase(
        _settings(knowledge_base_dir),
        faiss_loader=FakeFaissLoader(),
    )

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()


def test_quick_check_requires_exactly_ok() -> None:
    """Any SQLite quick-check diagnostic must stop delivery loading."""
    class Cursor:
        def fetchall(self):
            return [("database disk image is malformed",)]

    class Connection:
        def execute(self, statement: str):
            assert statement == "PRAGMA quick_check"
            return Cursor()

    with pytest.raises(KnowledgeBaseLoadError):
        KnowledgeBase._validate_quick_check(Connection())  # type: ignore[arg-type]


def test_rejects_duplicate_sqlite_vector_ids(knowledge_base_dir: Path) -> None:
    """A duplicate vector ID makes FAISS hits ambiguous and must block loading."""
    connection = sqlite3.connect(knowledge_base_dir / "metadata.db")
    try:
        connection.execute("UPDATE chunks SET vector_id = 0 WHERE chunk_id = 'chunk-2'")
        connection.commit()
    finally:
        connection.close()

    knowledge_base = KnowledgeBase(
        _settings(knowledge_base_dir),
        faiss_loader=FakeFaissLoader(),
    )

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()


def test_rejects_sqlite_and_metadata_vector_count_mismatch(
    knowledge_base_dir: Path,
) -> None:
    """A stale index metadata count cannot be combined with a newer SQLite file."""
    _write_metadata(knowledge_base_dir, vector_count=2)
    knowledge_base = KnowledgeBase(
        _settings(knowledge_base_dir),
        faiss_loader=FakeFaissLoader(),
    )

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()


@pytest.mark.parametrize(
    "index",
    [FakeIndex(dimension=2), FakeIndex(vector_count=2)],
)
def test_rejects_faiss_dimension_or_vector_count_mismatch(
    knowledge_base_dir: Path,
    index: FakeIndex,
) -> None:
    """FAISS dimensions and totals must agree with the metadata and SQLite mapping."""
    knowledge_base = KnowledgeBase(
        _settings(knowledge_base_dir),
        faiss_loader=FakeFaissLoader(index),
    )

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()

    assert knowledge_base.is_ready is False


def test_failed_load_leaves_no_half_ready_state_and_can_be_retried(
    knowledge_base_dir: Path,
) -> None:
    """A repaired asset can be loaded by the same instance after an earlier failure."""
    metadata_path = knowledge_base_dir / "vector_index" / "index.meta.json"
    metadata_path.unlink()
    loader = FakeFaissLoader()
    knowledge_base = KnowledgeBase(_settings(knowledge_base_dir), faiss_loader=loader)

    with pytest.raises(KnowledgeBaseLoadError):
        knowledge_base.load()

    assert knowledge_base.is_ready is False
    with pytest.raises(RuntimeError, match="not ready"):
        _ = knowledge_base.index

    metadata_path.write_text(
        json.dumps(
            {
                "embedding_model": "embed-test",
                "embedding_dim": 3,
                "vector_metric": "inner_product",
                "vector_normalized": True,
                "vector_count": 3,
                "is_partial_embedding_index": False,
            }
        ),
        encoding="utf-8",
    )
    knowledge_base.load()

    assert knowledge_base.is_ready is True
    assert len(loader.paths) == 1


def _loaded_vector_knowledge_base(
    root: Path,
    index: SearchableFakeIndex,
) -> KnowledgeBase:
    """Load a valid synthetic knowledge base using a deterministic fake index."""
    knowledge_base = KnowledgeBase(
        _settings(root),
        faiss_loader=FakeFaissLoader(index),
    )
    knowledge_base.load()
    return knowledge_base


def test_prepare_query_vector_normalizes_to_contiguous_float32(
    knowledge_base_dir: Path,
) -> None:
    """A stage-2 list embedding becomes the exact one-row matrix expected by FAISS."""
    index = SearchableFakeIndex([[0.9]], [[0]])
    knowledge_base = _loaded_vector_knowledge_base(knowledge_base_dir, index)

    prepared = knowledge_base.prepare_query_vector([3.0, 4.0, 0.0])

    assert prepared.dtype == np.float32
    assert prepared.shape == (1, 3)
    assert prepared.flags.c_contiguous is True
    assert np.linalg.norm(prepared) == pytest.approx(1.0)
    np.testing.assert_allclose(prepared, np.array([[0.6, 0.8, 0.0]], dtype=np.float32))


@pytest.mark.parametrize(
    "query_vector",
    [
        [1.0, 2.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [0.0, 0.0, 0.0],
        [float("nan"), 0.0, 1.0],
        ["not-a-number", 0.0, 1.0],
    ],
)
def test_prepare_query_vector_rejects_invalid_inputs(
    knowledge_base_dir: Path,
    query_vector: object,
) -> None:
    """Invalid query embeddings cannot reach FAISS or leak their contents in errors."""
    index = SearchableFakeIndex([[0.9]], [[0]])
    knowledge_base = _loaded_vector_knowledge_base(knowledge_base_dir, index)

    with pytest.raises(KnowledgeBaseQueryError) as error:
        knowledge_base.prepare_query_vector(query_vector)

    assert "not-a-number" not in str(error.value)


def test_vector_search_skips_padding_and_restores_faiss_order_with_one_sql_query(
    knowledge_base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padding IDs are ignored while SQLite rows are batch-loaded then rank-restored."""
    index = SearchableFakeIndex(
        [[0.91, 0.42, -np.inf, -np.inf]],
        [[2, 0, -1, -1]],
    )
    knowledge_base = _loaded_vector_knowledge_base(knowledge_base_dir, index)
    original_open = knowledge_base.open_readonly_connection
    connection_count = 0

    @contextmanager
    def counting_connection():
        nonlocal connection_count
        connection_count += 1
        with original_open() as connection:
            yield connection

    monkeypatch.setattr(knowledge_base, "open_readonly_connection", counting_connection)
    database_path = knowledge_base_dir / "metadata.db"
    before = database_path.read_bytes()
    hits = knowledge_base.search_vector([1.0, 0.0, 0.0], top_k=10)

    assert connection_count == 1
    assert [hit.vector_id for hit in hits] == [2, 0]
    assert [hit.vector_rank for hit in hits] == [1, 2]
    assert [hit.vector_score for hit in hits] == pytest.approx([0.91, 0.42])
    assert [hit.chunk_id for hit in hits] == ["chunk-3", "chunk-1"]
    assert index.search_calls[0][0].dtype == np.float32
    assert index.search_calls[0][0].shape == (1, 3)
    assert index.search_calls[0][1] == 10
    assert database_path.read_bytes() == before


def test_vector_search_returns_full_chunk_and_source_metadata(
    knowledge_base_dir: Path,
) -> None:
    """The result preserves data required by later context and citation stages."""
    index = SearchableFakeIndex([[0.75]], [[1]])
    knowledge_base = _loaded_vector_knowledge_base(knowledge_base_dir, index)

    hit = knowledge_base.search_vector([1.0, 0.0, 0.0], top_k=1)[0]

    assert hit.chunk_id == "chunk-2"
    assert hit.doc_id == "doc-1"
    assert hit.chunk_text == "Synthetic text for chunk-2"
    assert hit.title == "Synthetic title"
    assert hit.doc_title == "Synthetic document 1"
    assert hit.section_path == "Synthetic section"
    assert hit.article_no == "Article 1"
    assert hit.article_range is None
    assert hit.relative_path == "fixtures/doc-1.docx"
    assert hit.paragraph_start == 1
    assert hit.paragraph_end == 1


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_vector_search_requires_positive_integer_top_k(
    knowledge_base_dir: Path,
    top_k: int | bool,
) -> None:
    """Invalid search limits are rejected before the index is called."""
    index = SearchableFakeIndex([[0.9]], [[0]])
    knowledge_base = _loaded_vector_knowledge_base(knowledge_base_dir, index)

    with pytest.raises(KnowledgeBaseQueryError):
        knowledge_base.search_vector([1.0, 0.0, 0.0], top_k=top_k)

    assert index.search_calls == []


@pytest.mark.parametrize(
    "scores, vector_ids",
    [
        ([[0.9, 0.8]], [[0, 0]]),
        ([[float("nan")]], [[0]]),
        ([[0.9]], [[99]]),
    ],
)
def test_vector_search_rejects_duplicate_invalid_or_unmapped_hits(
    knowledge_base_dir: Path,
    scores: object,
    vector_ids: object,
) -> None:
    """FAISS output must remain finite, unique, and traceable to one SQLite chunk."""
    index = SearchableFakeIndex(scores, vector_ids)
    knowledge_base = _loaded_vector_knowledge_base(knowledge_base_dir, index)

    with pytest.raises(KnowledgeBaseQueryError):
        knowledge_base.search_vector([1.0, 0.0, 0.0], top_k=3)


def _loaded_fts_knowledge_base(root: Path, **settings_overrides: object) -> KnowledgeBase:
    """Load a valid synthetic knowledge base for SQLite-only FTS tests."""
    knowledge_base = KnowledgeBase(
        _settings(root, **settings_overrides),
        faiss_loader=FakeFaissLoader(),
    )
    knowledge_base.load()
    return knowledge_base


def test_build_fts_query_normalizes_chinese_terms_and_article_numbers() -> None:
    """Question wrappers are removed and structured terms become safe prefixes."""
    query = build_fts_query('广东省“安全生产条例”第十条是什么？')

    assert query is not None
    assert '"广东省"*' in query
    assert '"安全生产条例"*' in query
    assert '"第十条"*' in query
    assert "是什么" not in query
    assert " AND " in query


def test_build_fts_query_plan_rewrites_chinese_natural_language_question() -> None:
    """A location/topic question must not be sent to FTS as one full Chinese phrase."""
    plan = build_fts_query_plan("河北保定的城市更新规划是什么？")

    assert plan.strict_query == (
        '("河北省"* OR "保定市"*) AND '
        '("城市更新规划"* OR "城市更新"*)'
    )
    assert plan.relaxed_query == (
        '("河北省"* OR "保定市"* OR "城市更新规划"* OR "城市更新"*)'
    )
    assert plan.term_count == 4
    assert "河北保定的城市更新规划是什么" not in plan.strict_query


def test_build_fts_query_plan_does_not_expose_user_fts_syntax() -> None:
    """User-provided FTS operators cannot escape the deterministic term builder."""
    plan = build_fts_query_plan('" OR chunk_fts MATCH "* 保定市')

    assert plan.strict_query is not None
    assert "MATCH" not in plan.strict_query.upper()
    assert "*" in plan.strict_query  # Only code-generated legacy prefix syntax remains.
    assert '"保定市"*' in plan.strict_query


@pytest.mark.parametrize("text", ["   ", "()[]{}:*\t\n\x00", "——？！"])
def test_build_fts_query_skips_text_without_searchable_terms(text: str) -> None:
    """Whitespace, punctuation, and controls cannot become an FTS syntax expression."""
    assert build_fts_query(text) is None


def test_build_fts_query_bounds_long_or_repeated_input() -> None:
    """An oversized user input cannot create an unbounded FTS expression."""
    text = "甲" * 100 + " " + " ".join(f"term{number}" for number in range(30))
    query = build_fts_query(text)

    assert query is not None
    assert query.count('"') // 2 <= 16
    assert f'"{"甲" * 64}"*' in query
    assert len(query) <= 1024


def test_fts_search_uses_relaxed_query_only_after_strict_query_has_no_hits(
    knowledge_base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback broadens recall without paying for a second MATCH on strict hits."""
    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)
    captured_parameters: list[tuple[object, ...]] = []

    class CapturingConnection:
        def execute(self, _: str, parameters: tuple[object, ...]):
            captured_parameters.append(parameters)
            return self

        def fetchall(self) -> list[dict[str, object]]:
            if len(captured_parameters) == 1:
                return []
            return [{"chunk_id": "chunk-1", "bm25_score": -0.5}]

    @contextmanager
    def fake_connection():
        yield CapturingConnection()

    monkeypatch.setattr(knowledge_base, "open_readonly_connection", fake_connection)

    candidates = knowledge_base.search_fts("河北保定的城市更新规划是什么？", top_k=7)

    plan = build_fts_query_plan("河北保定的城市更新规划是什么？")
    assert [candidate.chunk_id for candidate in candidates] == ["chunk-1"]
    assert captured_parameters == [
        (plan.strict_query, 7),
        (plan.relaxed_query, 7),
    ]


def test_fts_search_matches_rewritten_natural_language_query_on_legacy_table(
    knowledge_base_dir: Path,
) -> None:
    """Legacy prefix matching solves the diagnosed city/topic query without rebuilding FTS."""
    connection = sqlite3.connect(knowledge_base_dir / "metadata.db")
    try:
        connection.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", ("chunk-1",))
        connection.execute(
            "INSERT INTO chunk_fts VALUES (?, ?, ?, ?, ?)",
            (
                "chunk-1",
                "保定市城市更新条例",
                "第二章 城市更新规划和计划",
                "第十条",
                "城市更新规划应当纳入相关规划。",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)

    candidates = knowledge_base.search_fts("河北保定的城市更新规划是什么？", top_k=5)

    assert [candidate.chunk_id for candidate in candidates] == ["chunk-1"]


def test_fts_search_returns_stable_bm25_candidates_without_writing_sqlite(
    knowledge_base_dir: Path,
) -> None:
    """FTS hits are ordered by SQLite BM25 then chunk ID and carry 1-based ranks."""
    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)
    database_path = knowledge_base_dir / "metadata.db"
    before = database_path.read_bytes()

    candidates = knowledge_base.search_fts("text", top_k=10)

    assert [candidate.chunk_id for candidate in candidates] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert [candidate.fts_rank for candidate in candidates] == [1, 2, 3]
    assert [candidate.bm25_score for candidate in candidates] == sorted(
        candidate.bm25_score for candidate in candidates
    )
    assert database_path.read_bytes() == before


def test_fts_search_uses_parameterized_match_query(
    knowledge_base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw user text is never concatenated into the SQL statement."""
    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)
    raw_text = '" OR chunk_fts MATCH "*'
    captured: dict[str, object] = {}

    class CapturingConnection:
        def execute(self, sql: str, parameters: tuple[object, ...]):
            captured["sql"] = sql
            captured["parameters"] = parameters
            return self

        def fetchall(self) -> list[dict[str, object]]:
            return [{"chunk_id": "chunk-1", "bm25_score": -0.5}]

    @contextmanager
    def fake_connection():
        yield CapturingConnection()

    monkeypatch.setattr(knowledge_base, "open_readonly_connection", fake_connection)

    candidates = knowledge_base.search_fts(raw_text, top_k=7)

    assert candidates[0].chunk_id == "chunk-1"
    assert "MATCH ?" in str(captured["sql"])
    assert raw_text not in str(captured["sql"])
    assert captured["parameters"] == (build_fts_query(raw_text), 7)


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_fts_search_requires_positive_integer_top_k(
    knowledge_base_dir: Path,
    top_k: int | bool,
) -> None:
    """Invalid limits are rejected before any SQLite FTS request is attempted."""
    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)

    with pytest.raises(KnowledgeBaseQueryError):
        knowledge_base.search_fts("text", top_k=top_k)


def test_fts_search_skips_empty_queries_and_disabled_fts(
    knowledge_base_dir: Path,
) -> None:
    """Normal non-searchable input and disabled optional FTS both return no candidates."""
    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)
    disabled_knowledge_base = _loaded_fts_knowledge_base(
        knowledge_base_dir,
        RAG_ENABLE_FTS=False,
    )

    assert knowledge_base.search_fts("！？", top_k=3) == []
    assert disabled_knowledge_base.search_fts("text", top_k=3) == []


def test_fts_search_distinguishes_recoverable_fts_failure_from_asset_failure(
    knowledge_base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later orchestration can downgrade FTS-only errors but not damaged KB errors."""
    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)

    class FailingConnection:
        def __init__(self, error: sqlite3.DatabaseError) -> None:
            self.error = error

        def execute(self, *_: object) -> None:
            raise self.error

    @contextmanager
    def fts_missing_connection():
        yield FailingConnection(sqlite3.OperationalError("no such module: fts5"))

    monkeypatch.setattr(
        knowledge_base,
        "open_readonly_connection",
        fts_missing_connection,
    )
    with pytest.raises(FtsSearchFallbackError):
        knowledge_base.search_fts("text", top_k=3)

    @contextmanager
    def damaged_asset_connection():
        yield FailingConnection(sqlite3.DatabaseError("database disk image is malformed"))

    monkeypatch.setattr(
        knowledge_base,
        "open_readonly_connection",
        damaged_asset_connection,
    )
    with pytest.raises(KnowledgeBaseQueryError):
        knowledge_base.search_fts("text", top_k=3)


def test_load_chunk_metadata_batch_loads_complete_fts_only_source_records(
    knowledge_base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-7 can resolve all fused chunk IDs in one read-only SQLite query."""
    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)
    original_open = knowledge_base.open_readonly_connection
    connection_count = 0

    @contextmanager
    def counting_connection():
        nonlocal connection_count
        connection_count += 1
        with original_open() as connection:
            yield connection

    monkeypatch.setattr(knowledge_base, "open_readonly_connection", counting_connection)
    database_path = knowledge_base_dir / "metadata.db"
    before = database_path.read_bytes()

    metadata = knowledge_base.load_chunk_metadata(["chunk-2", "chunk-1"])

    assert connection_count == 1
    assert set(metadata) == {"chunk-1", "chunk-2"}
    assert metadata["chunk-2"].doc_id == "doc-1"
    assert metadata["chunk-2"].chunk_text == "Synthetic text for chunk-2"
    assert metadata["chunk-2"].relative_path == "fixtures/doc-1.docx"
    assert metadata["chunk-2"].vector_id == 1
    assert database_path.read_bytes() == before


@pytest.mark.parametrize("chunk_ids", [["chunk-1", "chunk-1"], ["missing-chunk"]])
def test_load_chunk_metadata_rejects_duplicate_or_unmapped_chunk_ids(
    knowledge_base_dir: Path,
    chunk_ids: list[str],
) -> None:
    """Every selected fused candidate must be uniquely and completely traceable."""
    knowledge_base = _loaded_fts_knowledge_base(knowledge_base_dir)

    with pytest.raises(KnowledgeBaseQueryError):
        knowledge_base.load_chunk_metadata(chunk_ids)
