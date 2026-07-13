"""Read-only loading and consistency validation for delivered RAG assets."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from numbers import Integral
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterable, Iterator, Sequence
import unicodedata

import numpy as np

from local_rag_app.config import Settings
from rag_preprocess.faiss_builder import load_faiss_index


class KnowledgeBaseLoadError(RuntimeError):
    """A client-safe failure to load a delivered local knowledge base."""

    def __init__(self) -> None:
        super().__init__("The local knowledge base is unavailable")


class KnowledgeBaseQueryError(RuntimeError):
    """A client-safe failure while searching an already loaded knowledge base."""

    def __init__(self) -> None:
        super().__init__("Local knowledge retrieval failed")


class FtsSearchFallbackError(RuntimeError):
    """A non-fatal FTS5 failure that later retrieval orchestration may downgrade."""

    def __init__(self) -> None:
        super().__init__("Local keyword retrieval is unavailable")


@dataclass(frozen=True)
class IndexMetadata:
    """Validated fields required to query one FAISS knowledge-base index."""

    embedding_model: str
    embedding_dim: int
    vector_metric: str
    vector_normalized: bool
    vector_count: int
    is_partial_embedding_index: bool


@dataclass(frozen=True)
class VectorSearchCandidate:
    """One raw FAISS hit before its SQLite chunk metadata is loaded."""

    vector_id: int
    vector_score: float
    vector_rank: int


@dataclass(frozen=True)
class FtsSearchCandidate:
    """One ranked SQLite FTS5 hit before vector/keyword fusion."""

    chunk_id: str
    bm25_score: float
    fts_rank: int


@dataclass(frozen=True)
class FtsQueryPlan:
    """Bounded legacy-FTS expressions derived from one natural-language query."""

    strict_query: str | None
    relaxed_query: str | None
    term_count: int


@dataclass(frozen=True)
class VectorSearchHit:
    """One FAISS hit with its full, traceable local source metadata."""

    chunk_id: str
    doc_id: str
    chunk_text: str
    title: str | None
    doc_title: str | None
    section_path: str | None
    article_no: str | None
    article_range: str | None
    relative_path: str
    paragraph_start: int | None
    paragraph_end: int | None
    vector_id: int
    vector_score: float
    vector_rank: int


@dataclass(frozen=True)
class ChunkMetadata:
    """One chunk's traceable metadata, independent of its retrieval route."""

    chunk_id: str
    doc_id: str
    chunk_text: str
    title: str | None
    doc_title: str | None
    section_path: str | None
    article_no: str | None
    article_range: str | None
    relative_path: str
    paragraph_start: int | None
    paragraph_end: int | None
    vector_id: int | None


_REQUIRED_TABLE_COLUMNS = {
    "documents": {"doc_id", "title", "relative_path"},
    "chunks": {
        "chunk_id",
        "doc_id",
        "chunk_text",
        "title",
        "section_path",
        "article_no",
        "article_range",
        "paragraph_start",
        "paragraph_end",
        "vector_id",
        "embedding_status",
    },
    "chunk_fts": {
        "chunk_id",
        "title",
        "section_path",
        "article_no",
        "chunk_text",
    },
}

_FTS_TOKEN_PATTERN = re.compile(
    r"第[〇一二三四五六七八九十百千万零两0-9]+条|"
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[A-Za-z0-9]+"
)
_FTS_ARTICLE_PATTERN = re.compile(r"第[〇一二三四五六七八九十百千万零两0-9]+条")
_FTS_CJK_RUN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_FTS_ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_FTS_PUNCTUATION_PATTERN = re.compile(r"[\"'“”‘’《》〈〉【】()（）\[\]{}:：;；,，。！？!?、/\\|]+")
_FTS_SEMANTIC_SEPARATOR_PATTERN = re.compile(r"的|关于|如何|怎么|哪些|什么|是否|有无")
_FTS_QUESTION_PREFIXES = (
    "麻烦问一下",
    "帮我查一下",
    "我想了解",
    "请问",
)
_FTS_QUESTION_SUFFIXES = (
    "是怎么规定的",
    "有哪些规定",
    "请介绍一下",
    "如何规定",
    "怎么规定",
    "是什么",
    "有哪些",
    "吗",
    "呢",
)
_FTS_DOMAIN_SUFFIXES = (
    "条例",
    "办法",
    "规定",
    "规划",
    "计划",
    "标准",
    "细则",
    "通知",
    "意见",
    "决定",
    "措施",
    "制度",
    "责任",
    "处罚",
    "许可",
    "保护",
)
_FTS_RESERVED_WORDS = frozenset({"and", "or", "not", "near", "match"})
_FTS_PROVINCE_ALIASES = (
    ("内蒙古", "内蒙古自治区"),
    ("广西", "广西壮族自治区"),
    ("宁夏", "宁夏回族自治区"),
    ("新疆", "新疆维吾尔自治区"),
    ("西藏", "西藏自治区"),
    ("香港", "香港特别行政区"),
    ("澳门", "澳门特别行政区"),
    ("北京", "北京市"),
    ("天津", "天津市"),
    ("河北", "河北省"),
    ("山西", "山西省"),
    ("辽宁", "辽宁省"),
    ("吉林", "吉林省"),
    ("黑龙江", "黑龙江省"),
    ("上海", "上海市"),
    ("江苏", "江苏省"),
    ("浙江", "浙江省"),
    ("安徽", "安徽省"),
    ("福建", "福建省"),
    ("江西", "江西省"),
    ("山东", "山东省"),
    ("河南", "河南省"),
    ("湖北", "湖北省"),
    ("湖南", "湖南省"),
    ("广东", "广东省"),
    ("海南", "海南省"),
    ("重庆", "重庆市"),
    ("四川", "四川省"),
    ("贵州", "贵州省"),
    ("云南", "云南省"),
    ("陕西", "陕西省"),
    ("甘肃", "甘肃省"),
    ("青海", "青海省"),
    ("台湾", "台湾省"),
)
_FTS_EXPLICIT_LOCATION_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,12}?"
    r"(?:特别行政区|自治区|自治州|省|市|地区|盟|县|区)"
)
_MAX_FTS_TOKENS = 16
_MAX_FTS_TOKEN_LENGTH = 64
_MAX_FTS_MATCH_LENGTH = 1024


def build_fts_query(text: str) -> str | None:
    """Return the preferred safe legacy-FTS MATCH expression for one query."""
    return build_fts_query_plan(text).strict_query


def build_fts_query_plan(text: str) -> FtsQueryPlan:
    """Build bounded strict/relaxed legacy FTS5 expressions from a user question.

    The planner deliberately uses only deterministic local rules.  It never passes
    user FTS syntax through to SQLite: terms are extracted from CJK/alphanumeric
    runs and the prefix-query syntax is added by this module only.
    """
    if not isinstance(text, str):
        return FtsQueryPlan(None, None, 0)
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _CONTROL_CHARACTER_PATTERN.sub(" ", normalized)
    normalized = _FTS_PUNCTUATION_PATTERN.sub(" ", normalized)
    normalized = _strip_question_shell(normalized)
    normalized = " ".join(normalized.split())
    if not normalized:
        return FtsQueryPlan(None, None, 0)

    article_terms = _unique_terms(_FTS_ARTICLE_PATTERN.findall(normalized))
    semantic_parts = _semantic_parts(normalized)
    location_terms = _extract_location_terms(semantic_parts)
    subject_terms = _extract_subject_terms(
        semantic_parts,
        location_terms=location_terms,
        article_terms=article_terms,
    )
    identifier_terms = _extract_identifier_terms(normalized)

    groups = _bound_fts_groups([
        location_terms,
        subject_terms,
        article_terms,
        identifier_terms,
    ])
    non_empty_groups = [group for group in groups if group]
    all_terms = [term for group in non_empty_groups for term in group]
    if not all_terms:
        return FtsQueryPlan(None, None, 0)

    strict_query = _join_fts_groups(non_empty_groups, outer_operator="AND")
    relaxed_query = _join_fts_groups([all_terms], outer_operator="AND")
    if relaxed_query == strict_query:
        relaxed_query = None
    return FtsQueryPlan(
        strict_query=strict_query,
        relaxed_query=relaxed_query,
        term_count=len(all_terms),
    )


def _strip_question_shell(text: str) -> str:
    """Remove only known question wrappers at the start/end of a normalized query."""
    normalized = text.strip()
    for prefix in _FTS_QUESTION_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].lstrip()
            break
    changed = True
    while changed and normalized:
        changed = False
        for suffix in _FTS_QUESTION_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].rstrip()
                changed = True
                break
    return normalized


def _semantic_parts(text: str) -> list[str]:
    """Split one cleaned question into small deterministic semantic fragments."""
    return [part.strip() for part in _FTS_SEMANTIC_SEPARATOR_PATTERN.split(text) if part.strip()]


def _extract_location_terms(parts: Sequence[str]) -> list[str]:
    """Extract explicit locations and conservative province/city expansions."""
    terms: list[str] = []
    for part in parts:
        for location in _FTS_EXPLICIT_LOCATION_PATTERN.findall(part):
            _append_fts_term(terms, location)

    scope = parts[0] if len(parts) >= 2 else ""
    if scope:
        for alias, canonical in _FTS_PROVINCE_ALIASES:
            if scope.startswith(alias):
                _append_fts_term(terms, canonical)
                remainder = scope[len(alias):]
                if _is_bare_location_candidate(remainder):
                    _append_fts_term(terms, f"{remainder}市")
                break
        else:
            if _is_bare_location_candidate(scope):
                _append_fts_term(terms, f"{scope}市")
    return terms


def _is_bare_location_candidate(text: str) -> bool:
    return bool(_FTS_CJK_RUN_PATTERN.fullmatch(text)) and 2 <= len(text) <= 4


def _extract_subject_terms(
    parts: Sequence[str],
    *,
    location_terms: Sequence[str],
    article_terms: Sequence[str],
) -> list[str]:
    """Extract topic/title phrases while avoiding the location-only scope fragment."""
    sources = parts[1:] if len(parts) >= 2 else parts
    terms: list[str] = []
    for source in sources:
        for run in _FTS_CJK_RUN_PATTERN.findall(source):
            run = _remove_known_prefixes(run, location_terms)
            for article in article_terms:
                run = run.replace(article, "")
            _append_subject_variants(terms, run)
    return terms


def _remove_known_prefixes(text: str, prefixes: Sequence[str]) -> str:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _append_subject_variants(terms: list[str], text: str) -> None:
    _append_fts_term(terms, text)
    for suffix in _FTS_DOMAIN_SUFFIXES:
        start = 0
        while True:
            index = text.find(suffix, start)
            if index < 0:
                break
            phrase = text[:index + len(suffix)]
            _append_fts_term(terms, phrase)
            if len(phrase) > len(suffix):
                _append_fts_term(terms, phrase[:-len(suffix)])
            start = index + len(suffix)


def _extract_identifier_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in _FTS_ASCII_TOKEN_PATTERN.findall(text):
        if token.casefold() not in _FTS_RESERVED_WORDS:
            _append_fts_term(terms, token)
    return terms


def _append_fts_term(terms: list[str], candidate: str) -> None:
    token = candidate[:_MAX_FTS_TOKEN_LENGTH]
    if not token or token in terms:
        return
    if not _FTS_TOKEN_PATTERN.fullmatch(token):
        return
    if token.casefold() in _FTS_RESERVED_WORDS:
        return
    if len(terms) >= _MAX_FTS_TOKENS:
        return
    terms.append(token)


def _unique_terms(candidates: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for candidate in candidates:
        _append_fts_term(terms, candidate)
        if len(terms) == _MAX_FTS_TOKENS:
            break
    return terms


def _bound_fts_groups(groups: Sequence[Sequence[str]]) -> list[list[str]]:
    """Keep priority order while enforcing one global term budget across groups."""
    bounded: list[list[str]] = []
    seen_terms: set[str] = set()
    for group in groups:
        accepted: list[str] = []
        for term in group:
            if term in seen_terms or len(seen_terms) >= _MAX_FTS_TOKENS:
                continue
            accepted.append(term)
            seen_terms.add(term)
        if accepted:
            bounded.append(accepted)
    return bounded


def _join_fts_groups(groups: Sequence[Sequence[str]], *, outer_operator: str) -> str | None:
    rendered_groups: list[str] = []
    for group in groups:
        rendered_terms = [f'"{term}"*' for term in group]
        if not rendered_terms:
            continue
        rendered_groups.append(
            rendered_terms[0]
            if len(rendered_terms) == 1
            else f"({' OR '.join(rendered_terms)})"
        )
    if not rendered_groups:
        return None
    expression = f" {outer_operator} ".join(rendered_groups)
    if len(expression) > _MAX_FTS_MATCH_LENGTH:
        return None
    return expression


class KnowledgeBase:
    """Load one immutable knowledge base only after cross-asset validation."""

    def __init__(
        self,
        settings: Settings,
        *,
        faiss_loader: Callable[[Path], Any] = load_faiss_index,
    ) -> None:
        self._settings = settings
        self._faiss_loader = faiss_loader
        self._metadata_db_path = settings.rag_knowledge_base_dir / "metadata.db"
        self._index_path = settings.rag_knowledge_base_dir / "vector_index" / "index.faiss"
        self._index_meta_path = (
            settings.rag_knowledge_base_dir / "vector_index" / "index.meta.json"
        )
        self._index: Any | None = None
        self._index_metadata: IndexMetadata | None = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        """Whether all assets have been loaded and validated successfully."""
        return self._ready

    @property
    def index(self) -> Any:
        """Return the validated FAISS index after ``load`` has succeeded."""
        if not self._ready or self._index is None:
            raise RuntimeError("Knowledge base is not ready")
        return self._index

    @property
    def index_metadata(self) -> IndexMetadata:
        """Return the validated index metadata after ``load`` has succeeded."""
        if not self._ready or self._index_metadata is None:
            raise RuntimeError("Knowledge base is not ready")
        return self._index_metadata

    def load(self) -> None:
        """Load and validate assets atomically; repeated successful calls are no-ops."""
        if self._ready:
            return

        try:
            self._validate_asset_files()
            metadata = self._load_index_metadata()
            self._validate_sqlite(metadata)
            index = self._faiss_loader(self._index_path)
            self._validate_faiss_index(index, metadata)
        except KnowledgeBaseLoadError:
            self.close()
            raise
        except (
            OSError,
            sqlite3.Error,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            self.close()
            raise KnowledgeBaseLoadError() from exc

        self._index = index
        self._index_metadata = metadata
        self._ready = True

    def close(self) -> None:
        """Release in-process index references and mark this instance not ready."""
        self._index = None
        self._index_metadata = None
        self._ready = False

    @contextmanager
    def open_readonly_connection(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived SQLite connection that cannot mutate delivered assets."""
        if not self._ready:
            raise RuntimeError("Knowledge base is not ready")
        connection = self._connect_readonly()
        try:
            yield connection
        finally:
            connection.close()

    def prepare_query_vector(self, query_vector: Any) -> np.ndarray:
        """Return a contiguous normalized ``float32`` matrix ready for FAISS search."""
        metadata = self._require_ready_metadata()
        try:
            matrix = np.asarray(query_vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise KnowledgeBaseQueryError() from exc

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape != (1, metadata.embedding_dim):
            raise KnowledgeBaseQueryError()
        if not np.isfinite(matrix).all():
            raise KnowledgeBaseQueryError()

        norm = float(np.linalg.norm(matrix))
        if not np.isfinite(norm) or norm <= 0:
            raise KnowledgeBaseQueryError()
        if metadata.vector_normalized:
            matrix = matrix / norm
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def search_vector(
        self,
        query_vector: Any,
        top_k: int,
    ) -> list[VectorSearchHit]:
        """Search FAISS and batch-load matching chunks in the original rank order."""
        candidates = self.search_vector_candidates(query_vector, top_k)
        return self.load_vector_hits(candidates)

    def search_fts(self, query_text: str, top_k: int) -> list[FtsSearchCandidate]:
        """Return stable FTS5 candidates without accepting raw FTS syntax from users."""
        self._require_ready_metadata()
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
            raise KnowledgeBaseQueryError()
        if not self._settings.rag_enable_fts:
            return []

        query_plan = build_fts_query_plan(query_text)
        if query_plan.strict_query is None:
            return []

        try:
            with self.open_readonly_connection() as connection:
                rows: Sequence[sqlite3.Row] = []
                queries = [query_plan.strict_query]
                if query_plan.relaxed_query is not None:
                    queries.append(query_plan.relaxed_query)
                for fts_query in queries:
                    rows = connection.execute(
                        """
                        SELECT chunk_id, bm25(chunk_fts) AS bm25_score
                        FROM chunk_fts
                        WHERE chunk_fts MATCH ?
                        ORDER BY bm25_score ASC, chunk_id ASC
                        LIMIT ?
                        """,
                        (fts_query, int(top_k)),
                    ).fetchall()
                    if rows:
                        break
        except sqlite3.OperationalError as exc:
            if _is_recoverable_fts_error(exc):
                raise FtsSearchFallbackError() from exc
            raise KnowledgeBaseQueryError() from exc
        except sqlite3.DatabaseError as exc:
            raise KnowledgeBaseQueryError() from exc

        candidates: list[FtsSearchCandidate] = []
        seen_chunk_ids: set[str] = set()
        for row in rows:
            chunk_id = row["chunk_id"]
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen_chunk_ids:
                raise KnowledgeBaseQueryError()
            try:
                bm25_score = float(row["bm25_score"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise KnowledgeBaseQueryError() from exc
            if not np.isfinite(bm25_score):
                raise KnowledgeBaseQueryError()
            seen_chunk_ids.add(chunk_id)
            candidates.append(
                FtsSearchCandidate(
                    chunk_id=chunk_id,
                    bm25_score=bm25_score,
                    fts_rank=len(candidates) + 1,
                )
            )
        return candidates

    def search_vector_candidates(
        self,
        query_vector: Any,
        top_k: int,
    ) -> list[VectorSearchCandidate]:
        """Return validated FAISS vector IDs, scores, and ranks without SQLite lookups."""
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
            raise KnowledgeBaseQueryError()
        query_matrix = self.prepare_query_vector(query_vector)
        try:
            scores, vector_ids = self.index.search(query_matrix, int(top_k))
        except (TypeError, ValueError, RuntimeError) as exc:
            raise KnowledgeBaseQueryError() from exc

        score_matrix = np.asarray(scores)
        id_matrix = np.asarray(vector_ids)
        if (
            score_matrix.ndim != 2
            or id_matrix.ndim != 2
            or score_matrix.shape[0] != 1
            or score_matrix.shape != id_matrix.shape
        ):
            raise KnowledgeBaseQueryError()

        candidates: list[VectorSearchCandidate] = []
        seen_vector_ids: set[int] = set()
        for raw_score, raw_vector_id in zip(score_matrix[0], id_matrix[0]):
            if isinstance(raw_vector_id, bool) or not isinstance(raw_vector_id, Integral):
                raise KnowledgeBaseQueryError()
            vector_id = int(raw_vector_id)
            if vector_id < 0:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise KnowledgeBaseQueryError() from exc
            if not np.isfinite(score) or vector_id in seen_vector_ids:
                raise KnowledgeBaseQueryError()
            seen_vector_ids.add(vector_id)
            candidates.append(
                VectorSearchCandidate(
                    vector_id=vector_id,
                    vector_score=score,
                    vector_rank=len(candidates) + 1,
                )
            )
        return candidates

    def load_vector_hits(
        self,
        candidates: list[VectorSearchCandidate],
    ) -> list[VectorSearchHit]:
        """Batch-load complete chunk metadata and restore FAISS candidate order."""
        self._require_ready_metadata()
        if not candidates:
            return []

        vector_ids = [candidate.vector_id for candidate in candidates]
        if len(set(vector_ids)) != len(vector_ids):
            raise KnowledgeBaseQueryError()
        placeholders = ", ".join("?" for _ in vector_ids)
        query = f"""
            SELECT c.chunk_id,
                   c.doc_id,
                   c.chunk_text,
                   c.title,
                   c.section_path,
                   c.article_no,
                   c.article_range,
                   c.paragraph_start,
                   c.paragraph_end,
                   c.vector_id,
                   d.title AS doc_title,
                   d.relative_path
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.vector_id IN ({placeholders})
              AND c.embedding_status = 'success'
        """
        with self.open_readonly_connection() as connection:
            rows = connection.execute(query, vector_ids).fetchall()

        rows_by_vector_id: dict[int, sqlite3.Row] = {}
        for row in rows:
            value = row["vector_id"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise KnowledgeBaseQueryError()
            vector_id = int(value)
            if vector_id in rows_by_vector_id:
                raise KnowledgeBaseQueryError()
            rows_by_vector_id[vector_id] = row

        hits: list[VectorSearchHit] = []
        for candidate in candidates:
            row = rows_by_vector_id.get(candidate.vector_id)
            if row is None:
                raise KnowledgeBaseQueryError()
            hits.append(_vector_search_hit_from_row(row, candidate))
        return hits

    def load_chunk_metadata(
        self,
        chunk_ids: Sequence[str],
    ) -> dict[str, ChunkMetadata]:
        """Batch-load full source metadata for unique chunk IDs in one read query."""
        self._require_ready_metadata()
        if not chunk_ids:
            return {}
        if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in chunk_ids):
            raise KnowledgeBaseQueryError()
        if len(set(chunk_ids)) != len(chunk_ids):
            raise KnowledgeBaseQueryError()

        placeholders = ", ".join("?" for _ in chunk_ids)
        query = f"""
            SELECT c.chunk_id,
                   c.doc_id,
                   c.chunk_text,
                   c.title,
                   c.section_path,
                   c.article_no,
                   c.article_range,
                   c.paragraph_start,
                   c.paragraph_end,
                   c.vector_id,
                   d.title AS doc_title,
                   d.relative_path
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id IN ({placeholders})
        """
        with self.open_readonly_connection() as connection:
            rows = connection.execute(query, list(chunk_ids)).fetchall()

        metadata_by_chunk_id: dict[str, ChunkMetadata] = {}
        for row in rows:
            metadata = _chunk_metadata_from_row(row)
            if metadata.chunk_id in metadata_by_chunk_id:
                raise KnowledgeBaseQueryError()
            metadata_by_chunk_id[metadata.chunk_id] = metadata
        if len(metadata_by_chunk_id) != len(chunk_ids):
            raise KnowledgeBaseQueryError()
        return metadata_by_chunk_id

    def _require_ready_metadata(self) -> IndexMetadata:
        if not self._ready or self._index_metadata is None:
            raise KnowledgeBaseQueryError()
        return self._index_metadata

    def _validate_asset_files(self) -> None:
        for path in (
            self._metadata_db_path,
            self._index_path,
            self._index_meta_path,
        ):
            if not path.is_file():
                raise KnowledgeBaseLoadError()
            with path.open("rb"):
                pass

    def _load_index_metadata(self) -> IndexMetadata:
        with self._index_meta_path.open("r", encoding="utf-8") as file:
            raw_metadata = json.load(file)
        if not isinstance(raw_metadata, dict):
            raise KnowledgeBaseLoadError()

        embedding_model = _required_string(raw_metadata, "embedding_model")
        embedding_dim = _required_positive_int(raw_metadata, "embedding_dim")
        vector_metric = _required_string(raw_metadata, "vector_metric")
        vector_normalized = _required_bool(raw_metadata, "vector_normalized")
        vector_count = _required_positive_int(raw_metadata, "vector_count")
        is_partial = _required_bool(raw_metadata, "is_partial_embedding_index")

        if embedding_model != self._settings.upstream_embedding_model:
            raise KnowledgeBaseLoadError()
        if embedding_dim != self._settings.rag_embedding_dim:
            raise KnowledgeBaseLoadError()
        if vector_metric != "inner_product":
            raise KnowledgeBaseLoadError()
        if not vector_normalized:
            raise KnowledgeBaseLoadError()
        if is_partial and not self._settings.rag_allow_partial_index:
            raise KnowledgeBaseLoadError()

        return IndexMetadata(
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            vector_metric=vector_metric,
            vector_normalized=vector_normalized,
            vector_count=vector_count,
            is_partial_embedding_index=is_partial,
        )

    def _validate_sqlite(self, metadata: IndexMetadata) -> None:
        connection = self._connect_readonly()
        try:
            self._validate_quick_check(connection)
            self._validate_schema(connection)
            self._validate_vector_mapping(connection, metadata)
        finally:
            connection.close()

    def _connect_readonly(self) -> sqlite3.Connection:
        uri = self._metadata_db_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _validate_quick_check(connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA quick_check").fetchall()
        if len(rows) != 1 or rows[0][0] != "ok":
            raise KnowledgeBaseLoadError()

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        tables = {str(row[0]) for row in table_rows}
        if not set(_REQUIRED_TABLE_COLUMNS).issubset(tables):
            raise KnowledgeBaseLoadError()

        for table_name, required_columns in _REQUIRED_TABLE_COLUMNS.items():
            columns = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info([{table_name}])")
            }
            if not required_columns.issubset(columns):
                raise KnowledgeBaseLoadError()

    @staticmethod
    def _validate_vector_mapping(
        connection: sqlite3.Connection,
        metadata: IndexMetadata,
    ) -> None:
        row = connection.execute(
            """
            SELECT COUNT(*) AS mapped_count,
                   COUNT(DISTINCT vector_id) AS distinct_vector_count
            FROM chunks
            WHERE embedding_status = 'success' AND vector_id IS NOT NULL
            """
        ).fetchone()
        if row is None:
            raise KnowledgeBaseLoadError()
        mapped_count = int(row["mapped_count"])
        distinct_vector_count = int(row["distinct_vector_count"])
        if (
            mapped_count <= 0
            or mapped_count != distinct_vector_count
            or mapped_count != metadata.vector_count
        ):
            raise KnowledgeBaseLoadError()

    def _validate_faiss_index(self, index: Any, metadata: IndexMetadata) -> None:
        dimension = getattr(index, "d", None)
        vector_count = getattr(index, "ntotal", None)
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension != metadata.embedding_dim
        ):
            raise KnowledgeBaseLoadError()
        if (
            isinstance(vector_count, bool)
            or not isinstance(vector_count, int)
            or vector_count != metadata.vector_count
        ):
            raise KnowledgeBaseLoadError()


def _required_string(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeBaseLoadError()
    return value


def _required_positive_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KnowledgeBaseLoadError()
    return value


def _required_bool(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise KnowledgeBaseLoadError()
    return value


def _vector_search_hit_from_row(
    row: sqlite3.Row,
    candidate: VectorSearchCandidate,
) -> VectorSearchHit:
    """Build a traceable vector hit and reject incomplete or malformed DB rows."""
    chunk_id = _required_row_text(row, "chunk_id")
    doc_id = _required_row_text(row, "doc_id")
    chunk_text = _required_row_text(row, "chunk_text")
    relative_path = _required_row_text(row, "relative_path")
    row_vector_id = row["vector_id"]
    if isinstance(row_vector_id, bool) or not isinstance(row_vector_id, int):
        raise KnowledgeBaseQueryError()
    if int(row_vector_id) != candidate.vector_id:
        raise KnowledgeBaseQueryError()

    return VectorSearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_text=chunk_text,
        title=_optional_row_text(row, "title"),
        doc_title=_optional_row_text(row, "doc_title"),
        section_path=_optional_row_text(row, "section_path"),
        article_no=_optional_row_text(row, "article_no"),
        article_range=_optional_row_text(row, "article_range"),
        relative_path=relative_path,
        paragraph_start=_optional_row_int(row, "paragraph_start"),
        paragraph_end=_optional_row_int(row, "paragraph_end"),
        vector_id=candidate.vector_id,
        vector_score=candidate.vector_score,
        vector_rank=candidate.vector_rank,
    )


def _chunk_metadata_from_row(row: sqlite3.Row) -> ChunkMetadata:
    """Build one source record and reject malformed metadata before fusion output."""
    vector_id = row["vector_id"]
    if vector_id is not None and (
        isinstance(vector_id, bool) or not isinstance(vector_id, int)
    ):
        raise KnowledgeBaseQueryError()
    return ChunkMetadata(
        chunk_id=_required_row_text(row, "chunk_id"),
        doc_id=_required_row_text(row, "doc_id"),
        chunk_text=_required_row_text(row, "chunk_text"),
        title=_optional_row_text(row, "title"),
        doc_title=_optional_row_text(row, "doc_title"),
        section_path=_optional_row_text(row, "section_path"),
        article_no=_optional_row_text(row, "article_no"),
        article_range=_optional_row_text(row, "article_range"),
        relative_path=_required_row_text(row, "relative_path"),
        paragraph_start=_optional_row_int(row, "paragraph_start"),
        paragraph_end=_optional_row_int(row, "paragraph_end"),
        vector_id=int(vector_id) if vector_id is not None else None,
    )


def _required_row_text(row: sqlite3.Row, key: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value:
        raise KnowledgeBaseQueryError()
    return value


def _optional_row_text(row: sqlite3.Row, key: str) -> str | None:
    value = row[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise KnowledgeBaseQueryError()
    return value


def _optional_row_int(row: sqlite3.Row, key: str) -> int | None:
    value = row[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise KnowledgeBaseQueryError()
    return value


def _is_recoverable_fts_error(error: sqlite3.OperationalError) -> bool:
    """Classify only FTS-specific query/runtime failures as eligible for fallback."""
    message = str(error).casefold()
    return (
        "no such module: fts5" in message
        or "fts5: syntax error" in message
        or "malformed match expression" in message
    )
