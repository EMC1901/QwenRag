"""实现 FTS5 + FAISS 混合检索。"""

import sqlite3
from dataclasses import dataclass


@dataclass
class SearchResult:
    """检索结果。"""

    chunk_id: str
    score: float
    chunk_text: str | None = None
    title: str | None = None
    section_path: str | None = None
    article_no: str | None = None
    article_range: str | None = None
    source: str = "fts"  # "fts" | "vector" | "merged"


def search_fts(conn: sqlite3.Connection, query: str, top_k: int = 20) -> list[SearchResult]:
    """关键词检索 (FTS5 BM25)。"""
    try:
        cursor = conn.execute(
            """SELECT chunk_id, bm25(chunk_fts) AS score, title, section_path, article_no
               FROM chunk_fts
               WHERE chunk_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (query, top_k),
        )
        results = []
        for row in cursor:
            results.append(SearchResult(
                chunk_id=row[0],
                score=row[1],
                title=row[2],
                section_path=row[3],
                article_no=row[4],
                source="fts",
            ))
        return results
    except Exception:
        return []


def search_vector(index, query_vector, top_k: int = 20) -> list[SearchResult]:
    """向量检索 (FAISS)。"""
    # 第一版为桩实现
    return []


def merge_results(
    fts_results: list[SearchResult],
    vector_results: list[SearchResult],
    fts_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> list[SearchResult]:
    """融合两路结果，按加权分数排序。"""
    merged: dict[str, SearchResult] = {}

    # 归一化 FTS 分数
    max_fts = max((abs(r.score) for r in fts_results), default=1)
    for r in fts_results:
        norm_score = r.score / max_fts if max_fts > 0 else 0
        merged[r.chunk_id] = SearchResult(
            chunk_id=r.chunk_id,
            score=norm_score * fts_weight,
            chunk_text=r.chunk_text,
            title=r.title,
            section_path=r.section_path,
            article_no=r.article_no,
            source="merged",
        )

    # 归一化向量分数
    max_vec = max((r.score for r in vector_results), default=1)
    for r in vector_results:
        norm_score = r.score / max_vec if max_vec > 0 else 0
        if r.chunk_id in merged:
            merged[r.chunk_id].score += norm_score * vector_weight
        else:
            merged[r.chunk_id] = SearchResult(
                chunk_id=r.chunk_id,
                score=norm_score * vector_weight,
                chunk_text=r.chunk_text,
                title=r.title,
                section_path=r.section_path,
                article_no=r.article_no,
                source="merged",
            )

    # 按分数降序排列
    sorted_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
    return sorted_results


def load_chunks_by_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> list[dict]:
    """根据 chunk_id 读取完整 chunk 和来源。"""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    cursor = conn.execute(
        f"""SELECT c.chunk_id, c.chunk_text, c.title, c.section_path, c.article_no,
                   c.article_range,
                   c.doc_id, d.title AS doc_title, d.relative_path
            FROM chunks c
            LEFT JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id IN ({placeholders})""",
        chunk_ids,
    )
    results = []
    for row in cursor:
        results.append({
            "chunk_id": row[0],
            "chunk_text": row[1],
            "title": row[2],
            "section_path": row[3],
            "article_no": row[4],
            "article_range": row[5],
            "doc_id": row[6],
            "doc_title": row[7],
            "relative_path": row[8],
        })
    return results
