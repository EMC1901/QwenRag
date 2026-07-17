"""Small, disposable knowledge-base fixtures for incremental-ingestion tests.

The fixture deliberately contains only synthetic metadata and vectors.  It is
safe for fault-injection tests and must never be pointed at ``rag_data``.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Callable, Iterator

import numpy as np

from rag_preprocess.database import init_db
from rag_preprocess.faiss_builder import build_faiss_index


EMBEDDING_DIM = 3
EMBEDDING_MODEL = "incremental-fixture-embedding"

_DOCUMENTS = (
    ("fixture-doc-1", "存量制度甲", "存量制度甲.docx"),
    ("fixture-doc-2", "存量制度乙", "存量制度乙.txt"),
    ("fixture-doc-3", "存量制度丙", "存量制度丙.pdf"),
)
_VECTORS = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)

FaissIndexBuilder = Callable[[np.ndarray, list[int], Path], int]


def create_small_knowledge_base(
    destination: Path,
    *,
    duplicate_logical_name: bool = False,
    faiss_index_builder: FaissIndexBuilder = build_faiss_index,
) -> Path:
    """Create one isolated 3-document / 3-vector knowledge-base fixture.

    ``destination`` must not exist.  The default builder produces a real
    ``IndexIDMap2(IndexFlatIP)`` asset whenever ``faiss-cpu`` is installed.
    Tests without the optional runtime may inject a small writer and pair it
    with a FAISS-shaped loader when exercising :class:`KnowledgeBase`.
    """
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"fixture destination already exists: {destination}")

    vector_dir = destination / "vector_index"
    destination.mkdir(parents=True)
    vector_dir.mkdir()
    try:
        _write_database(destination / "metadata.db", duplicate_logical_name)
        _write_embeddings(vector_dir / "embeddings.jsonl")
        vector_count = faiss_index_builder(
            _VECTORS,
            list(range(len(_VECTORS))),
            vector_dir / "index.faiss",
        )
        if vector_count != len(_VECTORS):
            raise ValueError("fixture FAISS builder returned an unexpected vector count")
        _write_metadata(vector_dir, vector_count)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


@contextmanager
def temporary_small_knowledge_base(
    *,
    duplicate_logical_name: bool = False,
    faiss_index_builder: FaissIndexBuilder = build_faiss_index,
) -> Iterator[Path]:
    """Create and remove a fixture in an OS temporary directory."""
    with tempfile.TemporaryDirectory(prefix="qwenrag-incremental-") as directory:
        root = Path(directory) / "knowledge-base"
        yield create_small_knowledge_base(
            root,
            duplicate_logical_name=duplicate_logical_name,
            faiss_index_builder=faiss_index_builder,
        )


def _write_database(path: Path, duplicate_logical_name: bool) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        init_db(connection)
        document_rows = list(_DOCUMENTS)
        if duplicate_logical_name:
            document_rows[2] = ("fixture-doc-3", "存量制度丙", "存量制度甲.DOCX")

        source_rows = []
        document_values = []
        chunk_rows = []
        fts_rows = []
        for vector_id, (doc_id, title, file_name) in enumerate(document_rows):
            source_id = f"fixture-source-{vector_id + 1}"
            relative_path = f"fixtures/{file_name}"
            source_rows.append(
                (
                    source_id,
                    "fixture",
                    relative_path,
                    file_name,
                    Path(file_name).suffix.lower(),
                    1,
                    f"fixture-sha256-{vector_id + 1}",
                    None,
                    len(relative_path),
                    1 if file_name.lower().endswith(".docx") else 0,
                    "2026-01-01T00:00:00+00:00",
                )
            )
            document_values.append(
                (
                    doc_id,
                    source_id,
                    title,
                    relative_path,
                    Path(file_name).suffix.lower(),
                    "fixture-file-hash",
                    "success",
                    "success",
                    "2026-01-01T00:00:00+00:00",
                )
            )
            chunk_id = f"fixture-chunk-{vector_id + 1}"
            chunk_text = f"{title} 的测试正文第 {vector_id + 1} 段"
            chunk_rows.append(
                (
                    chunk_id,
                    doc_id,
                    vector_id,
                    chunk_text,
                    chunk_text,
                    title,
                    "第一章",
                    "第一条",
                    "第一条",
                    1,
                    1,
                    8,
                    vector_id,
                    "success",
                    "2026-01-01T00:00:00+00:00",
                )
            )
            fts_rows.append((chunk_id, title, "第一章", "第一条", chunk_text))

        connection.executemany(
            """INSERT INTO source_files (
                source_file_id, volume, relative_path, file_name, extension,
                file_size, file_hash_sha256, mtime, path_length, is_word_file,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            source_rows,
        )
        connection.executemany(
            """INSERT INTO documents (
                doc_id, source_file_id, title, relative_path, extension,
                file_hash_sha256, conversion_status, parse_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            document_values,
        )
        connection.executemany(
            """INSERT INTO chunks (
                chunk_id, doc_id, chunk_index, chunk_text, chunk_text_for_embedding,
                title, section_path, article_no, article_range, paragraph_start,
                paragraph_end, token_count, vector_id, embedding_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            chunk_rows,
        )
        connection.executemany("INSERT INTO chunk_fts VALUES (?, ?, ?, ?, ?)", fts_rows)
        connection.commit()
    finally:
        connection.close()


def _write_embeddings(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for vector_id, vector in enumerate(_VECTORS.tolist()):
            handle.write(
                json.dumps(
                    {
                        "vector_id": vector_id,
                        "chunk_id": f"fixture-chunk-{vector_id + 1}",
                        "model": EMBEDDING_MODEL,
                        "dim": EMBEDDING_DIM,
                        "normalized": True,
                        "vector": vector,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_metadata(vector_dir: Path, vector_count: int) -> None:
    common = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "vector_normalized": True,
        "vector_metric": "inner_product",
    }
    (vector_dir / "embeddings.meta.json").write_text(
        json.dumps(
            common
            | {
                "vector_file": "vector_index/embeddings.jsonl",
                "selected_chunk_count": vector_count,
                "success_count": vector_count,
                "failed_count": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (vector_dir / "index.meta.json").write_text(
        json.dumps(
            common
            | {
                "index_type": "faiss",
                "faiss_factory": "IndexIDMap2(IndexFlatIP)",
                "vector_count": vector_count,
                "source_vector_file_line_count": vector_count,
                "db_total_chunks": vector_count,
                "db_embedding_success_count": vector_count,
                "db_success_with_vector_id_count": vector_count,
                "is_partial_embedding_index": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
