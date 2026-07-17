"""Stage 11: task-private Delta SQLite, vector records and version tombstones.

This module deliberately never copies a Base database or reads a Base JSONL
file.  It produces only a small, task-private Delta package.  Stage 12 is
responsible for building its FTS/FAISS assets and publishing it through the
knowledge manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence
import unicodedata

from rag_preprocess.database import init_db

from .persistence import atomic_write_text


class DeltaError(RuntimeError):
    """Raised when a task Delta cannot be built safely."""


@dataclass(frozen=True)
class ActiveDocument:
    """The currently effective version of one logical filename."""

    logical_name_key: str
    doc_id: str
    version_id: str
    file_hash_sha256: str | None
    relative_path: str
    chunk_ids: tuple[str, ...]
    vector_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparedVectors:
    """One file's vector IDs, written outside the Delta SQLite transaction."""

    version_id: str
    vector_ids_by_chunk: Mapping[str, int]
    next_vector_id: int
    vector_count: int


@dataclass(frozen=True)
class DeltaFileResult:
    """Result of an idempotent file-level Delta transaction."""

    version_id: str
    chunk_count: int
    tombstone_count: int
    already_written: bool = False


_AUXILIARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_identities (
  logical_name_key TEXT PRIMARY KEY,
  display_file_name TEXT NOT NULL,
  doc_id TEXT NOT NULL UNIQUE,
  active_file_hash_sha256 TEXT NOT NULL,
  active_version_id TEXT NOT NULL,
  origin TEXT NOT NULL CHECK(origin IN ('legacy', 'incremental')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
);

CREATE TABLE IF NOT EXISTS document_versions (
  version_id TEXT PRIMARY KEY,
  logical_name_key TEXT NOT NULL,
  display_file_name TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  extension TEXT NOT NULL,
  file_hash_sha256 TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  title TEXT NOT NULL,
  parse_method TEXT NOT NULL,
  page_count INTEGER,
  paragraph_count INTEGER NOT NULL DEFAULT 0,
  table_row_count INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL,
  warning_count INTEGER NOT NULL DEFAULT 0,
  archive_relative_path TEXT,
  published_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_versions_name
ON document_versions(logical_name_key, published_at);

CREATE TABLE IF NOT EXISTS delta_tombstones (
  entity_type TEXT NOT NULL CHECK(entity_type IN ('doc_version', 'chunk', 'vector')),
  entity_id TEXT NOT NULL,
  superseded_by_version_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(entity_type, entity_id)
);

CREATE TABLE IF NOT EXISTS delta_build_metadata (
  metadata_key TEXT PRIMARY KEY,
  metadata_value TEXT NOT NULL
);
"""


def create_task_delta(work_dir: Path, task_id: str) -> Path:
    """Create or reopen ``work/<task>/delta_staging`` without copying Base.

    The returned directory is private to one task and is intentionally not a
    published Delta path.  Reopening an existing staging directory supports a
    safe retry after a Worker interruption.
    """

    root = work_dir / task_id / "delta_staging"
    database = root / "delta.db"
    root.mkdir(parents=True, exist_ok=True)
    (root / "vector_index").mkdir(exist_ok=True)
    (root / "staged_vectors").mkdir(exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        init_db(connection)
        _migrate_delta_schema(connection)
        connection.execute(
            "INSERT OR IGNORE INTO delta_build_metadata(metadata_key, metadata_value) VALUES(?, ?)",
            ("task_id", task_id),
        )
        connection.commit()
    return root


def initial_next_vector_id(
    base_database: Path,
    *,
    delta_databases: Sequence[Path] = (),
) -> int:
    """Return one greater than the maximum known ID without reading JSONL.

    The manifest introduced in stage 12 will persist this value.  This helper
    only supports the one-time legacy initialization and test fixtures.
    """

    maximum = -1
    for database in (base_database, *delta_databases):
        if not database.is_file():
            continue
        with _readonly_connection(database) as connection:
            if not _has_table(connection, "chunks"):
                continue
            value = connection.execute("SELECT MAX(vector_id) FROM chunks").fetchone()[0]
            if value is not None:
                maximum = max(maximum, int(value))
    return maximum + 1


def load_effective_identities(
    base_database: Path,
    *,
    delta_databases: Sequence[Path] = (),
) -> dict[str, tuple[str, str]]:
    """Return logical-name identities after applying ordered published Deltas."""

    identities: dict[str, tuple[str, str]] = {}
    if base_database.is_file():
        with _readonly_connection(base_database) as connection:
            rows = connection.execute("SELECT doc_id, relative_path, file_hash_sha256 FROM documents").fetchall()
        for doc_id, relative_path, digest in rows:
            key = _logical_key(Path(str(relative_path)).name)
            if key in identities:
                raise DeltaError("LEGACY_NAME_COLLISION")
            identities[key] = (str(doc_id), str(digest or ""))
    for database in delta_databases:
        if not database.is_file():
            raise DeltaError("DELTA_DATABASE_MISSING")
        with _readonly_connection(database) as connection:
            if not _has_table(connection, "document_identities"):
                raise DeltaError("DELTA_IDENTITY_SCHEMA_MISSING")
            rows = connection.execute(
                "SELECT logical_name_key,doc_id,active_file_hash_sha256 FROM document_identities"
            ).fetchall()
        for key, doc_id, digest in rows:
            identities[_logical_key(str(key))] = (str(doc_id), str(digest))
    return identities


def resolve_active_document(
    logical_key: str,
    *,
    base_database: Path,
    delta_databases: Sequence[Path] = (),
) -> ActiveDocument | None:
    """Resolve one logical filename from Base plus ordered prior Delta DBs.

    Newer Delta identity rows take precedence.  Only metadata and the old
    document's chunk/vector IDs are read; no Base body table or JSONL is
    copied into the task Delta.
    """

    normalized = _logical_key(logical_key)
    for database in reversed(tuple(delta_databases)):
        if not database.is_file():
            continue
        with _readonly_connection(database) as connection:
            if not _has_table(connection, "document_identities"):
                continue
            identity = connection.execute(
                "SELECT doc_id, active_version_id, active_file_hash_sha256 FROM document_identities WHERE logical_name_key=?",
                (normalized,),
            ).fetchone()
            if identity is None:
                continue
            document = connection.execute(
                "SELECT relative_path FROM documents WHERE doc_id=?", (identity[0],)
            ).fetchone()
            return _active_document_from_connection(
                connection,
                logical_name_key=normalized,
                doc_id=str(identity[0]),
                version_id=str(identity[1]),
                file_hash=str(identity[2]),
                relative_path=str(document[0]) if document is not None else f"incremental/{normalized}",
            )

    if not base_database.is_file():
        return None
    with _readonly_connection(base_database) as connection:
        rows = connection.execute(
            "SELECT doc_id, relative_path, file_hash_sha256 FROM documents"
        ).fetchall()
        matches = [row for row in rows if _logical_key(Path(str(row[1])).name) == normalized]
        if len(matches) > 1:
            raise DeltaError("LEGACY_NAME_COLLISION")
        if not matches:
            return None
        doc_id, relative_path, file_hash = matches[0]
        digest = str(file_hash or "")
        return _active_document_from_connection(
            connection,
            logical_name_key=normalized,
            doc_id=str(doc_id),
            version_id=_legacy_version_id(str(doc_id), digest),
            file_hash=digest or None,
            relative_path=str(relative_path),
        )


def prepare_delta_vectors(
    delta_root: Path,
    row: Mapping[str, object],
    work_dir: Path,
    *,
    first_vector_id: int,
    embedding_dim: int,
    embedding_model: str,
) -> PreparedVectors:
    """Assign IDs and persist one file's vectors without touching Base JSONL."""

    version_id = _required_text(row, "version_id")
    source = work_dir / "vectors" / f"{version_id}.jsonl"
    target = delta_root / "staged_vectors" / f"{version_id}.jsonl"
    if not source.is_file():
        raise DeltaError("VECTOR_TEMPORARY_FILE_MISSING")
    records: list[dict[str, object]] = []
    ids: dict[str, int] = {}
    next_id = first_vector_id
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = str(record.get("chunk_id", ""))
            if not chunk_id or chunk_id in ids:
                raise DeltaError("VECTOR_CHUNK_ID_INVALID")
            if record.get("model") != embedding_model or record.get("dim") != embedding_dim:
                raise DeltaError("VECTOR_CONFIGURATION_MISMATCH")
            if record.get("normalized") is not True or not _valid_vector(record.get("vector"), embedding_dim):
                raise DeltaError("VECTOR_VALUE_INVALID")
            record["vector_id"] = next_id
            record["version_id"] = version_id
            ids[chunk_id] = next_id
            next_id += 1
            records.append(record)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, DeltaError):
            raise
        raise DeltaError("VECTOR_TEMPORARY_FILE_INVALID") from exc
    if not records:
        raise DeltaError("VECTOR_TEMPORARY_FILE_EMPTY")
    atomic_write_text(
        target,
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
    )
    return PreparedVectors(version_id, ids, next_id, len(records))


def write_delta_file(
    delta_root: Path,
    row: Mapping[str, object],
    work_dir: Path,
    vector_ids_by_chunk: Mapping[str, int],
    *,
    task_id: str,
    prior: ActiveDocument | None,
) -> DeltaFileResult:
    """Write one new/update file atomically into a task-private Delta DB."""

    version_id = _required_text(row, "version_id")
    parsed = _load_object(work_dir / f"{version_id}.parsed.json", "document")
    chunks_payload = _load_object(work_dir / f"{version_id}.chunks.json", "chunks")
    if not isinstance(parsed, dict):
        raise DeltaError("DELTA_INTERMEDIATE_INVALID")
    blocks = parsed.get("blocks")
    chunks = chunks_payload if isinstance(chunks_payload, list) else []
    if not isinstance(blocks, list) or not chunks:
        raise DeltaError("DELTA_INTERMEDIATE_INVALID")
    chunk_ids = [str(item.get("chunk_id", "")) for item in chunks if isinstance(item, dict)]
    if not chunk_ids or len(chunk_ids) != len(set(chunk_ids)) or set(chunk_ids) != set(vector_ids_by_chunk):
        raise DeltaError("DELTA_VECTOR_MAPPING_INVALID")
    doc_id = _required_text(row, "doc_id")
    if prior is not None and prior.doc_id != doc_id:
        raise DeltaError("UPDATE_DOC_ID_MISMATCH")

    database = delta_root / "delta.db"
    now = datetime.now(timezone.utc).isoformat()
    logical_key = _logical_key(_required_text(row, "logical_name_key"))
    file_name = _required_text(row, "file_name")
    file_hash = _required_text(row, "sha256")
    extension = _required_text(row, "extension")
    file_size = _required_int(row, "size")
    title = str(parsed.get("title") or Path(file_name).stem)
    parse_method = str(parsed.get("parse_method") or "incremental")
    relative_path = prior.relative_path if prior is not None else f"incremental/{file_name}"
    source_id = f"incremental-source:{doc_id}"

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing = connection.execute(
                "SELECT chunk_count FROM document_versions WHERE version_id=?", (version_id,)
            ).fetchone()
            if existing is not None:
                connection.rollback()
                return DeltaFileResult(version_id, int(existing[0]), 0, already_written=True)
            tombstone_count = _write_tombstones(connection, prior, version_id, now)
            _insert_source_and_document(
                connection,
                source_id=source_id,
                doc_id=doc_id,
                relative_path=relative_path,
                file_name=file_name,
                extension=extension,
                file_size=file_size,
                file_hash=file_hash,
                title=title,
                now=now,
            )
            _insert_blocks(connection, doc_id, blocks, now)
            _insert_structured_blocks(connection, doc_id, chunks_payload, blocks, now)
            _insert_chunks(connection, doc_id, chunks, vector_ids_by_chunk, now)
            connection.execute(
                """INSERT INTO document_identities(
                    logical_name_key,display_file_name,doc_id,active_file_hash_sha256,
                    active_version_id,origin,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (logical_key, file_name, doc_id, file_hash, version_id, "incremental", now, now),
            )
            connection.execute(
                """INSERT INTO document_versions(
                    version_id,logical_name_key,display_file_name,doc_id,task_id,extension,
                    file_hash_sha256,file_size,title,parse_method,page_count,paragraph_count,
                    table_row_count,chunk_count,warning_count,published_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    version_id,
                    logical_key,
                    file_name,
                    doc_id,
                    task_id,
                    extension,
                    file_hash,
                    file_size,
                    title,
                    parse_method,
                    parsed.get("page_count"),
                    int(parsed.get("paragraph_count") or 0),
                    int(parsed.get("table_row_count") or 0),
                    len(chunks),
                    len(parsed.get("warnings") or []),
                    now,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return DeltaFileResult(version_id, len(chunks), tombstone_count)


def write_delta_embeddings(delta_root: Path, *, embedding_dim: int, embedding_model: str) -> int:
    """Assemble only committed task vectors into Delta ``embeddings.jsonl``."""

    database = delta_root / "delta.db"
    with sqlite3.connect(database) as connection:
        versions = [row[0] for row in connection.execute("SELECT version_id FROM document_versions ORDER BY published_at, version_id")]
    records: list[str] = []
    seen_ids: set[int] = set()
    for version_id in versions:
        path = delta_root / "staged_vectors" / f"{version_id}.jsonl"
        if not path.is_file():
            raise DeltaError("DELTA_STAGED_VECTOR_MISSING")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            record = json.loads(line)
            vector_id = record.get("vector_id")
            if (
                record.get("version_id") != version_id
                or record.get("model") != embedding_model
                or record.get("dim") != embedding_dim
                or not isinstance(vector_id, int)
                or vector_id in seen_ids
                or not _valid_vector(record.get("vector"), embedding_dim)
            ):
                raise DeltaError("DELTA_VECTOR_RECORD_INVALID")
            seen_ids.add(vector_id)
            records.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    atomic_write_text(delta_root / "vector_index" / "embeddings.jsonl", "".join(records))
    return len(records)


def validate_delta_database(delta_root: Path) -> dict[str, int]:
    """Validate the stage-11 SQLite portion before later index construction."""

    with sqlite3.connect(delta_root / "delta.db") as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        counts = {
            "document_count": int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
            "chunk_count": int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
            "version_count": int(connection.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0]),
            "tombstone_count": int(connection.execute("SELECT COUNT(*) FROM delta_tombstones").fetchone()[0]),
        }
    if quick_check != "ok" or foreign_key_errors:
        raise DeltaError("DELTA_DATABASE_INCONSISTENT")
    if counts["document_count"] != counts["version_count"]:
        raise DeltaError("DELTA_DOCUMENT_VERSION_MISMATCH")
    return counts


def write_delta_metadata(
    delta_root: Path,
    *,
    delta_id: str,
    task_id: str,
    embedding_model: str,
    embedding_dim: int,
    parent_manifest_revision: int | None,
) -> dict[str, object]:
    """Write a provisional stage-11 metadata record for stage 12 to complete."""

    counts = validate_delta_database(delta_root)
    vector_count = _line_count(delta_root / "vector_index" / "embeddings.jsonl")
    if vector_count != counts["chunk_count"]:
        raise DeltaError("DELTA_VECTOR_COUNT_MISMATCH")
    metadata: dict[str, object] = {
        "schema_version": 1,
        "delta_id": delta_id,
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_manifest_revision": parent_manifest_revision,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "vector_metric": "inner_product",
        "vector_normalized": True,
        "document_count": counts["document_count"],
        "chunk_count": counts["chunk_count"],
        "vector_count": vector_count,
        "tombstone_count": counts["tombstone_count"],
        "validation_status": "pending_stage_12",
    }
    atomic_write_text(delta_root / "delta.meta.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def build_task_delta(
    work_dir: Path,
    task_id: str,
    rows: Iterable[Mapping[str, object]],
    *,
    base_database: Path,
    embedding_dim: int,
    embedding_model: str,
    prior_delta_databases: Sequence[Path] = (),
    first_vector_id: int | None = None,
    parent_manifest_revision: int | None = None,
) -> tuple[Path, dict[str, DeltaFileResult], dict[str, object]]:
    """Build the complete stage-11 Delta hand-off for already embedded files.

    The caller supplies only files whose parsing, Chunking and Embedding are
    complete.  This function intentionally stops before FTS/FAISS construction
    and manifest publication, which belong to stage 12.
    """

    task_work_dir = work_dir / task_id
    delta_root = create_task_delta(work_dir, task_id)
    cursor = (
        first_vector_id
        if first_vector_id is not None
        else initial_next_vector_id(base_database, delta_databases=prior_delta_databases)
    )
    cursor = max(cursor, _next_delta_vector_id(delta_root))
    results: dict[str, DeltaFileResult] = {}
    for row in rows:
        version_id = _required_text(row, "version_id")
        if _version_is_committed(delta_root / "delta.db", version_id):
            results[version_id] = DeltaFileResult(version_id, 0, 0, already_written=True)
            continue
        prior = resolve_active_document(
            _required_text(row, "logical_name_key"),
            base_database=base_database,
            delta_databases=prior_delta_databases,
        )
        prepared = prepare_delta_vectors(
            delta_root,
            row,
            task_work_dir,
            first_vector_id=cursor,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
        )
        result = write_delta_file(
            delta_root,
            row,
            task_work_dir,
            prepared.vector_ids_by_chunk,
            task_id=task_id,
            prior=prior,
        )
        results[version_id] = result
        cursor = prepared.next_vector_id
    write_delta_embeddings(delta_root, embedding_dim=embedding_dim, embedding_model=embedding_model)
    metadata = write_delta_metadata(
        delta_root,
        delta_id=f"delta-{task_id}",
        task_id=task_id,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        parent_manifest_revision=parent_manifest_revision,
    )
    return delta_root, results, metadata


def _migrate_delta_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(_AUXILIARY_SCHEMA)
    _add_column_if_missing(connection, "parsed_blocks", "page_number", "INTEGER")
    _add_column_if_missing(connection, "parsed_blocks", "ocr_confidence", "REAL")
    _add_column_if_missing(connection, "parsed_blocks", "quality_status", "TEXT")
    _add_column_if_missing(connection, "parsed_blocks", "source_locator", "TEXT")
    _add_column_if_missing(connection, "chunks", "page_start", "INTEGER")
    _add_column_if_missing(connection, "chunks", "page_end", "INTEGER")


def _add_column_if_missing(connection: sqlite3.Connection, table: str, name: str, kind: str) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if name not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")


def _active_document_from_connection(
    connection: sqlite3.Connection,
    *,
    logical_name_key: str,
    doc_id: str,
    version_id: str,
    file_hash: str | None,
    relative_path: str,
) -> ActiveDocument:
    rows = connection.execute(
        "SELECT chunk_id, vector_id FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,)
    ).fetchall()
    return ActiveDocument(
        logical_name_key=logical_name_key,
        doc_id=doc_id,
        version_id=version_id,
        file_hash_sha256=file_hash,
        relative_path=relative_path,
        chunk_ids=tuple(str(row[0]) for row in rows),
        vector_ids=tuple(int(row[1]) for row in rows if row[1] is not None),
    )


def _write_tombstones(
    connection: sqlite3.Connection,
    prior: ActiveDocument | None,
    replacement_version_id: str,
    now: str,
) -> int:
    if prior is None:
        return 0
    items = [("doc_version", prior.version_id)]
    items.extend(("chunk", chunk_id) for chunk_id in prior.chunk_ids)
    items.extend(("vector", str(vector_id)) for vector_id in prior.vector_ids)
    connection.executemany(
        "INSERT INTO delta_tombstones(entity_type,entity_id,superseded_by_version_id,created_at) VALUES(?,?,?,?)",
        [(kind, entity_id, replacement_version_id, now) for kind, entity_id in items],
    )
    return len(items)


def _insert_source_and_document(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    doc_id: str,
    relative_path: str,
    file_name: str,
    extension: str,
    file_size: int,
    file_hash: str,
    title: str,
    now: str,
) -> None:
    connection.execute(
        """INSERT INTO source_files(
            source_file_id,volume,relative_path,file_name,extension,file_size,
            file_hash_sha256,mtime,path_length,is_word_file,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (source_id, "incremental", relative_path, file_name, extension, file_size, file_hash, None, len(relative_path), int(extension == ".docx"), now),
    )
    connection.execute(
        """INSERT INTO documents(
            doc_id,source_file_id,title,relative_path,extension,file_size,
            file_hash_sha256,conversion_status,parse_status,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (doc_id, source_id, title, relative_path, extension, file_size, file_hash, "success", "success", now),
    )


def _insert_blocks(connection: sqlite3.Connection, doc_id: str, blocks: Iterable[object], now: str) -> None:
    for raw in blocks:
        if not isinstance(raw, dict):
            raise DeltaError("DELTA_PARSED_BLOCK_INVALID")
        index = int(raw.get("block_index", -1))
        text = str(raw.get("text", ""))
        if index < 0 or not text:
            raise DeltaError("DELTA_PARSED_BLOCK_INVALID")
        connection.execute(
            """INSERT INTO parsed_blocks(
                block_id,doc_id,block_index,block_type,text,paragraph_index,table_index,
                row_index,style_name,page_number,ocr_confidence,quality_status,
                source_locator,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"{doc_id}:p:{index}", doc_id, index, str(raw.get("block_type") or "paragraph"), text,
                raw.get("paragraph_index"), raw.get("table_index"), raw.get("row_index"), raw.get("style_name"),
                raw.get("page_number"), raw.get("ocr_confidence"), raw.get("quality_status"), raw.get("source_locator"), now,
            ),
        )


def _insert_structured_blocks(
    connection: sqlite3.Connection,
    doc_id: str,
    chunks_payload: object,
    blocks: Iterable[object],
    now: str,
) -> None:
    structured = chunks_payload.get("structured_blocks") if isinstance(chunks_payload, dict) else None
    if not isinstance(structured, list):
        structured = [
            {
                "block_index": item.get("block_index"),
                "block_type": item.get("block_type"),
                "raw_text": item.get("text"),
                "clean_text": item.get("text"),
                "detected_level": None,
                "section_path": None,
                "article_no": None,
            }
            for item in blocks
            if isinstance(item, dict)
        ]
    for raw in structured:
        if not isinstance(raw, dict):
            raise DeltaError("DELTA_STRUCTURED_BLOCK_INVALID")
        index = int(raw.get("block_index", -1))
        clean_text = str(raw.get("clean_text") or raw.get("text") or "")
        if index < 0 or not clean_text:
            raise DeltaError("DELTA_STRUCTURED_BLOCK_INVALID")
        connection.execute(
            """INSERT INTO structured_blocks(
                structured_block_id,block_id,doc_id,block_index,block_type,raw_text,
                clean_text,detected_level,section_path,article_no,is_noise,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"{doc_id}:s:{index}", f"{doc_id}:p:{index}", doc_id, index,
                str(raw.get("block_type") or "paragraph"), raw.get("raw_text") or clean_text,
                clean_text, raw.get("detected_level"), raw.get("section_path"), raw.get("article_no"),
                int(bool(raw.get("is_noise", False))), now,
            ),
        )


def _insert_chunks(
    connection: sqlite3.Connection,
    doc_id: str,
    chunks: Iterable[object],
    vector_ids_by_chunk: Mapping[str, int],
    now: str,
) -> None:
    for raw in chunks:
        if not isinstance(raw, dict):
            raise DeltaError("DELTA_CHUNK_INVALID")
        chunk_id = str(raw.get("chunk_id", ""))
        if not chunk_id:
            raise DeltaError("DELTA_CHUNK_INVALID")
        connection.execute(
            """INSERT INTO chunks(
                chunk_id,doc_id,chunk_index,chunk_text,chunk_text_for_embedding,title,
                section_path,article_no,article_range,paragraph_start,paragraph_end,
                token_count,vector_id,embedding_status,page_start,page_end,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                chunk_id, doc_id, int(raw.get("chunk_index", -1)), raw.get("chunk_text"),
                raw.get("chunk_text_for_embedding"), raw.get("title"), raw.get("section_path"),
                raw.get("article_no"), raw.get("article_range"), raw.get("paragraph_start"),
                raw.get("paragraph_end"), raw.get("token_count"), vector_ids_by_chunk[chunk_id],
                "success", raw.get("page_start"), raw.get("page_end"), now,
            ),
        )


def _load_object(path: Path, key: str) -> object:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DeltaError("DELTA_INTERMEDIATE_MISSING_OR_INVALID") from exc
    if not isinstance(payload, dict) or key not in payload:
        raise DeltaError("DELTA_INTERMEDIATE_MISSING_OR_INVALID")
    return payload[key]


def _readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _logical_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().casefold()


def _legacy_version_id(doc_id: str, digest: str) -> str:
    raw = f"legacy:{doc_id}:{digest}".encode("utf-8")
    return "legacy-" + hashlib.sha256(raw).hexdigest()[:24]


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise DeltaError(f"DELTA_REQUIRED_FIELD_MISSING:{key}")
    return value


def _required_int(row: Mapping[str, object], key: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeltaError(f"DELTA_REQUIRED_FIELD_INVALID:{key}") from exc
    if value < 0:
        raise DeltaError(f"DELTA_REQUIRED_FIELD_INVALID:{key}")
    return value


def _valid_vector(value: object, dimension: int) -> bool:
    if not isinstance(value, list) or len(value) != dimension:
        return False
    try:
        return all(float(item) == float(item) and abs(float(item)) != float("inf") for item in value)
    except (TypeError, ValueError):
        return False


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def _next_delta_vector_id(delta_root: Path) -> int:
    database = delta_root / "delta.db"
    with sqlite3.connect(database) as connection:
        value = connection.execute("SELECT MAX(vector_id) FROM chunks").fetchone()[0]
    return int(value) + 1 if value is not None else 0


def _version_is_committed(database: Path, version_id: str) -> bool:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT 1 FROM document_versions WHERE version_id=?", (version_id,)
        ).fetchone() is not None
