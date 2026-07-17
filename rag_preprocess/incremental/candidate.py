"""Idempotent, versioned source-file archiving for incremental tasks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, Mapping

from .intake import sha256_stream
from .persistence import atomic_write_text


_TASK_TIME = re.compile(r"(\d{8}-\d{6})")


@dataclass(frozen=True)
class ArchiveOutcome:
    """The durable result for one source file, without exposing local paths."""

    archived: bool
    relative_path: str | None = None
    error_code: str | None = None


def archive_files(
    settings: object,
    rows: Iterable[Mapping[str, object]],
    task_id: str,
) -> dict[str, ArchiveOutcome]:
    """Archive published or unchanged files without ever overwriting history.

    Each file version has a task-specific directory.  The source is moved to a
    temporary name first and SHA-256 is checked before the final rename.  A
    matching existing target is a successful recovery from a prior interrupted
    archive; a differing target is deliberately left untouched.
    """

    outcomes: dict[str, ArchiveOutcome] = {}
    for row in rows:
        if row.get("state") not in {"PUBLISHED", "DUPLICATE_UNCHANGED"}:
            continue
        key = _row_key(row)
        try:
            outcome = _archive_one(settings, row, task_id)
        except (OSError, ValueError, shutil.Error):
            outcome = ArchiveOutcome(False, error_code="ARCHIVE_FAILED")
        outcomes[key] = outcome
    return outcomes


def _archive_one(
    settings: object,
    row: Mapping[str, object],
    task_id: str,
) -> ArchiveOutcome:
    file_name = _safe_file_name(row.get("file_name"))
    expected_hash = _required_hash(row.get("sha256"))
    source = Path(_required_text(row.get("frozen_path")))
    version_dir = _version_directory(task_id)
    archive_root = Path(getattr(settings, "archive_dir"))
    target = archive_root / file_name / version_dir / file_name
    temporary = target.with_name(f".{target.name}.archiving")
    relative_path = target.relative_to(archive_root).as_posix()

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _verify_hash(target, expected_hash)
        _write_archive_manifest(target.parent, row, task_id, relative_path)
        return ArchiveOutcome(True, relative_path)

    if temporary.exists():
        _verify_hash(temporary, expected_hash)
        os.replace(temporary, target)
        _write_archive_manifest(target.parent, row, task_id, relative_path)
        return ArchiveOutcome(True, relative_path)

    if not source.is_file():
        raise ValueError("ARCHIVE_SOURCE_MISSING")
    _verify_hash(source, expected_hash)
    shutil.move(str(source), str(temporary))
    try:
        _verify_hash(temporary, expected_hash)
        os.replace(temporary, target)
    except Exception:
        # Keep the verified temporary artifact for a later idempotent retry.
        raise
    _write_archive_manifest(target.parent, row, task_id, relative_path)
    return ArchiveOutcome(True, relative_path)


def _write_archive_manifest(
    directory: Path,
    row: Mapping[str, object],
    task_id: str,
    relative_path: str,
) -> None:
    """Write metadata only; source content and local absolute paths are excluded."""

    values = {
        "task_id": task_id,
        "file_name": _safe_file_name(row.get("file_name")),
        "sha256": _required_hash(row.get("sha256")),
        "size": str(row.get("size") or 0),
        "title": _metadata_text(row.get("title")),
        "action": str(row.get("action") or ""),
        "warning_count": str(len(row.get("warning_codes") or [])),
        "published_at": str(row.get("published_at") or ""),
        "delta_id": str(row.get("delta_id") or ""),
        "manifest_revision": str(row.get("manifest_revision") or ""),
        "archive_relative_path": relative_path,
    }
    text = "\n".join(f"{key}: {value}" for key, value in values.items()) + "\n"
    atomic_write_text(directory / "manifest.txt", text)


def _row_key(row: Mapping[str, object]) -> str:
    return str(row.get("version_id") or row.get("frozen_path") or row.get("file_name") or "")


def _version_directory(task_id: str) -> str:
    match = _TASK_TIME.search(task_id)
    return f"{match.group(1) if match else task_id}_{task_id}"


def _safe_file_name(value: object) -> str:
    name = _required_text(value)
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("ARCHIVE_FILE_NAME_INVALID")
    return name


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("ARCHIVE_METADATA_INVALID")
    return value


def _required_hash(value: object) -> str:
    digest = _required_text(value)
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError("ARCHIVE_HASH_INVALID")
    return digest.lower()


def _verify_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_stream(path).lower() != expected:
        raise ValueError("ARCHIVE_HASH_MISMATCH")


def _metadata_text(value: object) -> str:
    """Keep one manifest line per field even if a parser produced hostile text."""
    normalized = " ".join(str(value or "").split())[:300]
    normalized = re.sub(r"(?i)(?:[a-z]:[\\/]|\\\\)[^\s]*", "[已隐藏路径]", normalized)
    return re.sub(r"(?<!\S)/(?:projects|home|var|tmp|sevenH|opt|usr|etc)(?:/[^\s]*)?", "[已隐藏路径]", normalized)
