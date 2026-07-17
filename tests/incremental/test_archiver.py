"""Stage-13 regression tests for source-file version archiving."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from rag_preprocess.incremental.candidate import archive_files


def _row(source: Path, content: bytes, *, state: str = "PUBLISHED") -> dict[str, object]:
    return {
        "file_name": source.name,
        "frozen_path": str(source),
        "sha256": sha256(content).hexdigest(),
        "size": len(content),
        "state": state,
        "action": "NEW",
        "version_id": "ver-test",
        "warning_codes": [],
        "delta_id": "delta-test",
        "manifest_revision": 4,
    }


def test_archiving_is_versioned_hash_checked_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "incoming" / "example.txt"
    source.parent.mkdir()
    content = b"customer metadata only"
    source.write_bytes(content)
    settings = SimpleNamespace(archive_dir=tmp_path / "archive")
    row = _row(source, content)

    first = archive_files(settings, [row], "ingest-20260717-120000-a1b2")
    outcome = first["ver-test"]
    assert outcome.archived
    assert outcome.relative_path == "example.txt/20260717-120000_ingest-20260717-120000-a1b2/example.txt"
    target = settings.archive_dir / outcome.relative_path
    assert target.read_bytes() == content
    assert not source.exists()
    manifest = target.parent / "manifest.txt"
    assert "delta_id: delta-test" in manifest.read_text(encoding="utf-8")
    assert str(source.parent) not in manifest.read_text(encoding="utf-8")

    second = archive_files(settings, [row], "ingest-20260717-120000-a1b2")
    assert second["ver-test"].archived


def test_archiving_never_overwrites_a_different_existing_version(tmp_path: Path) -> None:
    source = tmp_path / "incoming" / "example.txt"
    source.parent.mkdir()
    content = b"expected source"
    source.write_bytes(content)
    settings = SimpleNamespace(archive_dir=tmp_path / "archive")
    task_id = "ingest-20260717-120000-a1b2"
    target = settings.archive_dir / "example.txt" / "20260717-120000_ingest-20260717-120000-a1b2" / "example.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different historical file")

    result = archive_files(settings, [_row(source, content)], task_id)
    assert not result["ver-test"].archived
    assert result["ver-test"].error_code == "ARCHIVE_FAILED"
    assert source.read_bytes() == content
    assert target.read_bytes() == b"different historical file"
