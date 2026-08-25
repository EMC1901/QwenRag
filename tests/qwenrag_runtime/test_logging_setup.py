"""Tests for bounded component log storage used by gateway and RAG."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import uuid

from logging.handlers import RotatingFileHandler

from qwenrag_runtime.logging_setup import configure_component_logging


def test_component_logging_uses_bounded_rotation_and_removes_expired_logs(tmp_path: Path) -> None:
    expired = tmp_path / "previous.log.1"
    expired.write_text("old", encoding="utf-8")
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=181)).timestamp()
    os.utime(expired, (old_timestamp, old_timestamp))
    name = f"test_component_{uuid.uuid4().hex}"

    logger = configure_component_logging(name, "INFO", tmp_path)

    handlers = [handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)]
    assert len(handlers) == 1
    assert handlers[0].maxBytes == 10 * 1024 * 1024
    assert handlers[0].backupCount == 10
    assert not expired.exists()
    logger.info("metadata-only test event")
    assert (tmp_path / f"{name}.log").is_file()
