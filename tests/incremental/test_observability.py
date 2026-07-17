"""Stage-3 atomic status, privacy, retention, and heartbeat tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import threading
import time

import pytest

from rag_preprocess.incremental.observability import (
    PrivacyRedactor,
    cleanup_retained_files,
    run_with_heartbeat,
)
from rag_preprocess.incremental.persistence import (
    atomic_write_text,
    write_checkpoint,
    write_result,
    write_status,
)
from rag_preprocess.incremental.settings import load_incremental_settings


def _settings(tmp_path: Path):
    return load_incremental_settings(
        project_root=tmp_path,
        environ={"INCREMENTAL_KB_ROOT": "data", "OCR_MODEL_DIR": "models/ocr"},
    )


def test_status_and_result_use_atomic_utf8_bom_replacement(tmp_path: Path, monkeypatch) -> None:
    """A failed replacement leaves the prior user-visible status complete and intact."""
    status_path = tmp_path / "任务 状态.status.txt"
    result_path = tmp_path / "任务 结果.result.txt"
    write_status(status_path, "任务编号：ingest-test\n状态：SUBMITTED\n")
    write_result(result_path, "任务编号：ingest-test\n结果：等待处理\n")
    original = status_path.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("simulated crash")

    monkeypatch.setattr("rag_preprocess.incremental.persistence.os.replace", fail_replace)
    with pytest.raises(OSError):
        write_status(status_path, "任务编号：ingest-test\n状态：PREFLIGHT\n")

    assert status_path.read_bytes() == original
    assert status_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert result_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not list(tmp_path.glob(".*.tmp"))


def test_privacy_redactor_hides_body_absolute_path_and_authorization() -> None:
    """Technical logs retain diagnostics without leaking source text or secrets."""
    redactor = PrivacyRedactor(forbidden_literals=("秘密正文标记", "sk-test-secret"))

    rendered = redactor.redact(
        "正文=秘密正文标记 path=C:\\客户资料\\制度.pdf "
        "Authorization: Bearer sk-test-secret"
    )

    assert "秘密正文标记" not in rendered
    assert "sk-test-secret" not in rendered
    assert "C:\\客户资料" not in rendered
    assert "[REDACTED]" in rendered
    assert "[ABSOLUTE_PATH]" in rendered


def test_retention_cleans_old_logs_and_failed_work_but_never_results(tmp_path: Path) -> None:
    """Retention scopes deletion to expired technical logs and failed work only."""
    settings = _settings(tmp_path)
    settings.ensure_directories()
    now = datetime(2026, 7, 16, tzinfo=timezone.utc)
    old = (now - timedelta(days=181)).timestamp()
    recent = (now - timedelta(days=1)).timestamp()
    old_log = settings.logs_dir / "old.log"
    recent_log = settings.logs_dir / "recent.log"
    old_log.write_text("old", encoding="utf-8")
    recent_log.write_text("recent", encoding="utf-8")
    os.utime(old_log, (old, old))
    os.utime(recent_log, (recent, recent))

    expired_work = settings.work_dir / "ingest-expired"
    retained_work = settings.work_dir / "ingest-recent"
    expired_work.mkdir()
    retained_work.mkdir()
    write_checkpoint(expired_work / "task.json", {"state": "FAILED_RESUMABLE"})
    write_checkpoint(retained_work / "task.json", {"state": "FAILED_FINAL"})
    os.utime(expired_work, (now.timestamp() - 8 * 86400, now.timestamp() - 8 * 86400))
    os.utime(retained_work, (recent, recent))
    result = settings.results_dir / "ingest-expired.result.txt"
    write_result(result, "长期保留\n")
    os.utime(result, (old, old))

    report = cleanup_retained_files(settings, now=now)

    assert report.deleted_log_count == 1
    assert report.deleted_work_count == 1
    assert not old_log.exists()
    assert recent_log.exists()
    assert not expired_work.exists()
    assert retained_work.exists()
    assert result.exists()


def test_heartbeat_continues_during_blocking_third_party_call() -> None:
    """A heartbeat thread reports liveness even when work emits no callbacks."""
    heartbeats: list[float] = []
    entered = threading.Event()

    def blocking_work() -> str:
        entered.set()
        time.sleep(0.06)
        return "done"

    def heartbeat() -> None:
        heartbeats.append(time.monotonic())

    assert run_with_heartbeat(blocking_work, heartbeat, interval_seconds=0.01) == "done"
    assert entered.is_set()
    assert len(heartbeats) >= 3
