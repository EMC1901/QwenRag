"""Privacy-safe logs, retention, and liveness heartbeat utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import re
import shutil
import threading
from typing import Callable, Iterable, TypeVar

from rag_preprocess.incremental.persistence import read_checkpoint
from rag_preprocess.incremental.settings import IncrementalSettings


_T = TypeVar("_T")
_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_SECRET_FIELD_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|token|password)\s*[:=]\s*)[^\s,;]+"
)
_WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<!\w)[A-Za-z]:\\[^\s\"']+")
_UNIX_ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w:])/[^\s\"']+")


class PrivacyRedactor:
    """Remove known bodies/secrets and generic sensitive fields from log strings."""

    def __init__(self, *, forbidden_literals: Iterable[str] = ()) -> None:
        self._forbidden_literals = tuple(value for value in forbidden_literals if value)

    def redact(self, value: object) -> str:
        text = str(value)
        for literal in self._forbidden_literals:
            text = text.replace(literal, "[REDACTED]")
        text = _AUTHORIZATION_PATTERN.sub(r"\1[REDACTED]", text)
        text = _SECRET_FIELD_PATTERN.sub(r"\1[REDACTED]", text)
        text = _WINDOWS_ABSOLUTE_PATH_PATTERN.sub("[ABSOLUTE_PATH]", text)
        return _UNIX_ABSOLUTE_PATH_PATTERN.sub("[ABSOLUTE_PATH]", text)


class _PrivacyJsonFormatter(logging.Formatter):
    """Emit one redacted structured JSON event per technical-log line."""

    def __init__(self, redactor: PrivacyRedactor) -> None:
        super().__init__()
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": self._redactor.redact(record.getMessage()),
        }
        return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def configure_task_logger(
    name: str,
    log_path: Path,
    *,
    forbidden_literals: Iterable[str] = (),
) -> logging.Logger:
    """Create an isolated JSONL logger that never writes source text by default."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(_PrivacyJsonFormatter(PrivacyRedactor(forbidden_literals=forbidden_literals)))
    logger.addHandler(handler)
    return logger


@dataclass(frozen=True)
class RetentionReport:
    deleted_log_count: int = 0
    deleted_work_count: int = 0


def cleanup_retained_files(
    settings: IncrementalSettings,
    *,
    now: datetime | None = None,
) -> RetentionReport:
    """Delete only expired technical logs and failed/interrupted work directories."""
    current = now or datetime.now(timezone.utc)
    log_cutoff = current - timedelta(days=settings.tech_log_retention_days)
    work_cutoff = current - timedelta(days=settings.failed_work_retention_days)
    deleted_logs = 0
    deleted_work = 0
    if settings.logs_dir.is_dir():
        for path in settings.logs_dir.rglob("*.log"):
            if _older_than(path, log_cutoff):
                path.unlink(missing_ok=True)
                deleted_logs += 1
    if settings.work_dir.is_dir():
        for task_dir in settings.work_dir.iterdir():
            if not task_dir.is_dir() or not _older_than(task_dir, work_cutoff):
                continue
            state = _checkpoint_state(task_dir / "task.json")
            if state in {"FAILED_RESUMABLE", "FAILED_FINAL"}:
                shutil.rmtree(task_dir)
                deleted_work += 1
    return RetentionReport(deleted_log_count=deleted_logs, deleted_work_count=deleted_work)


def run_with_heartbeat(
    work: Callable[[], _T],
    on_heartbeat: Callable[[], None],
    *,
    interval_seconds: float = 10.0,
) -> _T:
    """Run blocking work while a daemon thread refreshes liveness at an interval."""
    if interval_seconds <= 0:
        raise ValueError("heartbeat interval must be positive")
    stop = threading.Event()

    def heartbeat_loop() -> None:
        while not stop.wait(interval_seconds):
            on_heartbeat()

    thread = threading.Thread(target=heartbeat_loop, name="incremental-heartbeat", daemon=True)
    thread.start()
    try:
        return work()
    finally:
        stop.set()
        thread.join(timeout=interval_seconds + 1)


def _older_than(path: Path, cutoff: datetime) -> bool:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return modified < cutoff


def _checkpoint_state(path: Path) -> str | None:
    try:
        state = read_checkpoint(path).get("state")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return state if isinstance(state, str) else None
