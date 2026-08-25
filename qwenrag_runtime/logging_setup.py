"""Bounded, metadata-only component log configuration for installed QwenRAG."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_component_logging(name: str, level: str, directory: Path) -> logging.Logger:
    """Attach one 10 MB / 10-file handler and remove expired logs safely."""
    directory.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=180)
    for candidate in directory.glob("*.log*"):
        try:
            if datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc) < cutoff:
                candidate.unlink()
        except OSError:
            continue
    logger = logging.getLogger(name)
    logger.setLevel(level)
    target = (directory / f"{name}.log").resolve()
    if not any(isinstance(item, RotatingFileHandler) and Path(item.baseFilename).resolve() == target for item in logger.handlers):
        handler = RotatingFileHandler(target, maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    # Keep propagation enabled so embedding applications and pytest's caplog can
    # still collect component events.  The rotating file handler remains the
    # durable local log sink.
    logger.propagate = True
    return logger
