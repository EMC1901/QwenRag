"""Atomic checkpoint, status, and result writers for incremental tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
from typing import Mapping


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace one file only after its complete temporary content is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_checkpoint(path: Path, checkpoint: Mapping[str, object]) -> None:
    """Write one versioned task/file checkpoint atomically as UTF-8 JSON."""
    atomic_write_text(path, json.dumps(dict(checkpoint), ensure_ascii=False, indent=2) + "\n")


def read_checkpoint(path: Path) -> dict[str, object]:
    """Read a checkpoint object without accepting non-object JSON."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("checkpoint 必须是 JSON 对象")
    return value


def write_status(path: Path, text: str) -> None:
    """Write Windows-Notepad-friendly task status text atomically."""
    atomic_write_text(path, text, encoding="utf-8-sig")


def write_result(path: Path, text: str) -> None:
    """Write Windows-Notepad-friendly final result text atomically."""
    atomic_write_text(path, text, encoding="utf-8-sig")
