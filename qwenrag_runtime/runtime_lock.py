"""A held Windows file lock for mutually exclusive QwenRAG runtime sessions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import IO


class RuntimeAlreadyRunningError(RuntimeError):
    """Raised when an already-running QwenRAG supervisor owns the lock."""


class RuntimeLock:
    """Keep an OS lock open for the lifetime of one RAG runtime session.

    The operating system releases the lock if the launcher crashes, so no code
    ever deletes an unknown stale lock file merely because its PID looks old.
    """

    def __init__(self, path: Path, *, mode: str) -> None:
        self._path = path
        self._mode = mode
        self._file: IO[str] | None = None

    def acquire(self) -> None:
        if self._file is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+", encoding="utf-8")
        try:
            handle.seek(0)
            if not handle.read(1):
                handle.seek(0)
                handle.write(" ")
                handle.flush()
            _lock_file(handle)
            handle.seek(0)
            handle.truncate()
            json.dump(
                {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "mode": self._mode,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                handle,
                ensure_ascii=False,
            )
            handle.flush()
        except OSError as exc:
            handle.close()
            raise RuntimeAlreadyRunningError("另一个 QwenRAG 会话正在运行") from exc
        self._file = handle

    def release(self) -> None:
        if self._file is None:
            return
        try:
            _unlock_file(self._file)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "RuntimeLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def is_runtime_lock_held(path: Path) -> bool:
    """Return whether a live supervisor owns ``path`` without deleting it.

    A lock file may legitimately remain after a crash, so its existence alone
    is not evidence that QwenRAG is still running.  Only the OS file lock is
    authoritative.
    """
    if not path.is_file():
        return False
    try:
        handle = path.open("a+", encoding="utf-8")
    except OSError:
        # Treat an inaccessible lock as held: snapshots must fail closed.
        return True
    try:
        try:
            _lock_file(handle)
        except OSError:
            return True
        _unlock_file(handle)
        return False
    finally:
        handle.close()


def _lock_file(handle: IO[str]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: IO[str]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
