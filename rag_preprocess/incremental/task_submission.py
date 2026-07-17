"""Stage-2 single-task submission and Windows worker identity primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import ctypes
import json
import os
from pathlib import Path
import secrets
import shutil
import time
from typing import Callable, Protocol

from rag_preprocess.incremental.persistence import (
    write_checkpoint,
    write_status,
)
from rag_preprocess.incremental.settings import IncrementalSettings


class TaskSubmissionError(RuntimeError):
    """Raised when a task or its active-lock record cannot be safely handled."""


class WorkerStartError(TaskSubmissionError):
    """Raised after a launcher failure has released the task lock."""


class TaskAction(str, Enum):
    """Submission result suitable for both the CLI and PowerShell wrapper."""

    CREATED = "created"
    ACTIVE = "active"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class ProcessIdentity:
    """PID plus immutable Windows process creation timestamp ticks."""

    pid: int
    created_at_ticks: int


class ProcessInspector(Protocol):
    """Injectable interface for PID-reuse-safe worker liveness checks."""

    def current_identity(self) -> ProcessIdentity: ...

    def is_same_process(self, identity: ProcessIdentity) -> bool: ...


class WindowsProcessInspector:
    """Read process creation time with Windows APIs, without a psutil dependency."""

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def current_identity(self) -> ProcessIdentity:
        pid = os.getpid()
        return ProcessIdentity(pid=pid, created_at_ticks=self._creation_ticks(pid))

    def is_same_process(self, identity: ProcessIdentity) -> bool:
        try:
            return self._creation_ticks(identity.pid) == identity.created_at_ticks
        except OSError:
            return False

    def _creation_ticks(self, pid: int) -> int:
        if os.name != "nt":
            raise OSError("增量 Worker 的 PID 创建时间校验仅支持 Windows")
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(self._PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            raise OSError(f"无法读取进程 {pid} 的创建时间")
        try:
            creation = _FILETIME()
            exit_time = _FILETIME()
            kernel_time = _FILETIME()
            user_time = _FILETIME()
            ok = kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            )
            if not ok:
                raise OSError(f"无法读取进程 {pid} 的创建时间")
            return (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
        finally:
            kernel32.CloseHandle(handle)


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]


@dataclass(frozen=True)
class SubmissionOutcome:
    """One created, active, or recoverable task submission response."""

    action: TaskAction
    task_id: str
    status_relative_path: str
    worker_stdout_relative_path: str
    worker_stderr_relative_path: str

    @property
    def should_start_worker(self) -> bool:
        return self.action in {TaskAction.CREATED, TaskAction.RECOVERY}

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "action": self.action.value,
            "task_id": self.task_id,
            "status_relative_path": self.status_relative_path,
            "worker_stdout_relative_path": self.worker_stdout_relative_path,
            "worker_stderr_relative_path": self.worker_stderr_relative_path,
            "should_start_worker": self.should_start_worker,
        }


def submit_task(
    settings: IncrementalSettings,
    *,
    process_inspector: ProcessInspector | None = None,
) -> SubmissionOutcome:
    """Atomically reserve the only active incremental task or return its details."""
    settings.ensure_directories()
    lock_dir = _lock_dir(settings)
    try:
        lock_dir.mkdir()
    except FileExistsError:
        return _handle_existing_lock(settings, process_inspector or WindowsProcessInspector())
    except OSError as exc:
        raise TaskSubmissionError("无法创建增量任务锁目录") from exc

    task_id = _new_task_id()
    try:
        return _create_reserved_task(settings, task_id)
    except Exception:
        _remove_lock_dir(settings)
        raise


def submit_and_launch(
    settings: IncrementalSettings,
    *,
    launcher: Callable[[str], None],
    process_inspector: ProcessInspector | None = None,
) -> SubmissionOutcome:
    """Submit and invoke an injected worker launcher; clean up launch failures."""
    outcome = submit_task(settings, process_inspector=process_inspector)
    if not outcome.should_start_worker:
        return outcome
    try:
        launcher(outcome.task_id)
    except Exception as exc:
        release_worker_start_failure(settings, outcome.task_id)
        raise WorkerStartError("后台 Worker 未能启动，任务锁已释放") from exc
    return outcome


def claim_worker(
    settings: IncrementalSettings,
    task_id: str,
    *,
    process_inspector: ProcessInspector | None = None,
) -> ProcessIdentity:
    """Record the current Worker PID and creation time after it has started."""
    identity = (process_inspector or WindowsProcessInspector()).current_identity()
    lock = _read_lock(settings)
    if lock.get("task_id") != task_id:
        raise TaskSubmissionError("任务锁与 Worker 任务编号不匹配")
    lock.update(
        {
            "state": "RUNNING",
            "worker_pid": identity.pid,
            "worker_process_created_at": identity.created_at_ticks,
            "worker_started_at": _utc_now(),
            "heartbeat_at": _utc_now(),
        }
    )
    _write_lock(settings, lock)
    return identity


def release_worker_start_failure(settings: IncrementalSettings, task_id: str) -> None:
    """Release only the matching lock after PowerShell cannot start a Worker."""
    release_task_lock(settings, task_id)


def release_task_lock(settings: IncrementalSettings, task_id: str) -> None:
    """Release only the matching task lock after a Worker reaches a terminal state."""
    try:
        lock = _read_lock(settings)
    except TaskSubmissionError:
        return
    if lock.get("task_id") != task_id:
        return
    _remove_lock_dir(settings)


def build_worker_command(
    *,
    python_executable: Path,
    script_path: Path,
    task_id: str,
) -> list[str]:
    """Return discrete worker arguments so callers can preserve spaced paths."""
    return [
        str(python_executable),
        "-u",
        str(script_path),
        "worker",
        "--task-id",
        task_id,
    ]


def _create_reserved_task(settings: IncrementalSettings, task_id: str) -> SubmissionOutcome:
    workspace = settings.work_dir / task_id
    workspace.mkdir(parents=True, exist_ok=False)
    status_relative_path = _relative(settings, settings.results_dir / f"{task_id}.status.txt")
    task_relative_path = _relative(settings, workspace / "task.json")
    stdout_relative_path = _relative(settings, workspace / "worker.stdout.log")
    stderr_relative_path = _relative(settings, workspace / "worker.stderr.log")
    submitted_at = _utc_now()
    task = {
        "schema_version": 1,
        "task_id": task_id,
        "state": "SUBMITTED",
        "submitted_at": submitted_at,
        "status_relative_path": status_relative_path,
    }
    lock = {
        "schema_version": 1,
        "task_id": task_id,
        "state": "SUBMITTED",
        "submitted_at": submitted_at,
        "worker_pid": None,
        "worker_process_created_at": None,
        "worker_started_at": None,
        "heartbeat_at": submitted_at,
        "status_relative_path": status_relative_path,
        "task_relative_path": task_relative_path,
        "worker_stdout_relative_path": stdout_relative_path,
        "worker_stderr_relative_path": stderr_relative_path,
    }
    write_checkpoint(workspace / "task.json", task)
    write_status(
        settings.results_dir / f"{task_id}.status.txt",
        f"任务编号：{task_id}\n状态：SUBMITTED\n提交时间：{submitted_at}\n",
    )
    _write_lock(settings, lock)
    return _outcome_from_lock(TaskAction.CREATED, lock)


def _handle_existing_lock(
    settings: IncrementalSettings,
    process_inspector: ProcessInspector,
) -> SubmissionOutcome:
    lock = _read_lock(settings, retry_for_initial_write=True)
    worker_identity = _identity_from_lock(lock)
    if worker_identity is None:
        return _outcome_from_lock(TaskAction.ACTIVE, lock)
    if process_inspector.is_same_process(worker_identity):
        return _outcome_from_lock(TaskAction.ACTIVE, lock)

    lock.update(
        {
            "state": "RECOVERY_PENDING",
            "worker_pid": None,
            "worker_process_created_at": None,
            "worker_started_at": None,
            "recovery_requested_at": _utc_now(),
        }
    )
    _write_lock(settings, lock)
    return _outcome_from_lock(TaskAction.RECOVERY, lock)


def _identity_from_lock(lock: dict[str, object]) -> ProcessIdentity | None:
    pid = lock.get("worker_pid")
    ticks = lock.get("worker_process_created_at")
    if pid is None and ticks is None:
        return None
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(ticks, bool)
        or not isinstance(ticks, int)
        or ticks <= 0
    ):
        raise TaskSubmissionError("任务锁中的 Worker 进程信息损坏")
    return ProcessIdentity(pid=pid, created_at_ticks=ticks)


def _outcome_from_lock(action: TaskAction, lock: dict[str, object]) -> SubmissionOutcome:
    required = (
        "task_id",
        "status_relative_path",
        "worker_stdout_relative_path",
        "worker_stderr_relative_path",
    )
    if any(not isinstance(lock.get(key), str) or not lock[key] for key in required):
        raise TaskSubmissionError("任务锁缺少必要字段")
    return SubmissionOutcome(
        action=action,
        task_id=str(lock["task_id"]),
        status_relative_path=str(lock["status_relative_path"]),
        worker_stdout_relative_path=str(lock["worker_stdout_relative_path"]),
        worker_stderr_relative_path=str(lock["worker_stderr_relative_path"]),
    )


def _read_lock(
    settings: IncrementalSettings,
    *,
    retry_for_initial_write: bool = False,
) -> dict[str, object]:
    path = _lock_dir(settings) / "lock.json"
    attempts = 20 if retry_for_initial_write else 1
    for attempt in range(attempts):
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise TaskSubmissionError("任务锁内容不是 JSON 对象")
            return value
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                break
            time.sleep(0.01)
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskSubmissionError("无法读取任务锁") from exc
    raise TaskSubmissionError("任务锁尚未完成初始化")


def _write_lock(settings: IncrementalSettings, value: dict[str, object]) -> None:
    write_checkpoint(_lock_dir(settings) / "lock.json", value)


def _lock_dir(settings: IncrementalSettings) -> Path:
    path = settings.locks_dir / "active_task.lock"
    try:
        path.relative_to(settings.locks_dir)
    except ValueError as exc:
        raise TaskSubmissionError("任务锁路径不安全") from exc
    return path


def _remove_lock_dir(settings: IncrementalSettings) -> None:
    lock_dir = _lock_dir(settings)
    if lock_dir.exists():
        shutil.rmtree(lock_dir)


def _relative(settings: IncrementalSettings, path: Path) -> str:
    return path.relative_to(settings.project_root).as_posix()


def _new_task_id() -> str:
    now = datetime.now(timezone.utc).astimezone()
    return f"ingest-{now:%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
