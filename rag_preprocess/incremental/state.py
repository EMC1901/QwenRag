"""Explicit task and file state machines for incremental ingestion."""

from __future__ import annotations

from enum import Enum


class InvalidStateTransition(ValueError):
    """Raised when a checkpoint attempts an undeclared state transition."""


class TaskState(str, Enum):
    SUBMITTED = "SUBMITTED"
    PREFLIGHT = "PREFLIGHT"
    SNAPSHOTTING = "SNAPSHOTTING"
    PROCESSING_FILES = "PROCESSING_FILES"
    BUILDING_DELTA_DB = "BUILDING_DELTA_DB"
    BUILDING_DELTA_FTS = "BUILDING_DELTA_FTS"
    BUILDING_DELTA_FAISS = "BUILDING_DELTA_FAISS"
    VALIDATING_DELTA = "VALIDATING_DELTA"
    PUBLISHING = "PUBLISHING"
    ARCHIVING = "ARCHIVING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    NO_CHANGES = "NO_CHANGES"
    REJECTED_SERVICE_RUNNING = "REJECTED_SERVICE_RUNNING"
    FAILED_RESUMABLE = "FAILED_RESUMABLE"
    FAILED_FINAL = "FAILED_FINAL"


class FileState(str, Enum):
    FROZEN = "FROZEN"
    NOT_READY = "NOT_READY"
    UNSUPPORTED = "UNSUPPORTED"
    NEW = "NEW"
    UPDATE = "UPDATE"
    DUPLICATE_UNCHANGED = "DUPLICATE_UNCHANGED"
    PARSING = "PARSING"
    QUALITY_CHECK = "QUALITY_CHECK"
    STRUCTURING = "STRUCTURING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    READY_TO_DELTA = "READY_TO_DELTA"
    DELTA_BUILT = "DELTA_BUILT"
    DELTA_VALIDATED = "DELTA_VALIDATED"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"
    PUBLISHED_ARCHIVE_FAILED = "PUBLISHED_ARCHIVE_FAILED"


_TASK_NEXT: dict[TaskState, set[TaskState]] = {
    TaskState.SUBMITTED: {TaskState.PREFLIGHT},
    TaskState.PREFLIGHT: {TaskState.SNAPSHOTTING, TaskState.REJECTED_SERVICE_RUNNING},
    TaskState.SNAPSHOTTING: {TaskState.PROCESSING_FILES, TaskState.NO_CHANGES},
    TaskState.PROCESSING_FILES: {TaskState.BUILDING_DELTA_DB, TaskState.NO_CHANGES},
    TaskState.BUILDING_DELTA_DB: {TaskState.BUILDING_DELTA_FTS},
    TaskState.BUILDING_DELTA_FTS: {TaskState.BUILDING_DELTA_FAISS},
    TaskState.BUILDING_DELTA_FAISS: {TaskState.VALIDATING_DELTA},
    TaskState.VALIDATING_DELTA: {TaskState.PUBLISHING},
    TaskState.PUBLISHING: {TaskState.ARCHIVING},
    TaskState.ARCHIVING: {TaskState.SUCCEEDED, TaskState.PARTIAL_SUCCESS},
}
_TASK_FAILURE_STATES = {
    TaskState.REJECTED_SERVICE_RUNNING,
    TaskState.FAILED_RESUMABLE,
    TaskState.FAILED_FINAL,
}
_TASK_TERMINAL_STATES = {
    TaskState.SUCCEEDED,
    TaskState.PARTIAL_SUCCESS,
    TaskState.NO_CHANGES,
    *_TASK_FAILURE_STATES,
}

_FILE_NEXT: dict[FileState, set[FileState]] = {
    FileState.FROZEN: {
        FileState.NOT_READY,
        FileState.UNSUPPORTED,
        FileState.NEW,
        FileState.UPDATE,
        FileState.DUPLICATE_UNCHANGED,
    },
    FileState.NEW: {FileState.PARSING},
    FileState.UPDATE: {FileState.PARSING},
    FileState.PARSING: {FileState.QUALITY_CHECK},
    FileState.QUALITY_CHECK: {FileState.STRUCTURING},
    FileState.STRUCTURING: {FileState.CHUNKING},
    FileState.CHUNKING: {FileState.EMBEDDING},
    FileState.EMBEDDING: {FileState.READY_TO_DELTA},
    FileState.READY_TO_DELTA: {FileState.DELTA_BUILT},
    FileState.DELTA_BUILT: {FileState.DELTA_VALIDATED},
    FileState.DELTA_VALIDATED: {FileState.PUBLISHED},
    FileState.PUBLISHED: {FileState.ARCHIVED, FileState.PUBLISHED_ARCHIVE_FAILED},
}
_FILE_FAILURE_STATES = {FileState.FAILED}
_FILE_TERMINAL_STATES = {
    FileState.NOT_READY,
    FileState.UNSUPPORTED,
    FileState.DUPLICATE_UNCHANGED,
    FileState.ARCHIVED,
    FileState.FAILED,
    FileState.PUBLISHED_ARCHIVE_FAILED,
}


def validate_task_transition(current: TaskState, target: TaskState) -> None:
    """Raise unless a task checkpoint moves along an approved edge."""
    if current in _TASK_TERMINAL_STATES or target not in (_TASK_NEXT.get(current, set()) | _TASK_FAILURE_STATES):
        raise InvalidStateTransition(f"不允许任务状态从 {current.value} 变更为 {target.value}")


def validate_file_transition(current: FileState, target: FileState) -> None:
    """Raise unless a file checkpoint moves along an approved edge."""
    if current in _FILE_TERMINAL_STATES or target not in (_FILE_NEXT.get(current, set()) | _FILE_FAILURE_STATES):
        raise InvalidStateTransition(f"不允许文件状态从 {current.value} 变更为 {target.value}")
