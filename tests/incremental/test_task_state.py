"""Stage-3 task and file state transition tests."""

from __future__ import annotations

import pytest

from rag_preprocess.incremental.state import (
    FileState,
    InvalidStateTransition,
    TaskState,
    validate_file_transition,
    validate_task_transition,
)


def test_task_state_allows_normal_submission_to_processing_path() -> None:
    """A submitted task may progress through the declared production states."""
    validate_task_transition(TaskState.SUBMITTED, TaskState.PREFLIGHT)
    validate_task_transition(TaskState.PREFLIGHT, TaskState.SNAPSHOTTING)
    validate_task_transition(TaskState.SNAPSHOTTING, TaskState.PROCESSING_FILES)
    validate_task_transition(TaskState.PROCESSING_FILES, TaskState.BUILDING_DELTA_DB)


def test_task_state_rejects_skipping_to_publish() -> None:
    """Publish cannot occur before Delta construction and validation."""
    with pytest.raises(InvalidStateTransition):
        validate_task_transition(TaskState.SUBMITTED, TaskState.PUBLISHING)


def test_file_state_allows_publish_then_archive_and_rejects_invalid_jump() -> None:
    """File success ends at archive; unsupported input cannot be published."""
    validate_file_transition(FileState.PUBLISHED, FileState.ARCHIVED)
    with pytest.raises(InvalidStateTransition):
        validate_file_transition(FileState.UNSUPPORTED, FileState.PUBLISHED)
