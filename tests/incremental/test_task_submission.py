"""Stage-2 task submission, worker identity, and single-lock tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rag_preprocess.incremental.settings import load_incremental_settings
from rag_preprocess.incremental.task_submission import (
    ProcessIdentity,
    TaskAction,
    WorkerStartError,
    build_worker_command,
    claim_worker,
    release_task_lock,
    submit_and_launch,
    submit_task,
)


class _ProcessInspector:
    def __init__(self, *, current: ProcessIdentity, alive: bool = True) -> None:
        self.current = current
        self.alive = alive

    def current_identity(self) -> ProcessIdentity:
        return self.current

    def is_same_process(self, identity: ProcessIdentity) -> bool:
        return self.alive and identity == self.current


def _settings(tmp_path: Path):
    return load_incremental_settings(
        project_root=tmp_path,
        environ={"INCREMENTAL_KB_ROOT": "data", "OCR_MODEL_DIR": "models/ocr"},
    )


def test_concurrent_submit_creates_one_task_and_one_active_lock(tmp_path: Path) -> None:
    """Racing submit calls share the atomically created task instead of duplicating it."""
    settings = _settings(tmp_path)
    inspector = _ProcessInspector(current=ProcessIdentity(pid=100, created_at_ticks=1))

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                lambda _unused: submit_task(settings, process_inspector=inspector),
                range(2),
            )
        )

    assert {outcome.action for outcome in outcomes} == {TaskAction.CREATED, TaskAction.ACTIVE}
    assert len({outcome.task_id for outcome in outcomes}) == 1
    assert (settings.locks_dir / "active_task.lock" / "lock.json").is_file()


def test_second_submit_returns_existing_active_task(tmp_path: Path) -> None:
    """A worker with matching PID and creation time keeps its original task ID."""
    settings = _settings(tmp_path)
    identity = ProcessIdentity(pid=200, created_at_ticks=2)
    inspector = _ProcessInspector(current=identity)
    created = submit_task(settings, process_inspector=inspector)
    claim_worker(settings, created.task_id, process_inspector=inspector)

    repeated = submit_task(settings, process_inspector=inspector)

    assert created.action is TaskAction.CREATED
    assert repeated.action is TaskAction.ACTIVE
    assert repeated.task_id == created.task_id
    assert repeated.status_relative_path == created.status_relative_path


def test_dead_worker_pid_reuses_original_task_id_for_recovery(tmp_path: Path) -> None:
    """A stale PID never creates a second task and instead requests recovery."""
    settings = _settings(tmp_path)
    original = _ProcessInspector(current=ProcessIdentity(pid=300, created_at_ticks=3))
    created = submit_task(settings, process_inspector=original)
    claim_worker(settings, created.task_id, process_inspector=original)
    restarted = _ProcessInspector(current=ProcessIdentity(pid=301, created_at_ticks=4), alive=False)

    recovered = submit_task(settings, process_inspector=restarted)

    assert recovered.action is TaskAction.RECOVERY
    assert recovered.task_id == created.task_id


def test_worker_command_preserves_chinese_and_space_paths(tmp_path: Path) -> None:
    """Worker argument construction keeps script paths as one argument each."""
    project_root = tmp_path / "\u4e2d\u6587 \u7a7a\u683c"
    settings = _settings(project_root)
    outcome = submit_task(
        settings,
        process_inspector=_ProcessInspector(
            current=ProcessIdentity(pid=350, created_at_ticks=35)
        ),
    )
    script = project_root / "scripts" / "incremental_import.py"

    command = build_worker_command(
        python_executable=Path("C:/Python/python.exe"),
        script_path=script,
        task_id="ingest-20260716-120000-a1b2c3d4",
    )

    assert command == [
        str(Path("C:/Python/python.exe")),
        "-u",
        str(script),
        "worker",
        "--task-id",
        "ingest-20260716-120000-a1b2c3d4",
    ]
    assert (project_root / outcome.status_relative_path).is_file()


def test_worker_launch_failure_releases_lock_without_fake_active_task(tmp_path: Path) -> None:
    """A failed launcher removes the lock so later submission can safely retry."""
    settings = _settings(tmp_path)
    inspector = _ProcessInspector(current=ProcessIdentity(pid=400, created_at_ticks=5))

    def fail_launcher(_task_id: str) -> None:
        raise OSError("worker could not start")

    with pytest.raises(WorkerStartError):
        submit_and_launch(
            settings,
            launcher=fail_launcher,
            process_inspector=inspector,
        )

    assert not (settings.locks_dir / "active_task.lock").exists()
    retried = submit_task(settings, process_inspector=inspector)
    assert retried.action is TaskAction.CREATED


def test_terminal_worker_release_allows_a_fresh_task_id(tmp_path: Path) -> None:
    """A completed Worker must not cause the next batch to reuse its Delta ID."""
    settings = _settings(tmp_path)
    inspector = _ProcessInspector(current=ProcessIdentity(pid=410, created_at_ticks=6))
    first = submit_task(settings, process_inspector=inspector)
    claim_worker(settings, first.task_id, process_inspector=inspector)

    release_task_lock(settings, first.task_id)

    second = submit_task(settings, process_inspector=inspector)
    assert second.action is TaskAction.CREATED
    assert second.task_id != first.task_id
