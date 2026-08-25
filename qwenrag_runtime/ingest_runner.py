"""Console-owned incremental ingestion with exclusive RAG access.

This module deliberately starts only the embedding service.  The LLM, gateway,
and local RAG process are neither needed nor allowed during an ingestion task.
"""

from __future__ import annotations

from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Callable, Sequence

# This has to be set before any optional PaddleOCR import.  The delivery is
# offline and OCR models are provided only through the external resources tree.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from rag_preprocess.incremental.persistence import write_checkpoint, write_status
from rag_preprocess.incremental.settings import load_incremental_settings
from rag_preprocess.incremental.task_submission import (
    TaskSubmissionError,
    claim_worker,
    release_task_lock,
    submit_task,
)
from rag_preprocess.incremental.workflow import run_stages_4_to_9

from .deployment import DeploymentConfig, SecretsConfig, derive_process_environment
from .model_contracts import ModelContractChecker, ModelContractError
from .paths import RuntimePaths
from .runtime_lock import RuntimeAlreadyRunningError, RuntimeLock


class IngestRuntimeError(RuntimeError):
    """A customer-safe ingestion launcher error."""


def submit_ingest_task(
    deployment: DeploymentConfig,
    secrets: SecretsConfig,
    paths: RuntimePaths,
    *,
    spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> dict[str, object]:
    """Reserve one task and start the same frozen executable as a hidden worker."""
    settings = _settings(deployment, secrets, paths)
    # Briefly take the same lock as RAG so task submission cannot race a RAG
    # session starting at exactly the same moment. The worker holds it for the
    # real duration of processing.
    lock = RuntimeLock(paths.runtime_root / "locks" / "rag.lock", mode="ingest")
    try:
        lock.acquire()
    except RuntimeAlreadyRunningError as exc:
        raise IngestRuntimeError("QwenRAG 正在运行，请关闭启动窗口后再开始资料入库") from exc
    try:
        outcome = submit_task(settings)
    finally:
        lock.release()
    result = outcome.as_dict()
    result["task_file"] = str(settings.work_dir / outcome.task_id / "task.json")
    if not outcome.should_start_worker:
        return result
    command = [sys.executable, "ingest-worker", "--task-id", outcome.task_id]
    workspace = settings.work_dir / outcome.task_id
    stdout = (workspace / "worker.stdout.log").open("a", encoding="utf-8")
    stderr = (workspace / "worker.stderr.log").open("a", encoding="utf-8")
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = spawn(
            command,
            cwd=str(paths.install_root),
            env=_base_environment(paths),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            creationflags=creationflags,
        )
    except Exception as exc:
        release_task_lock(settings, outcome.task_id)
        raise IngestRuntimeError("资料入库后台程序未能启动，任务锁已释放") from exc
    finally:
        stdout.close()
        stderr.close()
    result["worker_pid"] = getattr(process, "pid", None)
    return result


def run_ingest_worker(
    deployment: DeploymentConfig,
    secrets: SecretsConfig,
    paths: RuntimePaths,
    task_id: str,
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    checker_factory: Callable[..., ModelContractChecker] = ModelContractChecker,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run one task while holding the shared RAG/ingestion runtime lock."""
    settings = _settings(deployment, secrets, paths)
    lock = RuntimeLock(paths.runtime_root / "locks" / "rag.lock", mode="ingest")
    try:
        lock.acquire()
    except RuntimeAlreadyRunningError as exc:
        _write_worker_failure(settings, task_id, "RAG_RUNNING")
        raise IngestRuntimeError("QwenRAG 正在运行，已拒绝资料入库") from exc

    embedding_process: subprocess.Popen | None = None
    try:
        claim_worker(settings, task_id)
        embedding_process = _ensure_embedding(
            deployment, secrets, paths, popen=popen, checker_factory=checker_factory, sleep=sleep
        )
        outcome = run_stages_4_to_9(settings, task_id)
        return 0 if outcome.get("state") != "FAILED_RESUMABLE" else 31
    except (ModelContractError, TaskSubmissionError, IngestRuntimeError):
        _write_worker_failure(settings, task_id, "INGEST_PREFLIGHT_FAILED")
        return 31
    except Exception:
        _write_worker_failure(settings, task_id, "WORKER_RUNTIME_FAILED")
        return 31
    finally:
        if embedding_process is not None:
            _stop_owned(embedding_process, deployment.runtime.graceful_shutdown_timeout_seconds)
        release_task_lock(settings, task_id)
        lock.release()


def diagnose_runtime(paths: RuntimePaths) -> dict[str, object]:
    """Load native delivery dependencies and return only non-secret facts."""
    # Do real imports rather than find_spec: the latter cannot detect a missing
    # DLL in a frozen onedir distribution.
    modules = ("faiss", "fitz", "docx", "paddle", "paddleocr", "charset_normalizer")
    loaded: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise IngestRuntimeError(f"冻结运行时组件无法加载：{name}") from exc
        loaded.append(name)
    return {
        "frozen": paths.frozen,
        "python_executable": sys.executable,
        "install_root": str(paths.install_root),
        "resource_root": str(paths.resource_root),
        "data_root": str(paths.data_root),
        "ocr_resource_root": str(paths.ocr_resource_root),
        "loaded_components": loaded,
    }


def _settings(deployment: DeploymentConfig, secrets: SecretsConfig, paths: RuntimePaths):
    values = {**_base_environment(paths), **derive_process_environment(deployment, secrets, paths)["incremental"]}
    return load_incremental_settings(environ=values, runtime_paths=paths)


def _ensure_embedding(
    deployment: DeploymentConfig,
    secrets: SecretsConfig,
    paths: RuntimePaths,
    *,
    popen: Callable[..., subprocess.Popen],
    checker_factory: Callable[..., ModelContractChecker],
    sleep: Callable[[float], None],
) -> subprocess.Popen | None:
    """Verify a reused embedding process or start/own exactly one configured one."""
    if _port_in_use(deployment.ports.embedding):
        _check_embedding(deployment, secrets, checker_factory)
        return None
    service = deployment.embedding
    if not service.executable.is_file() or not service.working_directory.is_dir():
        raise IngestRuntimeError("Embedding 模型程序路径或工作目录不存在")
    log_dir = paths.log_root / "ingest"
    log_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        stdout = stack.enter_context((log_dir / "embedding.stdout.log").open("a", encoding="utf-8"))
        stderr = stack.enter_context((log_dir / "embedding.stderr.log").open("a", encoding="utf-8"))
        process = popen(
            [str(service.executable), *service.arguments],
            cwd=str(service.working_directory),
            env=_base_environment(paths),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    deadline = time.monotonic() + service.startup_timeout_seconds
    while time.monotonic() <= deadline:
        if process.poll() is not None:
            raise IngestRuntimeError("Embedding 模型启动后异常退出")
        try:
            _check_embedding(deployment, secrets, checker_factory)
            return process
        except ModelContractError:
            sleep(1)
    _stop_owned(process, deployment.runtime.graceful_shutdown_timeout_seconds)
    raise IngestRuntimeError("Embedding 模型启动或契约检查超时")


def _check_embedding(deployment: DeploymentConfig, secrets: SecretsConfig, checker_factory: Callable[..., ModelContractChecker]) -> None:
    checker = checker_factory(deployment, secrets)
    try:
        checker.check_embedding(full=True)
    finally:
        checker.close()


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _stop_owned(process: subprocess.Popen, timeout_seconds: int) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)


def _write_worker_failure(settings, task_id: str, code: str) -> None:
    task_path = settings.work_dir / task_id / "task.json"
    write_checkpoint(task_path, {"schema_version": 1, "task_id": task_id, "state": "FAILED_RESUMABLE", "error_code": code})
    write_status(settings.results_dir / f"{task_id}.status.txt", "处理未完成；正式知识库未因本次任务发布新资料。请联系技术支持人员。\n")


def _base_environment(paths: RuntimePaths) -> dict[str, str]:
    return {
        **os.environ,
        "QWENRAG_DATA_ROOT": str(paths.data_root),
        "QWENRAG_CONFIG_ROOT": str(paths.config_root),
        "QWENRAG_LOG_ROOT": str(paths.log_root),
        "QWENRAG_KB_ROOT": str(paths.knowledge_base_root),
        "QWENRAG_RESOURCE_ROOT": str(paths.resource_root),
    }
