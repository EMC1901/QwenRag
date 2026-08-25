"""Ordered, ownership-aware startup and shutdown for the local QwenRAG stack."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Callable, Literal, Mapping, Protocol, Sequence
from urllib.parse import urlsplit
import uuid

import httpx

from .deployment import DeploymentConfig, SecretsConfig, derive_process_environment
from .model_contracts import ModelContractChecker, ModelContractError
from .paths import RuntimePaths
from .runtime_lock import RuntimeAlreadyRunningError, RuntimeLock
from .windows_job import WindowsJob


class SupervisorState(str, Enum):
    LOAD_CONFIG = "LOAD_CONFIG"
    VALIDATE_CONFIG = "VALIDATE_CONFIG"
    ACQUIRE_RUNTIME_LOCK = "ACQUIRE_RUNTIME_LOCK"
    CHECK_LLM = "CHECK_LLM"
    START_LLM = "START_LLM"
    WAIT_LLM_READY = "WAIT_LLM_READY"
    CHECK_EMBEDDING = "CHECK_EMBEDDING"
    START_EMBEDDING = "START_EMBEDDING"
    WAIT_EMBEDDING_READY = "WAIT_EMBEDDING_READY"
    START_GATEWAY = "START_GATEWAY"
    WAIT_GATEWAY_READY = "WAIT_GATEWAY_READY"
    VALIDATE_KB = "VALIDATE_KB"
    START_RAG = "START_RAG"
    WAIT_RAG_READY = "WAIT_RAG_READY"
    READY = "READY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class RuntimeLaunchError(RuntimeError):
    """Safe, Chinese startup failure associated with a documented exit code."""

    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ManagedProcess(Protocol):
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


ServiceName = Literal["llm", "embedding", "gateway", "rag"]


class ProcessSupervisor:
    """Start only missing services, and own only processes started by this run."""

    def __init__(
        self,
        deployment: DeploymentConfig,
        secret_values: SecretsConfig,
        paths: RuntimePaths,
        *,
        runtime_command: Sequence[str] | None = None,
        popen_factory: Callable[..., ManagedProcess] = subprocess.Popen,
        job_factory: Callable[[], WindowsJob] = WindowsJob,
        port_in_use: Callable[[ServiceName], bool] | None = None,
        model_checker: Callable[[Literal["llm", "embedding"], bool], None] | None = None,
        http_ready: Callable[[str], bool] | None = None,
        kb_validator: Callable[[], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        status_writer: Callable[[str], None] = print,
    ) -> None:
        self._deployment = deployment
        self._secrets = secret_values
        self._paths = paths
        self._runtime_command = tuple(runtime_command or _self_runtime_command())
        self._popen = popen_factory
        self._job_factory = job_factory
        self._port_in_use = port_in_use or self._default_port_in_use
        self._model_checker = model_checker or self._default_model_checker
        self._http_ready = http_ready or _http_ready
        self._kb_validator = kb_validator or self._default_kb_validator
        self._sleep = sleep
        self._clock = clock
        self._status_writer = status_writer
        self._state = SupervisorState.LOAD_CONFIG
        self._lock = RuntimeLock(paths.runtime_root / "locks" / "rag.lock", mode="rag")
        self._job: WindowsJob | None = None
        self._processes: dict[ServiceName, ManagedProcess] = {}
        self._ownership: dict[ServiceName, Literal["started", "reused"]] = {}

    @property
    def state(self) -> SupervisorState:
        return self._state

    def start(self) -> Mapping[ServiceName, Literal["started", "reused"]]:
        """Bring the stack to READY or clean up every process created so far."""
        try:
            self._paths.ensure_mutable_directories()
            self._transition(SupervisorState.LOAD_CONFIG, "正在读取部署配置……")
            self._transition(SupervisorState.VALIDATE_CONFIG, "正在校验部署配置……")
            self._transition(SupervisorState.ACQUIRE_RUNTIME_LOCK, "正在获取运行锁……")
            try:
                self._lock.acquire()
            except RuntimeAlreadyRunningError as exc:
                raise RuntimeLaunchError(10, "已有另一个 QwenRAG 会话正在运行。") from exc
            try:
                self._job = self._job_factory()
            except Exception as exc:
                # A frozen executable may itself be hosted in a restricted
                # Windows job.  Do not prevent a customer from starting RAG
                # just because a nested kill-on-close job is unavailable.
                # The supervisor still retains every child handle and its
                # stop/failure paths explicitly terminate only children it
                # created; reused model services (including SSH tunnels) are
                # never touched.
                self._job = None
                self._emit(
                    "Windows 进程监督器不可用，已启用兼容停止机制"
                    f"（{type(exc).__name__}）。"
                )

            self._ensure_model("llm", 30)
            self._ensure_model("embedding", 31)
            self._start_internal("gateway", self._gateway_environment(), 40)
            self._wait_http(
                "模型网关", f"http://127.0.0.1:{self._deployment.ports.gateway}/health",
                self._deployment.runtime.gateway_startup_timeout_seconds, 40,
                SupervisorState.WAIT_GATEWAY_READY,
            )
            self._transition(SupervisorState.VALIDATE_KB, "正在校验本地知识库……")
            try:
                self._kb_validator()
            except Exception as exc:
                raise RuntimeLaunchError(50, "本地知识库校验失败，请查看日志。") from exc
            self._start_internal("rag", self._rag_environment(), 60)
            self._wait_http(
                "QwenRAG", f"http://127.0.0.1:{self._deployment.ports.rag}/health/ready",
                self._deployment.runtime.rag_startup_timeout_seconds, 60,
                SupervisorState.WAIT_RAG_READY,
            )
            self._transition(SupervisorState.READY, "QwenRAG 已就绪，可打开 Chatbox。")
            return dict(self._ownership)
        except RuntimeLaunchError:
            self._fail_cleanup()
            raise
        except Exception as exc:
            self._emit(_unexpected_startup_diagnostic(exc))
            self._fail_cleanup()
            raise RuntimeLaunchError(70, "启动器发生未预期错误，请查看日志。") from exc

    def wait_until_stopped(self) -> None:
        """Keep the console-owned supervisor alive until Ctrl+C or child failure."""
        try:
            while True:
                for service, process in self._processes.items():
                    if process.poll() is not None:
                        raise RuntimeLaunchError(70, f"{_service_label(service)} 意外退出。")
                self._sleep(1)
        except KeyboardInterrupt:
            self.stop()
        except RuntimeLaunchError:
            self._fail_cleanup()
            raise

    def stop(self) -> None:
        """Gracefully end local services, then close the Job as a final guarantee."""
        if self._state in {SupervisorState.STOPPING, SupervisorState.STOPPED}:
            return
        self._transition(SupervisorState.STOPPING, "正在停止 QwenRAG……")
        self._terminate_owned("rag")
        self._terminate_owned("gateway")
        if self._job is not None:
            self._job.close()
            self._job = None
        # The non-Windows test fallback has no kernel Job. This is also a safe
        # final fallback when a child refused to exit before Job closure.
        self._terminate_owned("embedding")
        self._terminate_owned("llm")
        self._lock.release()
        self._transition(SupervisorState.STOPPED, "QwenRAG 已停止。")

    def _ensure_model(self, service: Literal["llm", "embedding"], exit_code: int) -> None:
        check_state = SupervisorState.CHECK_LLM if service == "llm" else SupervisorState.CHECK_EMBEDDING
        start_state = SupervisorState.START_LLM if service == "llm" else SupervisorState.START_EMBEDDING
        wait_state = SupervisorState.WAIT_LLM_READY if service == "llm" else SupervisorState.WAIT_EMBEDDING_READY
        self._transition(check_state, f"正在检查 {_service_label(service)} 服务……")
        if self._port_in_use(service):
            try:
                self._model_checker(service, True)
            except ModelContractError as exc:
                raise RuntimeLaunchError(exit_code, f"{_service_label(service)} 端口已被不匹配的服务占用。") from exc
            self._ownership[service] = "reused"
            self._emit(f"{_service_label(service)} 已在运行，已安全复用。")
            return
        self._transition(start_state, f"正在启动 {_service_label(service)} 服务……")
        environment = self._base_environment()
        configuration = self._deployment.llm if service == "llm" else self._deployment.embedding
        self._start_external(service, configuration.executable, configuration.arguments, configuration.working_directory, environment, exit_code)
        self._wait_model(service, configuration.startup_timeout_seconds, exit_code, wait_state)

    def _start_external(
        self,
        service: Literal["llm", "embedding"],
        executable: Path,
        arguments: list[str],
        working_directory: Path,
        environment: Mapping[str, str],
        exit_code: int,
    ) -> None:
        if not executable.is_file() or not working_directory.is_dir():
            raise RuntimeLaunchError(exit_code, f"{_service_label(service)} 程序路径或工作目录不存在。")
        self._spawn(service, [str(executable), *arguments], working_directory, environment, exit_code)

    def _start_internal(self, service: Literal["gateway", "rag"], environment: Mapping[str, str], exit_code: int) -> None:
        state = SupervisorState.START_GATEWAY if service == "gateway" else SupervisorState.START_RAG
        self._transition(state, f"正在启动 {_service_label(service)}……")
        command = [*self._runtime_command, "serve-gateway" if service == "gateway" else "serve-rag"]
        self._spawn(service, command, self._paths.install_root, environment, exit_code)

    def _spawn(self, service: ServiceName, command: list[str], cwd: Path, environment: Mapping[str, str], exit_code: int) -> None:
        # Keep each internal service's startup stderr/stdout alongside its
        # rotating application log, so a support engineer has one component
        # directory to collect. External model-process output remains under
        # supervisor because it is not produced by QwenRAG itself.
        log_component = service if service in {"gateway", "rag"} else "supervisor"
        log_dir = self._paths.log_root / log_component
        log_dir.mkdir(parents=True, exist_ok=True)
        process: ManagedProcess | None = None
        try:
            with (log_dir / f"{service}.stdout.log").open("a", encoding="utf-8") as stdout, (log_dir / f"{service}.stderr.log").open("a", encoding="utf-8") as stderr:
                process = self._popen(command, cwd=str(cwd), env=dict(environment), stdout=stdout, stderr=stderr, shell=False)
            if self._job is not None:
                self._job.assign_process(process)
        except RuntimeLaunchError:
            if process is not None:
                _terminate_process(process, self._deployment.runtime.graceful_shutdown_timeout_seconds)
            raise
        except Exception as exc:
            if process is not None:
                _terminate_process(process, self._deployment.runtime.graceful_shutdown_timeout_seconds)
            raise RuntimeLaunchError(exit_code, f"无法启动 {_service_label(service)}。") from exc
        self._processes[service] = process
        self._ownership[service] = "started"

    def _wait_model(self, service: Literal["llm", "embedding"], timeout_seconds: int, exit_code: int, state: SupervisorState) -> None:
        self._transition(state, f"正在等待 {_service_label(service)} 就绪……")
        started = self._clock()
        deadline = started + timeout_seconds
        next_report = started + 5
        while self._clock() <= deadline:
            process = self._processes.get(service)
            if process is not None and process.poll() is not None:
                break
            try:
                self._model_checker(service, True)
                return
            except ModelContractError:
                now = self._clock()
                if now >= next_report:
                    self._emit(f"正在等待 {_service_label(service)}，已等待 {int(now - started)} 秒……")
                    next_report = now + 5
                self._sleep(1)
        raise RuntimeLaunchError(exit_code, f"{_service_label(service)} 启动或契约检查失败。")

    def _wait_http(self, label: str, url: str, timeout_seconds: int, exit_code: int, state: SupervisorState) -> None:
        self._transition(state, f"正在等待 {label} 就绪……")
        started = self._clock()
        deadline = started + timeout_seconds
        next_report = started + 5
        while self._clock() <= deadline:
            if self._http_ready(url):
                return
            now = self._clock()
            if now >= next_report:
                self._emit(f"正在等待 {label}，已等待 {int(now - started)} 秒……")
                next_report = now + 5
            self._sleep(1)
        raise RuntimeLaunchError(exit_code, f"{label} 启动失败或等待超时。")

    def _base_environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "QWENRAG_DATA_ROOT": str(self._paths.data_root),
            "QWENRAG_CONFIG_ROOT": str(self._paths.config_root),
            "QWENRAG_LOG_ROOT": str(self._paths.log_root),
            "QWENRAG_KB_ROOT": str(self._paths.knowledge_base_root),
            "QWENRAG_RESOURCE_ROOT": str(self._paths.resource_root),
        }

    def _gateway_environment(self) -> dict[str, str]:
        values = derive_process_environment(self._deployment, self._secrets, self._paths)["gateway"]
        return {**self._base_environment(), **values, "MODEL_GATEWAY_DISABLE_ENV_FILE": "true"}

    def _rag_environment(self) -> dict[str, str]:
        values = derive_process_environment(self._deployment, self._secrets, self._paths)["local_rag"]
        return {**self._base_environment(), **values, "LOCAL_RAG_DISABLE_ENV_FILE": "true"}

    def _default_port_in_use(self, service: ServiceName) -> bool:
        port = getattr(self._deployment.ports, service)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            return probe.connect_ex(("127.0.0.1", port)) == 0

    def _default_model_checker(self, service: Literal["llm", "embedding"], full: bool) -> None:
        checker = ModelContractChecker(self._deployment, self._secrets)
        try:
            if service == "llm":
                checker.check_llm(full=full)
            else:
                checker.check_embedding(full=full)
        finally:
            checker.close()

    def _default_kb_validator(self) -> None:
        from local_rag_app.config import Settings
        from local_rag_app.knowledge_base import KnowledgeBase

        settings = Settings(
            _env_file=None,
            RAG_KNOWLEDGE_BASE_DIR=self._paths.knowledge_base_root,
            UPSTREAM_EMBEDDING_MODEL=self._deployment.embedding.expected_model,
            UPSTREAM_EMBEDDING_REVISION=self._deployment.embedding.expected_revision,
            RAG_EMBEDDING_DIM=self._deployment.rag.embedding_dimension,
        )
        knowledge_base = KnowledgeBase(settings)
        knowledge_base.load()
        knowledge_base.close()

    def _terminate_owned(self, service: ServiceName) -> None:
        process = self._processes.get(service)
        if process is None or self._ownership.get(service) != "started" or process.poll() is not None:
            return
        _terminate_process(process, self._deployment.runtime.graceful_shutdown_timeout_seconds)

    def _fail_cleanup(self) -> None:
        self._state = SupervisorState.FAILED
        self._emit("启动失败，正在清理本次启动的服务……")
        self.stop()
        self._state = SupervisorState.FAILED

    def _transition(self, state: SupervisorState, message: str) -> None:
        self._state = state
        self._emit(message)
        self._write_state_file(state)

    def _write_state_file(self, state: SupervisorState) -> None:
        """Best-effort state marker; diagnostics must never stop QwenRAG.

        Windows antivirus/indexing can briefly hold the previous JSON file.
        The console and rotating log still contain the state, so a persistent
        marker-write failure is deliberately non-fatal after small retries.
        """
        state_dir = self._paths.runtime_root / "state"
        temporary: Path | None = None
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            target = state_dir / "supervisor-state.json"
            temporary = state_dir / f".supervisor-state-{uuid.uuid4().hex}.tmp"
            temporary.write_text(
                json.dumps(
                    {"state": state.value, "updated_at": datetime.now(timezone.utc).isoformat()},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            for attempt in range(3):
                try:
                    os.replace(temporary, target)
                    return
                except PermissionError:
                    if attempt == 2:
                        return
                    time.sleep(0.05)
        except OSError:
            return
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _emit(self, message: str) -> None:
        self._status_writer(message)
        try:
            log_dir = self._paths.log_root / "supervisor"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).isoformat()
            with (log_dir / "supervisor.log").open("a", encoding="utf-8") as log_file:
                log_file.write(f"{timestamp} {self._state.value} {message}\n")
        except OSError:
            # Console output remains available when a customer disk is full.
            pass


def _self_runtime_command() -> tuple[str, ...]:
    return (sys.executable,) if getattr(sys, "frozen", False) else (sys.executable, "-m", "qwenrag_runtime")


def _http_ready(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def _service_label(service: ServiceName) -> str:
    return {"llm": "Qwen LLM", "embedding": "Embedding 模型", "gateway": "模型网关", "rag": "QwenRAG"}[service]


def _terminate_process(process: ManagedProcess, timeout_seconds: int) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=timeout_seconds)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _unexpected_startup_diagnostic(exc: Exception) -> str:
    """Return a bounded, secret-free diagnostic for pre-service failures."""
    error_code = getattr(exc, "winerror", None)
    if not isinstance(error_code, int):
        error_code = getattr(exc, "errno", None)
    detail = f" Windows 错误码={error_code}" if isinstance(error_code, int) else ""
    return f"启动器内部错误：{type(exc).__name__}{detail}。"


__all__ = ["ProcessSupervisor", "RuntimeAlreadyRunningError", "RuntimeLaunchError", "SupervisorState"]
