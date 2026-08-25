from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from qwenrag_runtime.deployment import DeploymentConfig, SecretsConfig, default_deployment
from qwenrag_runtime.model_contracts import ModelContractError
from qwenrag_runtime.process_supervisor import (
    RuntimeAlreadyRunningError,
    RuntimeLaunchError,
    SupervisorState,
    ProcessSupervisor,
)
from qwenrag_runtime.runtime_lock import RuntimeLock
from qwenrag_runtime.paths import RuntimePaths


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9


class FakeJob:
    def __init__(self) -> None:
        self.assigned: list[FakeProcess] = []
        self.closed = False

    def assign_process(self, process: FakeProcess) -> None:
        self.assigned.append(process)

    def close(self) -> None:
        self.closed = True


class RejectingJob(FakeJob):
    def assign_process(self, process: FakeProcess) -> None:
        raise RuntimeError("job assignment failed")


class UnavailableJob:
    def __init__(self) -> None:
        raise RuntimeError("nested Windows Job is unavailable")


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths.resolve(environ={"LOCALAPPDATA": str(tmp_path)}, frozen=True)


def _supervisor(tmp_path: Path, **overrides):
    paths = _paths(tmp_path)
    executable_dir = tmp_path / "模型 空格"
    executable_dir.mkdir()
    executable = executable_dir / "模型服务.exe"
    executable.write_bytes(b"test")
    payload = default_deployment().model_dump(mode="json")
    for name in ("llm", "embedding"):
        payload[name]["executable"] = str(executable)
        payload[name]["working_directory"] = str(executable_dir)
        payload[name]["arguments"] = ["--model", "含 空格.gguf"]
    deployment = DeploymentConfig.model_validate(payload)
    processes: list[tuple[list[str], dict[str, str], FakeProcess]] = []
    job = FakeJob()

    def popen(command, *, cwd, env, stdout, stderr, shell):
        assert shell is False
        process = FakeProcess()
        processes.append((list(command), dict(env), process))
        return process

    values = {
        "deployment": deployment,
        "secret_values": SecretsConfig(local_rag_api_key="local", gateway_api_key="gateway"),
        "paths": paths,
        "runtime_command": ["runtime.exe"],
        "popen_factory": popen,
        "job_factory": lambda: job,
        "port_in_use": lambda _service: False,
        "model_checker": lambda _kind, _full: None,
        "http_ready": lambda _url: True,
        "kb_validator": lambda: None,
        "sleep": lambda _seconds: None,
        "status_writer": lambda _message: None,
    }
    values.update(overrides)
    return ProcessSupervisor(**values), processes, job, deployment


def test_starts_components_in_order_with_unmodified_argument_arrays(tmp_path: Path) -> None:
    supervisor, processes, job, deployment = _supervisor(tmp_path)

    result = supervisor.start()

    assert result["llm"] == "started"
    assert result["embedding"] == "started"
    assert result["gateway"] == "started"
    assert result["rag"] == "started"
    assert supervisor.state is SupervisorState.READY
    assert processes[0][0] == [str(deployment.llm.executable), *deployment.llm.arguments]
    assert processes[1][0] == [str(deployment.embedding.executable), *deployment.embedding.arguments]
    assert processes[2][0] == ["runtime.exe", "serve-gateway"]
    assert processes[3][0] == ["runtime.exe", "serve-rag"]
    assert processes[2][1]["GATEWAY_API_KEYS"] == "gateway"
    assert processes[3][1]["LOCAL_RAG_API_KEYS"] == "local"
    assert processes[3][1]["LOCAL_RAG_ANSWER_MODE"] == "gateway"
    assert processes[3][1]["ENABLE_RAG_ROUTER"] == "true"
    assert processes[3][1]["ENABLE_LOCAL_RETRIEVAL"] == "true"
    assert processes[3][1]["ENABLE_RAG_ANSWER_GENERATION"] == "true"
    assert processes[3][1]["ENABLE_REFERENCE_DISPLAY"] == "true"
    assert processes[3][1]["UPSTREAM_EMBEDDING_REVISION"] == deployment.embedding.expected_revision
    assert len(job.assigned) == 4
    assert "READY" in (supervisor._paths.log_root / "supervisor" / "supervisor.log").read_text(encoding="utf-8")
    assert (supervisor._paths.log_root / "gateway" / "gateway.stdout.log").is_file()
    assert (supervisor._paths.log_root / "gateway" / "gateway.stderr.log").is_file()
    assert (supervisor._paths.log_root / "rag" / "rag.stdout.log").is_file()
    assert (supervisor._paths.log_root / "rag" / "rag.stderr.log").is_file()

    supervisor.stop()
    assert all(process.terminated for _, _, process in processes)
    assert job.closed is True


def test_occupied_wrong_model_port_is_not_started_or_terminated(tmp_path: Path) -> None:
    def wrong_model(_kind: str, _full: bool) -> None:
        raise ModelContractError("model_mismatch", "wrong model")

    supervisor, processes, job, _ = _supervisor(
        tmp_path,
        port_in_use=lambda service: service == "llm",
        model_checker=wrong_model,
    )

    with pytest.raises(RuntimeLaunchError) as error:
        supervisor.start()

    assert error.value.exit_code == 30
    assert processes == []
    assert job.closed is True


def test_job_assignment_failure_terminates_the_just_created_process(tmp_path: Path) -> None:
    job = RejectingJob()
    supervisor, processes, _, _ = _supervisor(tmp_path, job_factory=lambda: job)

    with pytest.raises(RuntimeLaunchError) as error:
        supervisor.start()

    assert error.value.exit_code == 30
    assert len(processes) == 1
    assert processes[0][2].terminated is True


def test_job_creation_failure_uses_explicit_child_cleanup_fallback(tmp_path: Path) -> None:
    supervisor, processes, _, _ = _supervisor(tmp_path, job_factory=UnavailableJob)

    result = supervisor.start()

    assert result["gateway"] == "started"
    assert result["rag"] == "started"
    assert len(processes) == 4
    assert "Windows 进程监督器不可用，已启用兼容停止机制" in (
        supervisor._paths.log_root / "supervisor" / "supervisor.log"
    ).read_text(encoding="utf-8")

    supervisor.stop()

    assert all(process.terminated for _, _, process in processes)


def test_unexpected_startup_error_records_safe_diagnostic(tmp_path: Path) -> None:
    def broken_port_probe(_service: str) -> bool:
        raise PermissionError(5, "access denied")

    supervisor, _, _, _ = _supervisor(tmp_path, port_in_use=broken_port_probe)

    with pytest.raises(RuntimeLaunchError) as error:
        supervisor.start()

    assert error.value.exit_code == 70
    log = (supervisor._paths.log_root / "supervisor" / "supervisor.log").read_text(encoding="utf-8")
    assert "启动器内部错误：PermissionError Windows 错误码=5。" in log


def test_reuses_verified_model_services_and_never_stops_them(tmp_path: Path) -> None:
    supervisor, processes, job, _ = _supervisor(
        tmp_path,
        port_in_use=lambda service: service in {"llm", "embedding"},
    )

    result = supervisor.start()
    supervisor.stop()

    assert result["llm"] == "reused"
    assert result["embedding"] == "reused"
    assert len(processes) == 2
    assert len(job.assigned) == 2
    assert all(process.terminated for _, _, process in processes)


def test_start_failure_cleans_only_previously_created_processes(tmp_path: Path) -> None:
    calls = 0

    def ready(url: str) -> bool:
        nonlocal calls
        calls += 1
        return "8010" not in url

    ticks = itertools.count(0.0, 31.0)
    supervisor, processes, job, _ = _supervisor(
        tmp_path,
        http_ready=ready,
        runtime_command=["runtime.exe"],
        clock=lambda: next(ticks),
    )

    with pytest.raises(RuntimeLaunchError) as error:
        supervisor.start()

    assert error.value.exit_code == 40
    assert supervisor.state is SupervisorState.FAILED
    assert len(processes) == 3
    assert all(process.terminated for _, _, process in processes)
    assert job.closed is True


def test_runtime_lock_allows_one_holder_only(tmp_path: Path) -> None:
    path = _paths(tmp_path).runtime_root / "locks" / "rag.lock"
    first = RuntimeLock(path, mode="rag")
    second = RuntimeLock(path, mode="rag")

    first.acquire()
    with pytest.raises(RuntimeAlreadyRunningError):
        second.acquire()
    first.release()
    second.acquire()
    second.release()


def test_state_marker_lock_does_not_stop_runtime_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    supervisor, _, _, _ = _supervisor(tmp_path)

    def deny_state_replace(_source: Path, _target: Path) -> None:
        raise PermissionError("temporary scanner lock")

    monkeypatch.setattr("qwenrag_runtime.process_supervisor.os.replace", deny_state_replace)

    result = supervisor.start()

    assert result["gateway"] == "started"
    assert supervisor.state is SupervisorState.READY
    supervisor.stop()
