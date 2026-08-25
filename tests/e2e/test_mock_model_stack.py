from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

from qwenrag_runtime.deployment import DeploymentConfig, SecretsConfig, default_deployment
from qwenrag_runtime.kb_initializer import initialize_empty_knowledge_base
from qwenrag_runtime.paths import RuntimePaths
from qwenrag_runtime.process_supervisor import ProcessSupervisor


ROOT = Path(__file__).resolve().parents[2]
FAKE_SERVER = Path(__file__).with_name("fake_model_server.py")


class FakeJob:
    def __init__(self) -> None:
        self.processes: list[subprocess.Popen[bytes]] = []
        self.closed = False

    def assign_process(self, process: subprocess.Popen[bytes]) -> None:
        self.processes.append(process)

    def close(self) -> None:
        self.closed = True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_ready(port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise AssertionError(f"fake model server on port {port} did not become ready")


def _deployment(llm_port: int, embedding_port: int, gateway_port: int, rag_port: int) -> DeploymentConfig:
    payload = default_deployment().model_dump(mode="json")
    payload["ports"] = {"llm": llm_port, "embedding": embedding_port, "gateway": gateway_port, "rag": rag_port}
    payload["llm"].update(
        {
            "base_url": f"http://127.0.0.1:{llm_port}/v1",
            "ready_url": f"http://127.0.0.1:{llm_port}/health",
            "expected_model": "test-llm",
            "executable": str(Path(sys.executable).resolve()),
            "working_directory": str(ROOT),
            "arguments": [str(FAKE_SERVER), "--kind", "llm", "--port", str(llm_port), "--model", "test-llm"],
            "startup_timeout_seconds": 10,
        }
    )
    payload["embedding"].update(
        {
            "base_url": f"http://127.0.0.1:{embedding_port}/v1",
            "ready_url": f"http://127.0.0.1:{embedding_port}/health",
            "expected_model": "test-embedding",
            "expected_revision": "test-artifact-v1",
            "expected_dimension": 8,
            "executable": str(Path(sys.executable).resolve()),
            "working_directory": str(ROOT),
            "arguments": [str(FAKE_SERVER), "--kind", "embedding", "--port", str(embedding_port), "--model", "test-embedding", "--dimension", "8"],
            "startup_timeout_seconds": 10,
        }
    )
    payload["rag"] = {"model_name": "local-rag", "embedding_dimension": 8, "llm_context_window": 8192}
    payload["runtime"] = {"gateway_startup_timeout_seconds": 10, "rag_startup_timeout_seconds": 10, "graceful_shutdown_timeout_seconds": 2}
    return DeploymentConfig.model_validate(payload)


def test_full_stack_starts_against_local_fake_models_and_stops_cleanly(tmp_path: Path) -> None:
    ports = [_free_port() for _ in range(4)]
    llm_port, embedding_port, gateway_port, rag_port = ports
    install_root = tmp_path / "installed runtime"
    install_root.mkdir()
    paths = RuntimePaths.resolve(
        environ={"LOCALAPPDATA": str(tmp_path)},
        frozen=True,
        executable=install_root / "QwenRagRuntime.exe",
    )
    deployment = _deployment(*ports)
    initialize_empty_knowledge_base(
        paths.knowledge_base_root,
        embedding_model=deployment.embedding.expected_model,
        embedding_revision=deployment.embedding.expected_revision,
        embedding_dimension=deployment.embedding.expected_dimension,
    )
    bootstrap = tmp_path / "runtime_bootstrap.py"
    bootstrap.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from qwenrag_runtime.cli import main\n"
        "raise SystemExit(main())\n",
        encoding="utf-8",
    )
    job = FakeJob()
    supervisor = ProcessSupervisor(
        deployment,
        SecretsConfig(local_rag_api_key="local-test-key", gateway_api_key="gateway-test-key"),
        paths,
        runtime_command=[sys.executable, str(bootstrap)],
        job_factory=lambda: job,
        status_writer=lambda _message: None,
    )

    try:
        ownership = supervisor.start()
        assert ownership == {"llm": "started", "embedding": "started", "gateway": "started", "rag": "started"}
        response = httpx.get(
            f"http://127.0.0.1:{rag_port}/v1/models",
            headers={"Authorization": "Bearer local-test-key"},
            timeout=5,
        )
        assert response.status_code == 200
        assert response.json()["data"]
        chat = httpx.post(
            f"http://127.0.0.1:{rag_port}/v1/chat/completions",
            headers={"Authorization": "Bearer local-test-key"},
            json={"model": "local-rag", "messages": [{"role": "user", "content": "hello"}], "stream": False},
            timeout=15,
        )
        assert chat.status_code == 200
        assert chat.json().get("choices")
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{rag_port}/v1/chat/completions",
            headers={"Authorization": "Bearer local-test-key"},
            json={"model": "local-rag", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            timeout=15,
        ) as stream:
            assert stream.status_code == 200
            assert any(line.strip() == "data: [DONE]" for line in stream.iter_lines())
    finally:
        supervisor.stop()

    assert job.closed is True
    assert all(process.poll() is not None for process in job.processes)
