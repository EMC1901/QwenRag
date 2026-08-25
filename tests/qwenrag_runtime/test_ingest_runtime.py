"""Installed ingestion must share the RAG lock and avoid Conda/Python launchers."""

from __future__ import annotations

import codecs
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwenrag_runtime import cli
from qwenrag_runtime import ingest_runner
from qwenrag_runtime.deployment import SecretsConfig, default_deployment
from qwenrag_runtime.paths import RuntimePaths
from qwenrag_runtime.runtime_lock import RuntimeLock


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths.resolve(environ={"LOCALAPPDATA": str(tmp_path)}, frozen=True)


def test_runtime_version_and_diagnostics_do_not_require_deployment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = _paths(tmp_path)

    assert cli.main(["version"], runtime_paths=paths) == 0
    assert capsys.readouterr().out.strip() == "1.0.0"
    assert cli.main(["diagnose-runtime"], runtime_paths=paths) == 0
    assert '"frozen": true' in capsys.readouterr().out


def test_ingest_worker_refuses_to_run_while_rag_owns_shared_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(tmp_path)
    settings = SimpleNamespace(work_dir=tmp_path / "work", results_dir=tmp_path / "results")
    monkeypatch.setattr(ingest_runner, "_settings", lambda *_: settings)
    lock = RuntimeLock(paths.runtime_root / "locks" / "rag.lock", mode="rag")
    lock.acquire()
    try:
        with pytest.raises(ingest_runner.IngestRuntimeError):
            ingest_runner.run_ingest_worker(default_deployment(), SecretsConfig.create(), paths, "task-1")
    finally:
        lock.release()

    assert (settings.work_dir / "task-1" / "task.json").is_file()
    assert (settings.results_dir / "task-1.status.txt").is_file()


def test_ingest_reuses_verified_embedding_without_owning_or_stopping_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class Checker:
        def check_embedding(self, *, full: bool) -> None:
            calls.append(f"check:{full}")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(ingest_runner, "_port_in_use", lambda _port: True)

    process = ingest_runner._ensure_embedding(
        default_deployment(),
        SecretsConfig.create(),
        _paths(tmp_path),
        popen=lambda **_kwargs: pytest.fail("a reused service must not be started"),
        checker_factory=lambda *_args: Checker(),
        sleep=lambda _seconds: None,
    )

    assert process is None
    assert calls == ["check:True", "close"]


def test_workbench_launcher_has_no_conda_or_system_python_dependency() -> None:
    launcher = (Path(__file__).resolve().parents[2] / "scripts" / "submit_incremental_import.ps1").read_text(encoding="utf-8").lower()

    assert "conda" not in launcher
    assert "incremental_rag" not in launcher
    assert "qwenragruntime.exe" in launcher


def test_start_launcher_is_ascii_safe_for_windows_powershell() -> None:
    launcher = (Path(__file__).resolve().parents[2] / "launch" / "Start-QwenRAG.ps1").read_text(encoding="utf-8")

    assert launcher.isascii()
    assert "QwenRagRuntime.exe" in launcher


def test_workbench_launcher_is_utf8_bom_encoded_for_windows_powershell() -> None:
    launcher = Path(__file__).resolve().parents[2] / "scripts" / "submit_incremental_import.ps1"

    assert launcher.read_bytes().startswith(codecs.BOM_UTF8)


def test_diagnostic_shortcut_uses_ascii_powershell_pause_text() -> None:
    installer = (Path(__file__).resolve().parents[2] / "packaging" / "installer" / "QwenRAG.iss").read_text(encoding="utf-8")

    diagnostic_line = next(line for line in installer.splitlines() if "config validate; Read-Host" in line)
    assert "Read-Host ''Press Enter to close''" in diagnostic_line


def test_pyinstaller_spec_explicitly_collects_native_runtime_packages() -> None:
    spec = (Path(__file__).resolve().parents[2] / "packaging" / "qwenrag_runtime.spec").read_text(encoding="utf-8")

    for package in ("faiss", "paddle", "paddleocr", "fitz", "charset_normalizer"):
        assert f'"{package}"' in spec
    assert "upx=False" in spec
    assert "OCR model files are intentionally not bundled" in spec
