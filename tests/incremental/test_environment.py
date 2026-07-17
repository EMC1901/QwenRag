"""Stage-1 offline environment-check tests."""

from __future__ import annotations

from pathlib import Path
import socket
from types import SimpleNamespace

from rag_preprocess.incremental.environment import check_incremental_environment
from rag_preprocess.incremental.settings import load_incremental_settings


def _available_dependency(_name: str) -> object:
    return SimpleNamespace(__version__="test")


def _settings(tmp_path: Path):
    return load_incremental_settings(
        project_root=tmp_path,
        environ={
            "INCREMENTAL_KB_ROOT": "data",
            "OCR_MODEL_DIR": "models/ocr",
            "OCR_TEXT_DETECTION_MODEL": "det",
            "OCR_TEXT_RECOGNITION_MODEL": "rec",
            "EMBEDDING_BASE_URL": "http://127.0.0.1:8002/v1",
            "EMBEDDING_MODEL": "fixture-embedding",
            "EMBEDDING_DIM": "3",
        },
    )


def test_environment_check_reports_missing_model_directory_with_recovery_action(
    tmp_path: Path,
) -> None:
    """Missing offline OCR models are a Chinese actionable environment error."""
    report = check_incremental_environment(
        _settings(tmp_path),
        module_importer=_available_dependency,
        module_finder=lambda _name: object(),
    )

    assert report.exit_code == 22
    assert any("OCR 模型目录不存在" in issue.message for issue in report.issues)
    assert any("放置已验证的离线 OCR 模型" in issue.remedy for issue in report.issues)


def test_environment_check_is_offline_and_succeeds_when_dependencies_and_models_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The checker imports and inspects locally; it never opens a network connection."""
    settings = _settings(tmp_path)
    (settings.ocr_model_dir / "det").mkdir(parents=True)
    (settings.ocr_model_dir / "rec").mkdir()

    def fail_network(*_args, **_kwargs):
        raise AssertionError("环境检查不得联网")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    report = check_incremental_environment(
        settings,
        module_importer=_available_dependency,
        module_finder=lambda _name: object(),
    )

    assert report.exit_code == 0
    assert report.available_free_bytes > 0
    assert settings.incoming_dir.is_dir()


def test_environment_check_never_imports_paddleocr(tmp_path: Path) -> None:
    """PaddleOCR import can probe model hosts, so preflight only finds its spec."""
    settings = _settings(tmp_path)
    (settings.ocr_model_dir / "det").mkdir(parents=True)
    (settings.ocr_model_dir / "rec").mkdir()
    imported: list[str] = []
    found: list[str] = []

    def importer(name: str) -> object:
        imported.append(name)
        return _available_dependency(name)

    def finder(name: str) -> object:
        found.append(name)
        return object()

    report = check_incremental_environment(
        settings,
        module_importer=importer,
        module_finder=finder,
    )

    assert report.exit_code == 0
    assert "paddleocr" not in imported
    assert "paddleocr" in found
