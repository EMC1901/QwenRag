"""Stage-1 configuration boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_preprocess.incremental.settings import (
    IncrementalConfigurationError,
    load_incremental_settings,
)


def test_settings_resolve_relative_paths_and_create_only_incremental_directories(
    tmp_path: Path,
) -> None:
    """Relative configuration stays under the declared knowledge-base root."""
    settings = load_incremental_settings(
        project_root=tmp_path,
        environ={
            "INCREMENTAL_KB_ROOT": "data",
            "INCREMENTAL_ROOT": "data/incremental",
            "OCR_MODEL_DIR": "models/ocr",
        },
    )

    assert settings.knowledge_base_root == tmp_path / "data"
    assert settings.incoming_dir == tmp_path / "data/incremental/incoming"
    assert settings.locks_dir == tmp_path / "data/incremental/locks"
    assert settings.logs_dir == tmp_path / "data/incremental/logs"
    assert settings.deltas_dir == tmp_path / "data/kb_deltas"
    assert settings.manifest_path == tmp_path / "data/knowledge_manifest.json"
    assert settings.current_pointer == tmp_path / "data/current_generation.txt"
    assert settings.ocr_model_dir == tmp_path / "models/ocr"

    created = settings.ensure_directories()

    assert settings.incoming_dir in created
    assert settings.work_dir.is_dir()
    assert not settings.ocr_model_dir.exists()


def test_settings_reject_data_path_outside_knowledge_base_root(tmp_path: Path) -> None:
    """An incremental data path cannot escape to an arbitrary local directory."""
    with pytest.raises(IncrementalConfigurationError, match="知识库根目录"):
        load_incremental_settings(
            project_root=tmp_path,
            environ={
                "INCREMENTAL_KB_ROOT": "data",
                "INCREMENTAL_INCOMING_DIR": "../outside/incoming",
            },
        )


def test_settings_reject_model_path_outside_project_root(tmp_path: Path) -> None:
    """OCR model selection cannot use a path outside the deployed project."""
    with pytest.raises(IncrementalConfigurationError, match="项目目录"):
        load_incremental_settings(
            project_root=tmp_path,
            environ={"OCR_MODEL_DIR": "../outside-models"},
        )


def test_settings_parse_env_incremental_file_without_changing_process_environment(
    tmp_path: Path,
) -> None:
    """Deployment settings are read from .env.incremental with explicit overrides."""
    env_file = tmp_path / ".env.incremental"
    env_file.write_text(
        "INCREMENTAL_KB_ROOT=data\nOCR_MODEL_DIR=models/ocr\nOCR_CPU_THREADS=6\n",
        encoding="utf-8",
    )

    settings = load_incremental_settings(
        project_root=tmp_path,
        env_file=env_file,
        environ={},
    )

    assert settings.knowledge_base_root == tmp_path / "data"
    assert settings.ocr_cpu_threads == 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"OCR_LOW_LINE_CONFIDENCE": "1.01"},
        {"OCR_PAGE_WARNING_LOW_LINE_RATIO": "-0.01"},
        {"OCR_RETRY_DPI": "199", "OCR_RENDER_DPI": "200"},
        {
            "OCR_SEVERE_PAGE_CONFIDENCE": "0.90",
            "OCR_PAGE_WARNING_CONFIDENCE": "0.85",
        },
        {"OCR_CPU_THREADS": "0"},
    ],
)
def test_settings_reject_invalid_ocr_threshold_combinations(
    tmp_path: Path,
    overrides: dict[str, str],
) -> None:
    """OCR values are validated before any worker or OCR model is started."""
    with pytest.raises(IncrementalConfigurationError):
        load_incremental_settings(project_root=tmp_path, environ=overrides)
