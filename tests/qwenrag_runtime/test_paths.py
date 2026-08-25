"""Tests for safe source and installed-runtime path resolution."""

from __future__ import annotations

from pathlib import Path
import stat

import pytest

from local_rag_app.config import (
    get_settings as get_local_rag_settings,
    reset_settings_cache as reset_local_rag_settings_cache,
)
from model_gateway.config import (
    get_settings as get_gateway_settings,
    reset_settings_cache as reset_gateway_settings_cache,
)
from qwenrag_runtime.paths import RuntimePathError, RuntimePaths
from qwenrag_runtime.paths import reset_runtime_paths_cache
from rag_preprocess.incremental.settings import load_incremental_settings


def test_source_mode_preserves_repository_layout(tmp_path: Path) -> None:
    """Source execution keeps existing .env and rag_data locations compatible."""
    source_root = tmp_path / "source"
    source_root.mkdir()

    paths = RuntimePaths.resolve(
        environ={}, frozen=False, source_root=source_root
    )

    assert paths.source_root == source_root
    assert paths.config_root == source_root
    assert paths.knowledge_base_root == source_root / "rag_data"
    assert paths.ocr_resource_root == source_root / "models" / "ocr"
    assert paths.local_rag_env_file == source_root / ".env.local-rag"


def test_frozen_mode_uses_local_app_data_and_chinese_space_paths(tmp_path: Path) -> None:
    """A frozen install never treats its program directory as mutable storage."""
    install_root = tmp_path / "安装目录 带空格"
    bundle_root = install_root / "_internal"
    local_app_data = tmp_path / "用户 数据" / "Local"
    executable = install_root / "QwenRagRuntime.exe"
    paths = RuntimePaths.resolve(
        environ={"LOCALAPPDATA": str(local_app_data)},
        frozen=True,
        executable=executable,
        bundle_root=bundle_root,
    )

    assert paths.install_root == install_root
    assert paths.resource_root == install_root / "resources"
    assert paths.data_root == local_app_data / "QwenRAG"
    assert paths.config_root == paths.data_root / "config"
    assert paths.knowledge_base_root == paths.data_root / "data"
    assert paths.ocr_resource_root == install_root / "resources" / "ocr"
    assert paths.workbench_incoming_dir.name == "01_请把新资料放这里"


def test_mutable_initialization_does_not_write_to_read_only_install_root(
    tmp_path: Path,
) -> None:
    """Config initialization still works when the program directory is read-only."""
    install_root = tmp_path / "readonly-install"
    install_root.mkdir()
    paths = RuntimePaths.resolve(
        environ={"LOCALAPPDATA": str(tmp_path / "appdata")},
        frozen=True,
        executable=install_root / "QwenRagRuntime.exe",
        bundle_root=tmp_path / "bundle",
    )
    install_root.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        created = paths.ensure_mutable_directories()
    finally:
        install_root.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)

    assert paths.config_root in created
    assert paths.workbench_archive_dir.is_dir()
    assert list(install_root.iterdir()) == []


@pytest.mark.parametrize(
    "override_name",
    ["QWENRAG_CONFIG_ROOT", "QWENRAG_LOG_ROOT", "QWENRAG_KB_ROOT"],
)
def test_rejects_configuration_log_and_knowledge_base_outside_data_root(
    tmp_path: Path,
    override_name: str,
) -> None:
    """Mutable data cannot escape into arbitrary directories through overrides."""
    data_root = tmp_path / "data"
    outside = tmp_path / "outside"

    with pytest.raises(RuntimePathError, match=override_name):
        RuntimePaths.resolve(
            environ={
                "QWENRAG_DATA_ROOT": str(data_root),
                override_name: str(outside),
            },
            frozen=False,
            source_root=tmp_path / "source",
        )

    paths = RuntimePaths.resolve(
        environ={"QWENRAG_DATA_ROOT": str(data_root)},
        frozen=False,
        source_root=tmp_path / "source",
    )
    with pytest.raises(RuntimePathError, match="RAG_KNOWLEDGE_BASE_DIR"):
        paths.require_data_path(outside, "RAG_KNOWLEDGE_BASE_DIR")


def test_incremental_defaults_use_real_workbench_for_installed_runtime(
    tmp_path: Path,
) -> None:
    """Installed ingestion receives the three real workbench directories."""
    paths = RuntimePaths.resolve(
        environ={"LOCALAPPDATA": str(tmp_path / "local-app-data")},
        frozen=True,
        executable=tmp_path / "program" / "QwenRagRuntime.exe",
        bundle_root=tmp_path / "bundle",
    )

    settings = load_incremental_settings(runtime_paths=paths, environ={})
    settings.ensure_directories()

    assert settings.incoming_dir == paths.workbench_incoming_dir
    assert settings.results_dir == paths.workbench_results_dir
    assert settings.archive_dir == paths.workbench_archive_dir
    assert settings.incoming_dir.is_dir()
    assert settings.results_dir.is_dir()
    assert settings.archive_dir.is_dir()


def test_local_rag_loads_configuration_and_knowledge_base_from_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed local-RAG process reads config and KB paths below data_root."""
    data_root = tmp_path / "用户数据"
    config_root = data_root / "config"
    config_root.mkdir(parents=True)
    (config_root / ".env.local-rag").write_text(
        "LOCAL_RAG_MODEL=installed-local-rag\n",
        encoding="utf-8",
    )

    with monkeypatch.context() as scoped:
        scoped.setenv("QWENRAG_DATA_ROOT", str(data_root))
        scoped.setenv("QWENRAG_CONFIG_ROOT", str(config_root))
        reset_runtime_paths_cache()
        reset_local_rag_settings_cache()
        settings = get_local_rag_settings()

        assert settings.local_rag_model == "installed-local-rag"
        assert settings.rag_knowledge_base_dir == data_root / "rag_data"

    reset_runtime_paths_cache()
    reset_local_rag_settings_cache()


def test_gateway_loads_configuration_from_unified_config_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gateway no longer depends on the current working directory .env file."""
    data_root = tmp_path / "data"
    config_root = data_root / "config"
    config_root.mkdir(parents=True)
    (config_root / ".env.gateway").write_text(
        "GATEWAY_API_KEYS=installed-key\nLLM_MODEL=installed-llm\n",
        encoding="utf-8",
    )

    with monkeypatch.context() as scoped:
        scoped.setenv("QWENRAG_DATA_ROOT", str(data_root))
        scoped.setenv("QWENRAG_CONFIG_ROOT", str(config_root))
        reset_runtime_paths_cache()
        reset_gateway_settings_cache()
        settings = get_gateway_settings()

        assert settings.gateway_api_keys == ["installed-key"]
        assert settings.llm_model == "installed-llm"

    reset_runtime_paths_cache()
    reset_gateway_settings_cache()
