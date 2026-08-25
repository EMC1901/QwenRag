"""Resolve trusted paths for source, frozen, and installed QwenRAG runtimes.

Only the resource root is allowed to live under the program installation.  Every
mutable location is rooted beneath ``data_root`` so an installed program can run
from a read-only directory without writing configuration or customer data there.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Mapping


class RuntimePathError(ValueError):
    """Raised when deployment paths are relative, unsafe, or escape their root."""


_WORKBENCH_INCOMING = "01_请把新资料放这里"
_WORKBENCH_RESULTS = "02_查看处理结果"
_WORKBENCH_ARCHIVE = "03_已处理资料归档"


@dataclass(frozen=True)
class RuntimePaths:
    """Absolute source, resource, and mutable data paths for one process."""

    source_root: Path
    bundle_root: Path
    install_root: Path
    resource_root: Path
    data_root: Path
    config_root: Path
    log_root: Path
    runtime_root: Path
    knowledge_base_root: Path
    ocr_resource_root: Path
    frozen: bool

    @property
    def local_rag_env_file(self) -> Path:
        """Return the local-RAG environment file for the active runtime mode."""
        return self.config_root / ".env.local-rag"

    @property
    def gateway_env_file(self) -> Path:
        """Return the model-gateway environment file for the active runtime mode."""
        return self.config_root / ".env.gateway"

    @property
    def incremental_env_file(self) -> Path:
        """Return the incremental-ingestion environment file for this runtime."""
        return self.config_root / ".env.incremental"

    @property
    def workbench_root(self) -> Path:
        """Return the customer-visible incremental-ingestion workbench root."""
        return self.knowledge_base_root / "workbench"

    @property
    def workbench_incoming_dir(self) -> Path:
        return self.workbench_root / _WORKBENCH_INCOMING

    @property
    def workbench_results_dir(self) -> Path:
        return self.workbench_root / _WORKBENCH_RESULTS

    @property
    def workbench_archive_dir(self) -> Path:
        return self.workbench_root / _WORKBENCH_ARCHIVE

    @property
    def mutable_directories(self) -> tuple[Path, ...]:
        """Directories that installation/config initialization may create."""
        return (
            self.data_root,
            self.config_root,
            self.log_root,
            self.runtime_root,
            self.knowledge_base_root,
            self.workbench_root,
            self.workbench_incoming_dir,
            self.workbench_results_dir,
            self.workbench_archive_dir,
        )

    def ensure_mutable_directories(self) -> tuple[Path, ...]:
        """Create only data-root locations; never write beneath the install root."""
        created: list[Path] = []
        for directory in self.mutable_directories:
            directory.mkdir(parents=True, exist_ok=True)
            if not directory.is_dir():
                raise RuntimePathError(f"Runtime path is not a directory: {directory}")
            created.append(directory)
        return tuple(created)

    def require_data_path(self, value: Path | str, setting_name: str) -> Path:
        """Resolve a path and reject values that escape the mutable data root."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.data_root / path
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise RuntimePathError(
                f"{setting_name} must be inside QWENRAG_DATA_ROOT: {resolved}"
            ) from exc
        return resolved

    @classmethod
    def resolve(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        frozen: bool | None = None,
        executable: Path | None = None,
        source_root: Path | None = None,
        bundle_root: Path | None = None,
    ) -> "RuntimePaths":
        """Resolve paths with environment overrides before mode-specific defaults."""
        values = dict(os.environ if environ is None else environ)
        is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
        discovered_source_root = (
            source_root or Path(__file__).resolve().parents[1]
        ).resolve(strict=False)

        if is_frozen:
            executable_path = (executable or Path(sys.executable)).resolve(strict=False)
            default_install_root = executable_path.parent
            default_bundle_root = (
                bundle_root
                or Path(getattr(sys, "_MEIPASS", default_install_root))
            ).resolve(strict=False)
            # PyInstaller one-folder places Python modules in ``_internal``.
            # Installer-managed OCR models instead live beside the EXE so that
            # they remain explicit, read-only resources rather than hidden
            # bundled data files.
            default_resource_root = default_install_root / "resources"
            default_data_root = _local_app_data(values) / "QwenRAG"
            default_config_root = default_data_root / "config"
            default_kb_root = default_data_root / "data"
            effective_source_root = default_bundle_root
        else:
            default_install_root = discovered_source_root
            default_bundle_root = discovered_source_root
            default_resource_root = discovered_source_root
            default_data_root = discovered_source_root
            effective_source_root = discovered_source_root

        install_root = _path_override(
            values, "QWENRAG_INSTALL_ROOT", default_install_root
        )
        resource_root = _path_override(
            values, "QWENRAG_RESOURCE_ROOT", default_resource_root
        )
        data_root = _path_override(values, "QWENRAG_DATA_ROOT", default_data_root)
        source_data_root_overridden = (
            not is_frozen and bool(values.get("QWENRAG_DATA_ROOT", "").strip())
        )
        default_config_root = (
            data_root / "config"
            if is_frozen or source_data_root_overridden
            else discovered_source_root
        )
        default_kb_root = (
            data_root / "data"
            if is_frozen
            else data_root / "rag_data"
        )
        config_root = _path_override(
            values, "QWENRAG_CONFIG_ROOT", default_config_root
        )
        log_root = _path_override(values, "QWENRAG_LOG_ROOT", data_root / "logs")
        runtime_root = data_root / "runtime"
        kb_root = _path_override(values, "QWENRAG_KB_ROOT", default_kb_root)

        for name, path in {
            "QWENRAG_CONFIG_ROOT": config_root,
            "QWENRAG_LOG_ROOT": log_root,
            "QWENRAG_KB_ROOT": kb_root,
            "runtime_root": runtime_root,
        }.items():
            _require_within(path, data_root, name)

        ocr_resource_root = (
            resource_root / "ocr" if is_frozen else resource_root / "models" / "ocr"
        )
        return cls(
            source_root=effective_source_root,
            bundle_root=default_bundle_root,
            install_root=install_root,
            resource_root=resource_root,
            data_root=data_root,
            config_root=config_root,
            log_root=log_root,
            runtime_root=runtime_root,
            knowledge_base_root=kb_root,
            ocr_resource_root=ocr_resource_root,
            frozen=is_frozen,
        )


def _local_app_data(values: Mapping[str, str]) -> Path:
    value = values.get("LOCALAPPDATA") or values.get("APPDATA")
    if value:
        return _absolute_path(value, "LOCALAPPDATA")
    return (Path.home() / "AppData" / "Local").resolve(strict=False)


def _path_override(
    values: Mapping[str, str], name: str, default: Path
) -> Path:
    value = values.get(name)
    return _absolute_path(value, name) if value and value.strip() else default.resolve(strict=False)


def _absolute_path(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimePathError(f"{name} must be an absolute path")
    return path.resolve(strict=False)


def _require_within(path: Path, root: Path, name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimePathError(f"{name} must be inside QWENRAG_DATA_ROOT") from exc


@lru_cache
def get_runtime_paths() -> RuntimePaths:
    """Return cached runtime paths for production code."""
    return RuntimePaths.resolve()


def reset_runtime_paths_cache() -> None:
    """Clear the process-local cache after tests or launcher environment changes."""
    get_runtime_paths.cache_clear()
