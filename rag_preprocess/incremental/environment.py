"""Offline-only deployment preflight for incremental ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from typing import Callable
from uuid import uuid4

from rag_preprocess.incremental.settings import IncrementalSettings


@dataclass(frozen=True)
class EnvironmentIssue:
    """One actionable preflight failure without secrets or document content."""

    code: str
    message: str
    remedy: str


@dataclass
class EnvironmentReport:
    """Result of a local, non-networked environment inspection."""

    issues: list[EnvironmentIssue] = field(default_factory=list)
    available_free_bytes: int = 0

    @property
    def is_ready(self) -> bool:
        return not self.issues

    @property
    def exit_code(self) -> int:
        return 0 if self.is_ready else 22


_IMPORT_REQUIRED_MODULES = {
    "faiss": "faiss-cpu",
    "paddle": "paddlepaddle（CPU 版）",
}

_FIND_REQUIRED_MODULES = {
    "paddleocr": "paddleocr",
    "fitz": "PyMuPDF",
    "charset_normalizer": "charset-normalizer",
    "requests": "requests",
    "dotenv": "python-dotenv",
    "docx": "python-docx",
}


def check_incremental_environment(
    settings: IncrementalSettings,
    *,
    module_importer: Callable[[str], object] = importlib.import_module,
    module_finder: Callable[[str], object | None] = importlib.util.find_spec,
    python_version: tuple[int, int] | None = None,
    disk_usage: Callable[[str | Path], shutil._ntuple_diskusage] = shutil.disk_usage,
) -> EnvironmentReport:
    """Check local prerequisites without contacting a service or downloading assets."""
    report = EnvironmentReport()
    version = python_version or (sys.version_info.major, sys.version_info.minor)
    if version < (3, 10):
        _issue(
            report,
            "PYTHON_VERSION",
            "Python 版本过低，增量入库要求 Python 3.10 或更高版本。",
            "安装已验证的 Python 版本后重新运行环境检查。",
        )

    for module_name, package_name in _IMPORT_REQUIRED_MODULES.items():
        try:
            module_importer(module_name)
        except Exception:
            _issue(
                report,
                "DEPENDENCY_MISSING",
                f"缺少依赖：{package_name}。",
                "按照 requirements/incremental-rag.txt 安装已锁定的离线 wheel 后重试。",
            )

    # PaddleOCR 3.x import can probe model-host connectivity.  Finding its
    # installed module spec is sufficient at this stage and keeps preflight
    # strictly offline; real OCR initialization belongs to the OCR stage.
    for module_name, package_name in _FIND_REQUIRED_MODULES.items():
        try:
            is_installed = module_finder(module_name) is not None
        except Exception:
            is_installed = False
        if not is_installed:
            _issue(
                report,
                "DEPENDENCY_MISSING",
                f"缺少依赖：{package_name}。",
                "按照 requirements/incremental-rag.txt 安装已锁定的离线 wheel 后重试。",
            )

    _check_ocr_models(settings, report)
    _check_writable_directories(settings, report)
    _check_disk(settings, report, disk_usage)
    _check_embedding_configuration(settings, report)
    return report


def _check_ocr_models(settings: IncrementalSettings, report: EnvironmentReport) -> None:
    if not settings.ocr_model_dir.is_dir():
        _issue(
            report,
            "OCR_MODEL_DIR_MISSING",
            "OCR 模型目录不存在。",
            "放置已验证的离线 OCR 模型，并将 OCR_MODEL_DIR 配置为该目录。",
        )
        return
    for model_name, label in (
        (settings.ocr_text_detection_model, "文本检测"),
        (settings.ocr_text_recognition_model, "文本识别"),
    ):
        if not (settings.ocr_model_dir / model_name).is_dir():
            _issue(
                report,
                "OCR_MODEL_MISSING",
                f"OCR {label}模型目录不存在。",
                "将对应离线模型目录放入 OCR_MODEL_DIR 后重新检查。",
            )


def _check_writable_directories(settings: IncrementalSettings, report: EnvironmentReport) -> None:
    try:
        directories = settings.ensure_directories()
    except (OSError, ValueError) as exc:
        _issue(
            report,
            "DIRECTORY_CREATE_FAILED",
            "无法创建增量入库所需目录。",
            "检查 rag_data 目录的 NTFS 写入权限和配置路径后重试。",
        )
        return
    for directory in directories:
        probe = directory / f".incremental-write-probe-{uuid4().hex}"
        try:
            descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.close(descriptor)
            probe.unlink()
        except OSError:
            _issue(
                report,
                "DIRECTORY_NOT_WRITABLE",
                "增量入库目录没有写入权限。",
                "为部署账号授予 rag_data/incremental 和 kb_generations 的修改权限后重试。",
            )


def _check_disk(
    settings: IncrementalSettings,
    report: EnvironmentReport,
    disk_usage: Callable[[str | Path], shutil._ntuple_diskusage],
) -> None:
    try:
        report.available_free_bytes = int(disk_usage(settings.knowledge_base_root).free)
    except OSError:
        _issue(
            report,
            "DISK_CHECK_FAILED",
            "无法读取知识库所在磁盘的可用空间。",
            "确认知识库目录可访问后重新运行环境检查。",
        )
        return
    if report.available_free_bytes < settings.incremental_min_free_bytes:
        _issue(
            report,
            "DISK_SPACE_LOW",
            "知识库所在磁盘可用空间不足。",
            "释放磁盘空间后重试；后续任务会按完整候选资产公式再次计算空间需求。",
        )


def _check_embedding_configuration(
    settings: IncrementalSettings,
    report: EnvironmentReport,
) -> None:
    if not settings.embedding_base_url or not settings.embedding_model or settings.embedding_dim <= 0:
        _issue(
            report,
            "EMBEDDING_CONFIG_INVALID",
            "Embedding 配置不完整。",
            "设置有效的 EMBEDDING_BASE_URL、EMBEDDING_MODEL 和 EMBEDDING_DIM 后重试。",
        )


def _issue(report: EnvironmentReport, code: str, message: str, remedy: str) -> None:
    report.issues.append(EnvironmentIssue(code=code, message=message, remedy=remedy))
