"""Configuration boundary for the incremental-ingestion workflow.

This module intentionally does not inspect incoming files or knowledge-base
assets.  It only resolves trusted deployment paths and validates immutable
operational settings before later stages start a task.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


class IncrementalConfigurationError(ValueError):
    """Raised when incremental configuration is unsafe or internally invalid."""


@dataclass(frozen=True)
class IncrementalSettings:
    """Validated, absolute paths and scalar settings for incremental ingestion."""

    project_root: Path
    knowledge_base_root: Path
    incremental_root: Path
    incoming_dir: Path
    archive_dir: Path
    results_dir: Path
    work_dir: Path
    locks_dir: Path
    logs_dir: Path
    generations_dir: Path
    deltas_dir: Path
    manifest_path: Path
    current_pointer: Path
    health_url: str
    local_rag_host: str
    local_rag_port: int
    ocr_device: str
    ocr_cpu_threads: int
    ocr_render_dpi: int
    ocr_retry_dpi: int
    ocr_max_concurrent_pages: int
    ocr_model_dir: Path
    ocr_text_detection_model: str
    ocr_text_recognition_model: str
    ocr_low_line_confidence: float
    ocr_page_warning_confidence: float
    ocr_page_warning_low_line_ratio: float
    ocr_severe_page_confidence: float
    ocr_severe_page_min_valid_chars: int
    ocr_severe_page_garbled_ratio: float
    ocr_document_severe_page_ratio: float
    ocr_document_max_consecutive_severe_pages: int
    ocr_document_max_severe_pages: int
    file_stability_probe_count: int
    file_stability_probe_interval_seconds: int
    embedding_max_retries: int
    embedding_base_url: str
    embedding_api_key: str | None
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    failed_work_retention_days: int
    tech_log_retention_days: int
    incremental_min_free_bytes: int

    @property
    def managed_directories(self) -> tuple[Path, ...]:
        """Directories the stage-1 checker may create and probe for writes."""
        return (
            self.incremental_root,
            self.incoming_dir,
            self.archive_dir,
            self.results_dir,
            self.work_dir,
            self.locks_dir,
            self.logs_dir,
            self.generations_dir,
            self.deltas_dir,
        )

    def ensure_directories(self) -> tuple[Path, ...]:
        """Create only configured incremental state directories, never model paths."""
        created: list[Path] = []
        for path in self.managed_directories:
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                raise IncrementalConfigurationError(f"目录无法创建或不是目录：{path.name}")
            created.append(path)
        return tuple(created)


def load_incremental_settings(
    *,
    env_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
    project_root: Path | None = None,
) -> IncrementalSettings:
    """Load ``.env.incremental`` without changing process environment variables."""
    root = (project_root or Path(__file__).resolve().parents[2]).resolve(strict=False)
    selected_env_file = env_file or root / ".env.incremental"
    values: dict[str, str] = {}
    if selected_env_file.is_file():
        values.update(
            {
                key: value
                for key, value in dotenv_values(selected_env_file).items()
                if value is not None
            }
        )
    values.update(dict(os.environ if environ is None else environ))

    kb_root = _path(values, "INCREMENTAL_KB_ROOT", "rag_data", root)
    incremental_root = _path(values, "INCREMENTAL_ROOT", kb_root / "incremental", root)
    incoming_dir = _path(values, "INCREMENTAL_INCOMING_DIR", incremental_root / "incoming", root)
    archive_dir = _path(values, "INCREMENTAL_ARCHIVE_DIR", incremental_root / "archive", root)
    results_dir = _path(values, "INCREMENTAL_RESULTS_DIR", incremental_root / "results", root)
    work_dir = _path(values, "INCREMENTAL_WORK_DIR", incremental_root / "work", root)
    locks_dir = _path(values, "INCREMENTAL_LOCKS_DIR", incremental_root / "locks", root)
    logs_dir = _path(values, "INCREMENTAL_LOG_DIR", incremental_root / "logs", root)
    generations_dir = _path(values, "KB_GENERATIONS_DIR", kb_root / "kb_generations", root)
    deltas_dir = _path(values, "KB_DELTAS_DIR", kb_root / "kb_deltas", root)
    manifest_path = _path(values, "KB_MANIFEST_PATH", kb_root / "knowledge_manifest.json", root)
    current_pointer = _path(values, "KB_CURRENT_POINTER", kb_root / "current_generation.txt", root)
    data_paths = {
        "INCREMENTAL_ROOT": incremental_root,
        "INCREMENTAL_INCOMING_DIR": incoming_dir,
        "INCREMENTAL_ARCHIVE_DIR": archive_dir,
        "INCREMENTAL_RESULTS_DIR": results_dir,
        "INCREMENTAL_WORK_DIR": work_dir,
        "INCREMENTAL_LOCKS_DIR": locks_dir,
        "INCREMENTAL_LOG_DIR": logs_dir,
        "KB_GENERATIONS_DIR": generations_dir,
        "KB_DELTAS_DIR": deltas_dir,
        "KB_MANIFEST_PATH": manifest_path,
        "KB_CURRENT_POINTER": current_pointer,
    }
    for key, path in data_paths.items():
        _require_within(path, kb_root, key, "知识库根目录")

    ocr_model_dir = _path(values, "OCR_MODEL_DIR", "models/ocr", root)
    _require_within(ocr_model_dir, root, "OCR_MODEL_DIR", "项目目录")
    settings = IncrementalSettings(
        project_root=root,
        knowledge_base_root=kb_root,
        incremental_root=incremental_root,
        incoming_dir=incoming_dir,
        archive_dir=archive_dir,
        results_dir=results_dir,
        work_dir=work_dir,
        locks_dir=locks_dir,
        logs_dir=logs_dir,
        generations_dir=generations_dir,
        deltas_dir=deltas_dir,
        manifest_path=manifest_path,
        current_pointer=current_pointer,
        health_url=_url(values, "LOCAL_RAG_HEALTH_URL", "http://127.0.0.1:18080/health"),
        local_rag_host=_loopback_host(values),
        local_rag_port=_positive_int(values, "LOCAL_RAG_PORT", 18080),
        ocr_device=_cpu_device(values),
        ocr_cpu_threads=_positive_int(values, "OCR_CPU_THREADS", 4),
        ocr_render_dpi=_positive_int(values, "OCR_RENDER_DPI", 200),
        ocr_retry_dpi=_positive_int(values, "OCR_RETRY_DPI", 300),
        ocr_max_concurrent_pages=_positive_int(values, "OCR_MAX_CONCURRENT_PAGES", 1),
        ocr_model_dir=ocr_model_dir,
        ocr_text_detection_model=_model_directory_name(values, "OCR_TEXT_DETECTION_MODEL", "PP-OCRv5_mobile_det"),
        ocr_text_recognition_model=_model_directory_name(values, "OCR_TEXT_RECOGNITION_MODEL", "PP-OCRv5_mobile_rec"),
        ocr_low_line_confidence=_ratio(values, "OCR_LOW_LINE_CONFIDENCE", 0.80),
        ocr_page_warning_confidence=_ratio(values, "OCR_PAGE_WARNING_CONFIDENCE", 0.85),
        ocr_page_warning_low_line_ratio=_ratio(values, "OCR_PAGE_WARNING_LOW_LINE_RATIO", 0.20),
        ocr_severe_page_confidence=_ratio(values, "OCR_SEVERE_PAGE_CONFIDENCE", 0.60),
        ocr_severe_page_min_valid_chars=_positive_int(values, "OCR_SEVERE_PAGE_MIN_VALID_CHARS", 20),
        ocr_severe_page_garbled_ratio=_ratio(values, "OCR_SEVERE_PAGE_GARBLED_RATIO", 0.10),
        ocr_document_severe_page_ratio=_ratio(values, "OCR_DOCUMENT_SEVERE_PAGE_RATIO", 0.20),
        ocr_document_max_consecutive_severe_pages=_positive_int(values, "OCR_DOCUMENT_MAX_CONSECUTIVE_SEVERE_PAGES", 5),
        ocr_document_max_severe_pages=_positive_int(values, "OCR_DOCUMENT_MAX_SEVERE_PAGES", 10),
        file_stability_probe_count=_positive_int(values, "FILE_STABILITY_PROBE_COUNT", 3),
        file_stability_probe_interval_seconds=_positive_int(values, "FILE_STABILITY_PROBE_INTERVAL_SECONDS", 2),
        embedding_max_retries=_positive_int(values, "EMBEDDING_MAX_RETRIES", 3),
        embedding_base_url=_url(values, "EMBEDDING_BASE_URL", "http://127.0.0.1:8002/v1"),
        embedding_api_key=_optional_text(values, "EMBEDDING_API_KEY"),
        embedding_model=_required_text(values, "EMBEDDING_MODEL", "qwen3-embedding-0.6b"),
        embedding_dim=_positive_int(values, "EMBEDDING_DIM", 1024),
        embedding_batch_size=_positive_int(values, "EMBEDDING_BATCH_SIZE", 128),
        failed_work_retention_days=_nonnegative_int(values, "FAILED_WORK_RETENTION_DAYS", 7),
        tech_log_retention_days=_nonnegative_int(values, "TECH_LOG_RETENTION_DAYS", 180),
        incremental_min_free_bytes=_positive_int(values, "INCREMENTAL_MIN_FREE_BYTES", 1),
    )
    _validate_cross_fields(settings)
    return settings


def _path(values: Mapping[str, str], key: str, default: str | Path, project_root: Path) -> Path:
    value = values.get(key, default)
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def _require_within(path: Path, root: Path, key: str, boundary_name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise IncrementalConfigurationError(f"{key} 必须位于{boundary_name}内") from exc


def _required_text(values: Mapping[str, str], key: str, default: str) -> str:
    value = str(values.get(key, default)).strip()
    if not value:
        raise IncrementalConfigurationError(f"{key} 不能为空")
    return value


def _optional_text(values: Mapping[str, str], key: str) -> str | None:
    value = str(values.get(key, "")).strip()
    return value or None


def _positive_int(values: Mapping[str, str], key: str, default: int) -> int:
    value = _integer(values, key, default)
    if value <= 0:
        raise IncrementalConfigurationError(f"{key} 必须是正整数")
    return value


def _nonnegative_int(values: Mapping[str, str], key: str, default: int) -> int:
    value = _integer(values, key, default)
    if value < 0:
        raise IncrementalConfigurationError(f"{key} 不能小于 0")
    return value


def _integer(values: Mapping[str, str], key: str, default: int) -> int:
    raw = values.get(key, default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise IncrementalConfigurationError(f"{key} 必须是整数") from exc


def _ratio(values: Mapping[str, str], key: str, default: float) -> float:
    raw = values.get(key, default)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise IncrementalConfigurationError(f"{key} 必须是 0 到 1 之间的数值") from exc
    if not 0 <= value <= 1:
        raise IncrementalConfigurationError(f"{key} 必须在 [0, 1] 范围内")
    return value


def _url(values: Mapping[str, str], key: str, default: str) -> str:
    from urllib.parse import urlsplit

    value = _required_text(values, key, default).rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise IncrementalConfigurationError(f"{key} 必须是有效的 http 或 https 地址")
    return value


def _loopback_host(values: Mapping[str, str]) -> str:
    value = _required_text(values, "LOCAL_RAG_HOST", "127.0.0.1")
    if value not in {"127.0.0.1", "::1"}:
        raise IncrementalConfigurationError(
            "LOCAL_RAG_HOST 首期只能是 127.0.0.1 或 ::1"
        )
    return value


def _cpu_device(values: Mapping[str, str]) -> str:
    value = _required_text(values, "OCR_DEVICE", "cpu").lower()
    if value != "cpu":
        raise IncrementalConfigurationError("OCR_DEVICE 首期必须为 cpu")
    return value


def _model_directory_name(values: Mapping[str, str], key: str, default: str) -> str:
    value = _required_text(values, key, default)
    if Path(value).name != value or value in {".", ".."}:
        raise IncrementalConfigurationError(f"{key} 必须是 OCR_MODEL_DIR 下的目录名")
    return value


def _validate_cross_fields(settings: IncrementalSettings) -> None:
    if settings.ocr_retry_dpi < settings.ocr_render_dpi:
        raise IncrementalConfigurationError("OCR_RETRY_DPI 不得小于 OCR_RENDER_DPI")
    if settings.ocr_low_line_confidence > settings.ocr_page_warning_confidence:
        raise IncrementalConfigurationError(
            "OCR_LOW_LINE_CONFIDENCE 不得高于 OCR_PAGE_WARNING_CONFIDENCE"
        )
    if settings.ocr_severe_page_confidence > settings.ocr_page_warning_confidence:
        raise IncrementalConfigurationError(
            "OCR_SEVERE_PAGE_CONFIDENCE 不得高于 OCR_PAGE_WARNING_CONFIDENCE"
        )
