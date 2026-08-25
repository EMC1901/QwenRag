"""Strict deployment configuration and secret handling for installed QwenRAG."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from .paths import RuntimePaths


class DeploymentConfigurationError(ValueError):
    """A safe configuration error that never includes a secret or response body."""


class ServicePorts(BaseModel):
    """Ports owned by the separately deployed model services and local RAG stack."""

    model_config = ConfigDict(extra="forbid")

    llm: int = Field(ge=1, le=65535)
    embedding: int = Field(ge=1, le=65535)
    gateway: int = Field(ge=1, le=65535)
    rag: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def validate_unique_ports(self) -> "ServicePorts":
        values = (self.llm, self.embedding, self.gateway, self.rag)
        if len(set(values)) != len(values):
            raise ValueError("服务端口不能重复")
        return self


class ModelServiceConfig(BaseModel):
    """One externally deployed OpenAI-compatible model service."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    ready_url: str
    expected_model: str = Field(min_length=1)
    executable: Path
    working_directory: Path
    arguments: list[StrictStr]
    startup_timeout_seconds: int = Field(ge=5, le=1800)

    @field_validator("base_url", "ready_url")
    @classmethod
    def validate_loopback_url(cls, value: str) -> str:
        normalized = str(value).strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("模型服务地址必须是本机回环 http 地址")
        try:
            if parsed.port is None:
                raise ValueError
        except ValueError as exc:
            raise ValueError("模型服务地址必须包含有效端口") from exc
        return normalized

    @field_validator("expected_model")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("模型名称不能为空")
        return normalized

    @field_validator("executable", "working_directory", mode="before")
    @classmethod
    def require_absolute_path(cls, value: Any) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("模型程序和工作目录必须使用绝对路径")
        return path.resolve(strict=False)

    @field_validator("arguments", mode="before")
    @classmethod
    def require_argument_array(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("模型启动参数必须是字符串数组")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("模型启动参数必须是非空字符串数组")
        return value


class EmbeddingServiceConfig(ModelServiceConfig):
    """Embedding-specific model contract settings."""

    expected_revision: str = Field(min_length=1)
    expected_dimension: int = Field(gt=0)

    @field_validator("expected_revision")
    @classmethod
    def normalize_revision(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Embedding 制品版本不能为空")
        return normalized


class RagDeploymentConfig(BaseModel):
    """Local RAG settings that must align with the embedding deployment."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    llm_context_window: int = Field(gt=0)

    @field_validator("model_name")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("RAG 模型名称不能为空")
        return normalized


class RuntimeServiceConfig(BaseModel):
    """Timeouts for the services owned by the QwenRAG process supervisor."""

    model_config = ConfigDict(extra="forbid")

    gateway_startup_timeout_seconds: int = Field(default=30, ge=5, le=1800)
    rag_startup_timeout_seconds: int = Field(default=300, ge=5, le=1800)
    graceful_shutdown_timeout_seconds: int = Field(default=10, ge=1, le=120)


class DeploymentConfig(BaseModel):
    """The single non-secret deployment source used by all local processes."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    ports: ServicePorts
    llm: ModelServiceConfig
    embedding: EmbeddingServiceConfig
    rag: RagDeploymentConfig
    runtime: RuntimeServiceConfig = Field(default_factory=RuntimeServiceConfig)

    @field_validator("schema_version")
    @classmethod
    def require_supported_schema(cls, value: int) -> int:
        if value != 1:
            raise ValueError("不支持的 deployment.json schema_version")
        return value

    @model_validator(mode="after")
    def validate_cross_service_contract(self) -> "DeploymentConfig":
        _require_url_port(self.llm.base_url, self.ports.llm, "LLM")
        _require_url_port(self.llm.ready_url, self.ports.llm, "LLM")
        _require_url_port(self.embedding.base_url, self.ports.embedding, "Embedding")
        _require_url_port(self.embedding.ready_url, self.ports.embedding, "Embedding")
        if self.embedding.expected_dimension != self.rag.embedding_dimension:
            raise ValueError("Embedding 维度必须与 RAG 配置一致")
        return self


class SecretsConfig(BaseModel):
    """Secrets separated from deployment.json and never included in normal output."""

    model_config = ConfigDict(extra="forbid")

    local_rag_api_key: str = Field(min_length=1)
    gateway_api_key: str = Field(min_length=1)
    llm_upstream_api_key: str | None = None
    embedding_upstream_api_key: str | None = None

    @field_validator(
        "local_rag_api_key",
        "gateway_api_key",
        "llm_upstream_api_key",
        "embedding_upstream_api_key",
        mode="before",
    )
    @classmethod
    def normalize_secret(cls, value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @model_validator(mode="after")
    def require_internal_keys(self) -> "SecretsConfig":
        if not self.local_rag_api_key or not self.gateway_api_key:
            raise ValueError("本地 RAG 和网关密钥必须已配置")
        return self

    @classmethod
    def create(cls) -> "SecretsConfig":
        return cls(
            local_rag_api_key=secrets.token_urlsafe(32),
            gateway_api_key=secrets.token_urlsafe(32),
        )

    def redacted_status(self) -> dict[str, bool]:
        """Expose only whether each secret exists, never its value."""
        return {
            "local_rag_api_key_configured": bool(self.local_rag_api_key),
            "gateway_api_key_configured": bool(self.gateway_api_key),
            "llm_upstream_api_key_configured": bool(self.llm_upstream_api_key),
            "embedding_upstream_api_key_configured": bool(
                self.embedding_upstream_api_key
            ),
        }


@dataclass(frozen=True)
class DeploymentFiles:
    """Paths to the two persistent configuration files."""

    deployment_path: Path
    secrets_path: Path


def deployment_files(paths: RuntimePaths) -> DeploymentFiles:
    return DeploymentFiles(
        deployment_path=paths.config_root / "deployment.json",
        secrets_path=paths.config_root / "secrets.json",
    )


def default_deployment() -> DeploymentConfig:
    """Return a valid template that the implementation engineer can edit safely."""
    llm_root = Path(r"C:\AI\llama.cpp")
    model_root = Path(r"C:\AI\models")
    return DeploymentConfig(
        schema_version=1,
        ports={"llm": 8001, "embedding": 8002, "gateway": 8010, "rag": 18080},
        llm={
            "base_url": "http://127.0.0.1:8001/v1",
            "ready_url": "http://127.0.0.1:8001/health",
            "expected_model": "qwen",
            "executable": llm_root / "llama-server.exe",
            "working_directory": llm_root,
            "arguments": [
                "-m", str(model_root / "Qwen-Q4_K_M.gguf"), "--host", "127.0.0.1",
                "--port", "8001", "--alias", "qwen", "-c", "8192", "--n-gpu-layers", "99",
            ],
            "startup_timeout_seconds": 600,
        },
        embedding={
            "base_url": "http://127.0.0.1:8002/v1",
            "ready_url": "http://127.0.0.1:8002/health",
            "expected_model": "qwen3-embedding-0.6b",
            "expected_revision": "IMPLEMENTER_MUST_SET_ARTIFACT_REVISION",
            "expected_dimension": 1024,
            "executable": llm_root / "llama-server.exe",
            "working_directory": llm_root,
            "arguments": [
                "-m", str(model_root / "Qwen3-Embedding-0.6B.gguf"), "--host", "127.0.0.1",
                "--port", "8002", "--alias", "qwen3-embedding-0.6b", "--embedding",
                "--pooling", "last", "--embd-normalize", "2", "--n-gpu-layers", "99",
            ],
            "startup_timeout_seconds": 300,
        },
        rag={"model_name": "local-rag", "embedding_dimension": 1024, "llm_context_window": 8192},
    )


def load_deployment(path: Path) -> DeploymentConfig:
    return _load_json_model(path, DeploymentConfig, "deployment.json")


def load_secrets(path: Path) -> SecretsConfig:
    return _load_json_model(path, SecretsConfig, "secrets.json")


def initialize_configuration(paths: RuntimePaths) -> tuple[DeploymentConfig, SecretsConfig]:
    """Create missing config files atomically; malformed existing files are untouched."""
    paths.ensure_mutable_directories()
    files = deployment_files(paths)
    deployment = load_deployment(files.deployment_path) if files.deployment_path.exists() else default_deployment()
    secret_values = load_secrets(files.secrets_path) if files.secrets_path.exists() else SecretsConfig.create()
    if not files.deployment_path.exists():
        _atomic_write_json(files.deployment_path, deployment.model_dump(mode="json"))
    if not files.secrets_path.exists():
        _atomic_write_json(
            files.secrets_path,
            secret_values.model_dump(mode="json"),
            restrict_to_current_user=True,
        )
    return deployment, secret_values


def backup_and_migrate_configuration(paths: RuntimePaths) -> Path | None:
    """Back up persistent configuration before applying a schema migration.

    Version 1 has no field transformation yet, but it still goes through this
    entry point.  That makes upgrades deterministic: a future migration must
    first leave an account-local, timestamped copy of both config files.
    Invalid configurations are backed up too, then rejected without modifying
    the originals.
    """
    files = deployment_files(paths)
    present = tuple(
        path for path in (files.deployment_path, files.secrets_path) if path.exists()
    )
    if not present:
        return None

    backup_root = paths.data_root / "backups" / "config"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    staging = backup_root / (timestamp + ".staging")
    destination = backup_root / timestamp
    try:
        staging.mkdir(parents=True, exist_ok=False)
        for source in present:
            copied = staging / source.name
            shutil.copy2(source, copied)
            if source == files.secrets_path:
                _restrict_secrets_file(copied)
        staging.rename(destination)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise DeploymentConfigurationError("配置备份失败，升级已停止且原配置未被修改。") from exc

    # Both models currently accept schema version 1 only.  Validate after the
    # backup; future versioned transformations belong immediately below this
    # comment and must write via _atomic_write_json.
    if files.deployment_path.exists():
        load_deployment(files.deployment_path)
    if files.secrets_path.exists():
        load_secrets(files.secrets_path)
    return destination


def deployment_summary(deployment: DeploymentConfig, secret_values: SecretsConfig) -> dict[str, Any]:
    """Return safe diagnostics that deliberately exclude secret values."""
    return {
        "schema_version": deployment.schema_version,
        "ports": deployment.ports.model_dump(),
        "llm_model": deployment.llm.expected_model,
        "embedding_model": deployment.embedding.expected_model,
        "embedding_revision_configured": bool(deployment.embedding.expected_revision),
        "embedding_dimension": deployment.embedding.expected_dimension,
        "secrets": secret_values.redacted_status(),
    }


def derive_process_environment(
    deployment: DeploymentConfig,
    secret_values: SecretsConfig,
    paths: RuntimePaths,
) -> dict[str, dict[str, str]]:
    """Derive all service environments from one config and one secrets file."""
    gateway_base_url = f"http://127.0.0.1:{deployment.ports.gateway}/v1"
    common = {
        "RAG_KNOWLEDGE_BASE_DIR": str(paths.knowledge_base_root),
        "RAG_EMBEDDING_DIM": str(deployment.rag.embedding_dimension),
        "UPSTREAM_EMBEDDING_MODEL": deployment.embedding.expected_model,
        "UPSTREAM_EMBEDDING_REVISION": deployment.embedding.expected_revision,
        "QWENRAG_LOG_ROOT": str(paths.log_root),
    }
    local_rag = {
        **common,
        "LOCAL_RAG_HOST": "127.0.0.1",
        "LOCAL_RAG_PORT": str(deployment.ports.rag),
        "LOCAL_RAG_MODEL": deployment.rag.model_name,
        "LOCAL_RAG_API_KEYS": secret_values.local_rag_api_key,
        "MODEL_GATEWAY_BASE_URL": gateway_base_url,
        "MODEL_GATEWAY_API_KEY": secret_values.gateway_api_key,
        "UPSTREAM_LLM_MODEL": deployment.llm.expected_model,
        "RAG_LLM_CONTEXT_WINDOW_TOKENS": str(deployment.rag.llm_context_window),
        "LOCAL_RAG_ANSWER_MODE": "gateway",
        "ENABLE_RAG_ROUTER": "true",
        "ENABLE_LOCAL_RETRIEVAL": "true",
        "ENABLE_RAG_ANSWER_GENERATION": "true",
        "ENABLE_REFERENCE_DISPLAY": "true",
    }
    gateway = {
        "GATEWAY_HOST": "127.0.0.1",
        "GATEWAY_PORT": str(deployment.ports.gateway),
        "GATEWAY_API_KEYS": secret_values.gateway_api_key,
        "LLM_BASE_URL": deployment.llm.base_url,
        "LLM_MODEL": deployment.llm.expected_model,
        "EMBEDDING_BASE_URL": deployment.embedding.base_url,
        "EMBEDDING_MODEL": deployment.embedding.expected_model,
        "EMBEDDING_REVISION": deployment.embedding.expected_revision,
    }
    if secret_values.llm_upstream_api_key:
        gateway["LLM_UPSTREAM_API_KEY"] = secret_values.llm_upstream_api_key
    if secret_values.embedding_upstream_api_key:
        gateway["EMBEDDING_UPSTREAM_API_KEY"] = secret_values.embedding_upstream_api_key
    incremental = {
        "INCREMENTAL_KB_ROOT": str(paths.knowledge_base_root),
        "INCREMENTAL_INCOMING_DIR": str(paths.workbench_incoming_dir),
        "INCREMENTAL_RESULTS_DIR": str(paths.workbench_results_dir),
        "INCREMENTAL_ARCHIVE_DIR": str(paths.workbench_archive_dir),
        # Incremental ingestion deliberately runs without the RAG process or
        # model gateway. It must therefore talk directly to the independently
        # deployed embedding endpoint using that endpoint's credential.
        "EMBEDDING_BASE_URL": deployment.embedding.base_url,
        "EMBEDDING_API_KEY": secret_values.embedding_upstream_api_key,
        "EMBEDDING_MODEL": deployment.embedding.expected_model,
        "EMBEDDING_REVISION": deployment.embedding.expected_revision,
        "EMBEDDING_DIM": str(deployment.embedding.expected_dimension),
        "OCR_MODEL_DIR": str(paths.ocr_resource_root),
    }
    return {"local_rag": local_rag, "gateway": gateway, "incremental": incremental}


def _load_json_model(path: Path, model_type: type[BaseModel], name: str):
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise TypeError
        return model_type.model_validate(raw)
    except (OSError, ValueError, TypeError) as exc:
        raise DeploymentConfigurationError(
            f"{name} 无法读取或格式不正确，请保留原文件并修复后重试。"
        ) from exc


def _atomic_write_json(
    path: Path,
    value: dict[str, Any],
    *,
    restrict_to_current_user: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if restrict_to_current_user:
            # Apply the ACL before replacement.  A failure removes only the
            # temporary file, never leaving a newly-created secret in place.
            _restrict_secrets_file(Path(temporary_name))
        os.replace(temporary_name, path)
    except OSError as exc:
        raise DeploymentConfigurationError("配置文件写入失败，请检查数据目录权限。") from exc
    finally:
        if temporary_name and os.path.exists(temporary_name):
            try:
                os.unlink(temporary_name)
            except OSError:
                # Preserve the original write/ACL failure instead of masking it.
                pass


def _restrict_secrets_file(path: Path) -> None:
    """Restrict secrets to the current Windows account after atomic replacement."""
    os.chmod(path, 0o600)
    if os.name != "nt":
        return
    username = os.environ.get("USERNAME", "").strip()
    if not username:
        raise DeploymentConfigurationError("无法确定当前 Windows 用户，未写入密钥文件。")
    result = subprocess.run(
        # Windows requires DELETE permission for the atomic ``os.replace``
        # that follows. Full control remains restricted to this one account.
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(F)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DeploymentConfigurationError("无法设置 secrets.json 的当前用户访问权限。")


def _require_url_port(url: str, expected_port: int, label: str) -> None:
    try:
        actual_port = urlsplit(url).port
    except ValueError as exc:
        raise ValueError(f"{label} 地址端口无效") from exc
    if actual_port != expected_port:
        raise ValueError(f"{label} 地址端口必须与 ports 配置一致")
