from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwenrag_runtime import cli
from qwenrag_runtime.deployment import (
    DeploymentConfig,
    DeploymentConfigurationError,
    SecretsConfig,
    default_deployment,
    deployment_files,
    deployment_summary,
    derive_process_environment,
    backup_and_migrate_configuration,
    initialize_configuration,
)
from qwenrag_runtime.paths import RuntimePaths


def _payload() -> dict[str, object]:
    return default_deployment().model_dump(mode="json")


def _frozen_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> RuntimePaths:
    return RuntimePaths.resolve(environ={"LOCALAPPDATA": str(tmp_path)}, frozen=True)


def test_deployment_config_accepts_valid_template() -> None:
    config = DeploymentConfig.model_validate(_payload())

    assert config.embedding.expected_dimension == config.rag.embedding_dimension == 1024
    assert config.llm.executable.is_absolute()


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda value: value["llm"].update({"executable": "relative\\server.exe"}), "executable"),
        (lambda value: value["llm"].update({"base_url": "http://10.0.0.8:8001/v1"}), "base_url"),
        (lambda value: value["ports"].update({"embedding": 8001}), "ports"),
        (lambda value: value["llm"].update({"arguments": "--port 8001"}), "arguments"),
        (lambda value: value["rag"].update({"embedding_dimension": 12}), "embedding_dimension"),
    ],
)
def test_deployment_config_rejects_unsafe_or_inconsistent_values(change, field: str) -> None:
    payload = _payload()
    change(payload)

    with pytest.raises(ValueError) as error:
        DeploymentConfig.model_validate(payload)

    assert str(error.value)


def test_initialize_configuration_is_atomic_safe_and_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("qwenrag_runtime.deployment._restrict_secrets_file", lambda _: None)

    deployment, secret_values = initialize_configuration(paths)
    files = deployment_files(paths)

    assert files.deployment_path.exists()
    assert files.secrets_path.exists()
    persisted = json.loads(files.secrets_path.read_text(encoding="utf-8"))
    assert persisted["local_rag_api_key"] == secret_values.local_rag_api_key
    assert len(secret_values.local_rag_api_key) >= 32
    assert secret_values.local_rag_api_key not in json.dumps(
        deployment_summary(deployment, secret_values), ensure_ascii=False
    )

    original = b"{ not valid json"
    files.deployment_path.write_bytes(original)
    with pytest.raises(DeploymentConfigurationError):
        initialize_configuration(paths)
    assert files.deployment_path.read_bytes() == original


def test_derived_environments_use_one_deployment_and_secrets_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)
    deployment = default_deployment()
    secret_values = SecretsConfig(
        local_rag_api_key="local-secret",
        gateway_api_key="gateway-secret",
        llm_upstream_api_key="llm-upstream",
        embedding_upstream_api_key="embedding-upstream",
    )

    environments = derive_process_environment(deployment, secret_values, paths)

    assert environments["local_rag"]["MODEL_GATEWAY_BASE_URL"] == "http://127.0.0.1:8010/v1"
    assert environments["gateway"]["LLM_BASE_URL"] == deployment.llm.base_url
    assert environments["gateway"]["LLM_UPSTREAM_API_KEY"] == "llm-upstream"
    assert environments["incremental"]["EMBEDDING_DIM"] == "1024"
    assert environments["incremental"]["EMBEDDING_REVISION"] == deployment.embedding.expected_revision
    assert environments["incremental"]["EMBEDDING_BASE_URL"] == deployment.embedding.base_url
    assert environments["incremental"]["EMBEDDING_API_KEY"] == "embedding-upstream"


def test_secret_acl_failure_removes_only_the_temporary_secret_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)
    files = deployment_files(paths)

    def deny_acl(_: Path) -> None:
        raise DeploymentConfigurationError("permission denied")

    monkeypatch.setattr("qwenrag_runtime.deployment._restrict_secrets_file", deny_acl)
    with pytest.raises(DeploymentConfigurationError):
        initialize_configuration(paths)

    assert files.deployment_path.exists()
    assert not files.secrets_path.exists()


def test_secret_file_can_be_written_with_real_windows_acl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)

    _, secret_values = initialize_configuration(paths)

    assert deployment_files(paths).secrets_path.exists()
    assert secret_values.local_rag_api_key


def test_cli_hides_secret_except_explicit_reveal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("qwenrag_runtime.deployment._restrict_secrets_file", lambda _: None)
    _, secret_values = initialize_configuration(paths)

    assert cli.main(["config", "validate"], runtime_paths=paths) == 0
    assert secret_values.local_rag_api_key not in capsys.readouterr().out

    assert cli.main(["config", "show-chatbox"], runtime_paths=paths) == 0
    assert secret_values.local_rag_api_key not in capsys.readouterr().out

    assert cli.main(["config", "show-chatbox", "--reveal-key"], runtime_paths=paths) == 0
    assert secret_values.local_rag_api_key in capsys.readouterr().out


def test_cli_run_maps_missing_config_and_secrets_to_documented_exit_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)
    files = deployment_files(paths)

    assert cli.main(["run"], runtime_paths=paths) == 20
    paths.ensure_mutable_directories()
    files.deployment_path.write_text(
        json.dumps(default_deployment().model_dump(mode="json")), encoding="utf-8"
    )
    assert cli.main(["run"], runtime_paths=paths) == 21


def test_install_diagnostics_and_empty_kb_initialization_use_only_local_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("qwenrag_runtime.deployment._restrict_secrets_file", lambda _: None)
    initialize_configuration(paths)

    assert cli.main(["diagnose-install"], runtime_paths=paths) == 31
    assert cli.main(["kb-init-empty"], runtime_paths=paths) == 0
    assert cli.main(["diagnose-install"], runtime_paths=paths) == 0
    assert '"status": "ready"' in capsys.readouterr().out


def test_configuration_migration_backs_up_existing_files_before_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("qwenrag_runtime.deployment._restrict_secrets_file", lambda _: None)
    initialize_configuration(paths)
    files = deployment_files(paths)

    backup = backup_and_migrate_configuration(paths)

    assert backup is not None
    assert (backup / "deployment.json").read_bytes() == files.deployment_path.read_bytes()
    assert (backup / "secrets.json").read_bytes() == files.secrets_path.read_bytes()


def test_configuration_migration_backs_up_invalid_config_without_overwriting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _frozen_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("qwenrag_runtime.deployment._restrict_secrets_file", lambda _: None)
    initialize_configuration(paths)
    files = deployment_files(paths)
    original = b"{ invalid configuration"
    files.deployment_path.write_bytes(original)

    with pytest.raises(DeploymentConfigurationError):
        backup_and_migrate_configuration(paths)

    backups = list((paths.data_root / "backups" / "config").glob("*"))
    assert len(backups) == 1
    assert (backups[0] / "deployment.json").read_bytes() == original
    assert files.deployment_path.read_bytes() == original
