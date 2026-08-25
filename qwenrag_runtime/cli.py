"""Command line surface later packaged as ``QwenRagRuntime.exe``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Callable, Sequence

from .deployment import (
    DeploymentConfigurationError,
    deployment_files,
    deployment_summary,
    backup_and_migrate_configuration,
    initialize_configuration,
    load_deployment,
    load_secrets,
)
from .model_contracts import ModelContractChecker, ModelContractError
from .paths import RuntimePaths, get_runtime_paths
from .process_supervisor import ProcessSupervisor, RuntimeLaunchError
from .ingest_runner import IngestRuntimeError, diagnose_runtime, run_ingest_worker, submit_ingest_task
from .installer_diagnostics import diagnose_install, runtime_is_active
from .kb_initializer import KnowledgeBaseInitializationError, initialize_empty_knowledge_base


RUNTIME_VERSION = "1.0.0"


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_paths: RuntimePaths | None = None,
    checker_factory: Callable[..., ModelContractChecker] = ModelContractChecker,
) -> int:
    """Run safe configuration commands or the console-owned runtime session."""
    parser = argparse.ArgumentParser(prog="QwenRagRuntime.exe")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="显示交付运行时版本")
    commands.add_parser("diagnose-runtime", help="检查冻结运行时的基础路径和资源")
    commands.add_parser("diagnose-install", help=argparse.SUPPRESS)
    commands.add_parser("check-runtime-active", help=argparse.SUPPRESS)
    commands.add_parser("run", help="启动并监督本次 QwenRAG 会话")
    commands.add_parser("serve-gateway", help=argparse.SUPPRESS)
    commands.add_parser("serve-rag", help=argparse.SUPPRESS)
    ingest = commands.add_parser("ingest", help="提交资料入库任务并启动后台 Worker")
    ingest.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    commands.add_parser("ingest-worker", help=argparse.SUPPRESS).add_argument("--task-id", required=True)
    commands.add_parser("kb-init-empty", help=argparse.SUPPRESS)
    config = commands.add_parser("config", help="初始化、校验和测试本地部署配置")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_commands.add_parser("init", help="创建缺失的 deployment.json 和 secrets.json")
    config_commands.add_parser("migrate", help=argparse.SUPPRESS)
    config_commands.add_parser("validate", help="校验配置，不显示任何密钥")
    test_models = config_commands.add_parser("test-models", help="验证 LLM 和 Embedding 服务契约")
    test_models.add_argument("--quick", action="store_true", help="只检查健康状态和模型名")
    chatbox = config_commands.add_parser("show-chatbox", help="显示 Chatbox 连接参数")
    chatbox.add_argument("--reveal-key", action="store_true", help="显式显示 Chatbox API Key")
    args = parser.parse_args(argv)
    paths = runtime_paths or get_runtime_paths()
    files = deployment_files(paths)

    try:
        if args.command == "version":
            print(RUNTIME_VERSION)
            return 0
        if args.command == "diagnose-runtime":
            print(json.dumps(diagnose_runtime(paths), ensure_ascii=False))
            return 0
        if args.command == "diagnose-install":
            result = diagnose_install(paths)
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 31
        if args.command == "check-runtime-active":
            return 10 if runtime_is_active(paths) else 0
        if args.command == "serve-gateway":
            return _serve_gateway()
        if args.command == "serve-rag":
            return _serve_rag()
        if args.command in {"ingest", "ingest-worker", "kb-init-empty"}:
            deployment = load_deployment(files.deployment_path)
            secret_values = load_secrets(files.secrets_path)
            if args.command == "kb-init-empty":
                initialize_empty_knowledge_base(
                    paths.knowledge_base_root,
                    embedding_model=deployment.embedding.expected_model,
                    embedding_revision=deployment.embedding.expected_revision,
                    embedding_dimension=deployment.embedding.expected_dimension,
                    allow_empty_workbench=True,
                )
                paths.ensure_mutable_directories()
                return 0
            if args.command == "ingest":
                if args.fixture is not None:
                    _copy_fixture(args.fixture, paths.workbench_incoming_dir)
                print(json.dumps(submit_ingest_task(deployment, secret_values, paths), ensure_ascii=False))
                return 0
            return run_ingest_worker(deployment, secret_values, paths, args.task_id)
        if args.command == "run":
            try:
                deployment = load_deployment(files.deployment_path)
            except DeploymentConfigurationError as exc:
                print(f"配置错误：{exc}")
                return 20
            try:
                secret_values = load_secrets(files.secrets_path)
            except DeploymentConfigurationError as exc:
                print(f"密钥文件错误：{exc}")
                return 21
            supervisor = ProcessSupervisor(deployment, secret_values, paths)
            supervisor.start()
            supervisor.wait_until_stopped()
            return 0
        if args.config_command == "init":
            deployment, secret_values = initialize_configuration(paths)
            print(json.dumps(deployment_summary(deployment, secret_values), ensure_ascii=False))
            return 0
        if args.config_command == "migrate":
            backup = backup_and_migrate_configuration(paths)
            print(json.dumps({"backup": str(backup) if backup else None}, ensure_ascii=False))
            return 0
        deployment = load_deployment(files.deployment_path)
        secret_values = load_secrets(files.secrets_path)
        if args.config_command == "validate":
            print(json.dumps(deployment_summary(deployment, secret_values), ensure_ascii=False))
            return 0
        if args.config_command == "test-models":
            checker = checker_factory(deployment, secret_values)
            results = checker.check_all(full=not args.quick)
            print(json.dumps([
                {"service": result.kind, "state": result.state, "expected_model": result.expected_model}
                for result in results
            ], ensure_ascii=False))
            return 0
        _show_chatbox(deployment.rag.model_name, deployment.ports.rag, secret_values.local_rag_api_key, args.reveal_key)
        return 0
    except DeploymentConfigurationError as exc:
        print(f"配置错误：{exc}")
        return 21
    except ModelContractError as exc:
        print(f"模型服务检查失败：{exc}")
        return 22
    except IngestRuntimeError as exc:
        print(f"资料入库失败：{exc} 日志目录：{paths.log_root}")
        return 31
    except KnowledgeBaseInitializationError as exc:
        print(f"初始化空知识库失败：{exc}")
        return 32
    except RuntimeLaunchError as exc:
        print(f"启动失败：{exc} 日志目录：{paths.log_root}")
        return exc.exit_code


def _show_chatbox(model_name: str, port: int, api_key: str, reveal_key: bool) -> None:
    print(f"Chatbox 地址：http://127.0.0.1:{port}/v1")
    print(f"模型名称：{model_name}")
    if reveal_key:
        print(f"API Key：{api_key}")
    else:
        print("API Key：已配置（如需显示，请显式传入 --reveal-key）")


def _serve_gateway() -> int:
    """Run the gateway in a supervisor-owned child process."""
    import uvicorn
    from model_gateway.config import get_settings
    from model_gateway.main import create_app

    settings = get_settings()
    uvicorn.run(create_app(), host=settings.gateway_host, port=settings.gateway_port)
    return 0


def _serve_rag() -> int:
    """Run the local RAG app in a supervisor-owned child process."""
    import uvicorn
    from local_rag_app.config import get_settings
    from local_rag_app.main import create_app

    settings = get_settings()
    uvicorn.run(create_app(), host=settings.local_rag_host, port=settings.local_rag_port)
    return 0


def _copy_fixture(source: Path, destination: Path) -> None:
    """Copy only direct fixture files without overwriting customer intake files."""
    if not source.is_dir():
        raise IngestRuntimeError("资料入库测试目录不存在")
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if not item.is_file():
            continue
        target = destination / item.name
        if target.exists():
            raise IngestRuntimeError("资料入库目录已有同名文件，拒绝覆盖")
        shutil.copy2(item, target)
