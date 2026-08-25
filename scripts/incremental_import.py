#!/usr/bin/env python3
"""Stage-2 CLI for submitting and claiming incremental-ingestion Workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_preprocess.incremental.settings import (
    IncrementalConfigurationError,
    load_incremental_settings,
)
from rag_preprocess.incremental.task_submission import (
    TaskSubmissionError,
    claim_worker,
    release_task_lock,
    submit_task,
)
from rag_preprocess.incremental.persistence import write_checkpoint, write_status
from rag_preprocess.incremental.workflow import run_stages_4_to_9


def main() -> int:
    _configure_utf8()
    parser = argparse.ArgumentParser(description="增量资料入库任务提交工具")
    parser.add_argument("--env-file", type=Path, help="增量配置文件，默认 .env.incremental")
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_parser = subparsers.add_parser("submit", help="创建或恢复单一导入任务")
    submit_parser.add_argument("--json", action="store_true", help="输出供 PowerShell 读取的 JSON")
    worker_parser = subparsers.add_parser("worker", help="由后台进程认领已提交任务")
    worker_parser.add_argument("--task-id", required=True)
    failure_parser = subparsers.add_parser("fail-start", help="Worker 启动失败后释放匹配的任务锁")
    failure_parser.add_argument("--task-id", required=True)
    args = parser.parse_args()

    try:
        settings = load_incremental_settings(env_file=args.env_file)
        if args.command == "submit":
            return _submit(settings, json_output=args.json)
        if args.command == "worker":
            return _worker(settings, args.task_id)
        release_worker_start_failure(settings, args.task_id)
        return 0
    except IncrementalConfigurationError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 21
    except TaskSubmissionError as exc:
        print(f"任务提交失败：{exc}", file=sys.stderr)
        return 30


def _submit(settings, *, json_output: bool) -> int:
    outcome = submit_task(settings)
    if json_output:
        print(json.dumps(outcome.as_dict(), ensure_ascii=False))
        return 0
    if outcome.should_start_worker:
        action = "已创建" if outcome.action.value == "created" else "将恢复"
        print(f"任务{action}：{outcome.task_id}")
    else:
        print(f"已有活跃任务：{outcome.task_id}")
    print(f"状态文件：{outcome.status_relative_path}")
    return 0 if outcome.should_start_worker else 10


def _worker(settings, task_id: str) -> int:
    identity = claim_worker(settings, task_id)
    print(f"Worker 已认领任务：{task_id}（PID {identity.pid}）")
    try:
        outcome = run_stages_4_to_9(settings, task_id)
    except Exception:
        # A worker crash must never leave an apparently active task lock or
        # make a customer infer success from a silently closed launcher.
        task_path = settings.work_dir / task_id / "task.json"
        write_checkpoint(
            task_path,
            {
                "schema_version": 1,
                "task_id": task_id,
                "state": "FAILED_RESUMABLE",
                "error_code": "WORKER_RUNTIME_FAILED",
            },
        )
        write_status(
            settings.results_dir / f"{task_id}.status.txt",
            "处理未完成：后台处理程序异常退出；正式知识库未因本次任务发布新资料。请联系技术支持人员。\n",
        )
        release_task_lock(settings, task_id)
        raise
    release_task_lock(settings, task_id)
    print(f"阶段 4–12 状态：{outcome['state']}")
    return 0 if outcome["state"] not in {"FAILED_RESUMABLE"} else 31


def _configure_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
