#!/usr/bin/env python3
"""Check incremental-ingestion prerequisites without handling user documents."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_preprocess.incremental.environment import check_incremental_environment
from rag_preprocess.incremental.settings import (
    IncrementalConfigurationError,
    load_incremental_settings,
)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="增量资料入库环境检查（不联网、不处理业务文件）")
    parser.add_argument("--env-file", type=Path, help="增量配置文件，默认 .env.incremental")
    args = parser.parse_args()
    try:
        settings = load_incremental_settings(env_file=args.env_file)
    except IncrementalConfigurationError as exc:
        print(f"配置错误：{exc}")
        print("请修正 .env.incremental 后重新运行。")
        return 21

    report = check_incremental_environment(settings)
    print(f"Python：{sys.version.split()[0]}")
    print(f"知识库磁盘可用空间：{report.available_free_bytes} bytes")
    if report.is_ready:
        print("环境检查通过：可进入后续增量入库阶段。")
        return 0
    print("环境检查未通过：")
    for issue in report.issues:
        print(f"- [{issue.code}] {issue.message} 建议：{issue.remedy}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
