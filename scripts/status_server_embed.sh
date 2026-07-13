#!/usr/bin/env bash
# 显示服务器阶段 8 的只读状态，不修改任务数据。
set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/projects/conda_envs/qwen-rag/bin/python}"
cd "${PROJECT_ROOT}"

echo "== consistency =="
"${PYTHON_BIN}" tools/check_embedding_consistency.py --mode quick
echo "== running embedding processes =="
ps aux | grep '[b]uild_kb.py.*--stage embed' || true
echo "== recent logs =="
latest_log="$(ls -1t logs/server-embed-*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "${latest_log}" ]]; then
  tail -n 80 "${latest_log}"
else
  echo "尚无 server embedding 日志。"
fi
