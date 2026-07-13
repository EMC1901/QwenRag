#!/usr/bin/env bash
# 在 Linux 服务器上安全地循环续跑阶段 8。应从 tmux 会话中启动。
set -u
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/projects/conda_envs/qwen-rag/bin/python}"
BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-128}"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/server-embed-$(date +%Y%m%d-%H%M%S).log"
LOCK_FILE="${LOG_DIR}/server-embed.lock"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "已有 server embedding 任务持有锁: ${LOCK_FILE}"
  exit 1
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8002/v1}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-none}"
export EMBEDDING_TIMEOUT="${EMBEDDING_TIMEOUT:-300}"
export EMBEDDING_TRUST_ENV_PROXY="${EMBEDDING_TRUST_ENV_PROXY:-0}"
export PYTHONUNBUFFERED=1

if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EMBEDDING_BATCH_SIZE 必须为正整数: ${BATCH_SIZE}"
  exit 2
fi

health_check() {
  "${PYTHON_BIN}" - <<'PY'
import os
import requests

base = os.environ["EMBEDDING_BASE_URL"].rstrip("/")
session = requests.Session()
session.trust_env = False
try:
    response = session.get(base + "/models", timeout=30)
    response.raise_for_status()
    response = session.post(
        base + "/embeddings",
        json={"model": "qwen3-embedding-0.6b", "input": ["health check"]},
        timeout=int(os.environ["EMBEDDING_TIMEOUT"]),
    )
    response.raise_for_status()
    vector = response.json()["data"][0]["embedding"]
    if len(vector) != 1024:
        raise RuntimeError(f"embedding 维度应为 1024，实际为 {len(vector)}")
finally:
    session.close()
PY
}

remaining_count() {
  "${PYTHON_BIN}" - <<'PY'
import sqlite3
conn = sqlite3.connect("rag_data/metadata.db")
try:
    print(conn.execute("""
        SELECT COUNT(*) FROM chunks
        WHERE embedding_status IS NULL OR embedding_status != 'success'
    """).fetchone()[0])
finally:
    conn.close()
PY
}

echo "server embedding job started: $(date -Is)"
echo "log: ${LOG_FILE}"
health_check
"${PYTHON_BIN}" tools/check_embedding_consistency.py --mode quick

no_progress_rounds=0
sleep_seconds=30
while true; do
  before="$(remaining_count)"
  echo "$(date -Is) remaining_before=${before}"
  if [[ "${before}" == "0" ]]; then
    "${PYTHON_BIN}" tools/check_embedding_consistency.py --mode full --require-complete
    echo "server embedding job completed"
    exit 0
  fi

  if ! "${PYTHON_BIN}" -u scripts/build_kb.py --stage embed --resume --embedding-batch-size "${BATCH_SIZE}"; then
    echo "WARNING: 本轮 build_kb.py 以非 0 状态退出；将在 ${sleep_seconds}s 后重试。"
  fi

  "${PYTHON_BIN}" tools/check_embedding_consistency.py --mode quick
  after="$(remaining_count)"
  echo "$(date -Is) remaining_after=${after}"
  if [[ "${after}" == "0" ]]; then
    "${PYTHON_BIN}" tools/check_embedding_consistency.py --mode full --require-complete
    echo "server embedding job completed"
    exit 0
  fi
  if (( after < before )); then
    no_progress_rounds=0
    sleep_seconds=30
  else
    no_progress_rounds=$((no_progress_rounds + 1))
    sleep_seconds=60
    echo "WARNING: 连续 ${no_progress_rounds} 轮没有成功数增长；已延长重试等待时间。"
  fi
  sleep "${sleep_seconds}"
done
