#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/projects/conda_envs/qwen-rag}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
ENV_FILE="${GATEWAY_ENV_FILE:-${PROJECT_DIR}/.env.gateway}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
export GATEWAY_PORT="${GATEWAY_PORT:-8010}"
export GATEWAY_API_KEYS="${GATEWAY_API_KEYS:-change-me}"
export GATEWAY_ALLOW_NO_AUTH="${GATEWAY_ALLOW_NO_AUTH:-false}"

export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8001/v1}"
export LLM_MODEL="${LLM_MODEL:-qwen}"
export LLM_UPSTREAM_API_KEY="${LLM_UPSTREAM_API_KEY:-}"

export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:8002/v1}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding-0.6b}"
export EMBEDDING_UPSTREAM_API_KEY="${EMBEDDING_UPSTREAM_API_KEY:-}"

export HTTP_CONNECT_TIMEOUT_SECONDS="${HTTP_CONNECT_TIMEOUT_SECONDS:-5}"
export HTTP_READ_TIMEOUT_SECONDS="${HTTP_READ_TIMEOUT_SECONDS:-180}"
export HTTP_WRITE_TIMEOUT_SECONDS="${HTTP_WRITE_TIMEOUT_SECONDS:-60}"
export HTTP_POOL_TIMEOUT_SECONDS="${HTTP_POOL_TIMEOUT_SECONDS:-5}"

export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export LOG_REQUEST_BODY="${LOG_REQUEST_BODY:-false}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ENV_PATH=/projects/conda_envs/qwen-rag or PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

if ! "${PYTHON_BIN}" - <<'PY'
import fastapi
import httpx
import pydantic_settings
import uvicorn
PY
then
  echo "Gateway dependencies are missing." >&2
  echo "Install them on the server with:" >&2
  echo "  ${PYTHON_BIN} -m pip install -r requirements-gateway.txt" >&2
  exit 1
fi

if [[ "${SKIP_UPSTREAM_PREFLIGHT:-0}" != "1" ]]; then
  if command -v curl >/dev/null 2>&1; then
    echo "Checking LLM upstream: ${LLM_BASE_URL}/models"
    curl -fsS "${LLM_BASE_URL%/}/models" >/dev/null
    echo "Checking embedding upstream: ${EMBEDDING_BASE_URL}/models"
    curl -fsS "${EMBEDDING_BASE_URL%/}/models" >/dev/null
  else
    echo "curl not found; skipping upstream preflight." >&2
  fi
fi

if [[ "${GATEWAY_API_KEYS}" == "change-me" ]]; then
  echo "WARNING: GATEWAY_API_KEYS is still change-me. Use a long random key for real deployment." >&2
fi

echo "Starting model gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "Project directory: ${PROJECT_DIR}"
echo "Python: ${PYTHON_BIN}"

exec "${PYTHON_BIN}" -m uvicorn model_gateway.main:app \
  --host "${GATEWAY_HOST}" \
  --port "${GATEWAY_PORT}"
