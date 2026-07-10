#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${GATEWAY_ENV_FILE:-${PROJECT_DIR}/.env.gateway}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8010}"
GATEWAY_API_KEYS_VALUE="${GATEWAY_API_KEYS:-}"
GATEWAY_API_KEY="${GATEWAY_API_KEY:-${GATEWAY_API_KEYS_VALUE%%,*}}"
GATEWAY_API_KEY="${GATEWAY_API_KEY:-change-me}"
LLM_MODEL="${LLM_MODEL:-qwen}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-qwen3-embedding-0.6b}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUEST_ID="${REQUEST_ID:-stage12-check-$(date +%Y%m%d%H%M%S)}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

pass() {
  echo "[PASS] $1"
}

fail() {
  echo "[FAIL] $1" >&2
  if [[ $# -gt 1 ]]; then
    echo "$2" >&2
  fi
  exit 1
}

require_status() {
  local actual="$1"
  local expected="$2"
  local name="$3"
  local body_file="$4"
  if [[ "${actual}" != "${expected}" ]]; then
    fail "${name}: expected HTTP ${expected}, got ${actual}" "$(cat "${body_file}")"
  fi
}

json_check() {
  local body_file="$1"
  local check_name="$2"
  local python_code="$3"
  if ! "${PYTHON_BIN}" - "${body_file}" <<PY
import json
import sys

body_path = sys.argv[1]
with open(body_path, "r", encoding="utf-8") as f:
    data = json.load(f)

${python_code}
PY
  then
    fail "${check_name}" "$(cat "${body_file}")"
  fi
}

health_body="${TMP_DIR}/health.json"
health_headers="${TMP_DIR}/health.headers"
health_status="$(
  curl -sS -D "${health_headers}" -o "${health_body}" -w "%{http_code}" \
    -H "X-Request-ID: ${REQUEST_ID}" \
    "${GATEWAY_URL%/}/health"
)"
require_status "${health_status}" "200" "/health" "${health_body}"
json_check "${health_body}" "/health returned unexpected body" \
'assert data["status"] == "ok"
assert data["service"] == "model-gateway"'
if ! grep -iq "^x-request-id: ${REQUEST_ID}" "${health_headers}"; then
  fail "/health did not return expected X-Request-ID" "$(cat "${health_headers}")"
fi
pass "/health returns ok and X-Request-ID"

upstreams_body="${TMP_DIR}/upstreams.json"
upstreams_status="$(
  curl -sS -o "${upstreams_body}" -w "%{http_code}" \
    -H "Authorization: Bearer ${GATEWAY_API_KEY}" \
    "${GATEWAY_URL%/}/health/upstreams"
)"
require_status "${upstreams_status}" "200" "/health/upstreams" "${upstreams_body}"
json_check "${upstreams_body}" "/health/upstreams is not ok" \
'assert data["status"] == "ok"
assert data["upstreams"]["llm"]["ok"] is True
assert data["upstreams"]["embedding"]["ok"] is True'
pass "/health/upstreams reports llm and embedding ok"

models_body="${TMP_DIR}/models.json"
models_status="$(
  curl -sS -o "${models_body}" -w "%{http_code}" \
    -H "Authorization: Bearer ${GATEWAY_API_KEY}" \
    "${GATEWAY_URL%/}/v1/models"
)"
require_status "${models_status}" "200" "/v1/models" "${models_body}"
json_check "${models_body}" "/v1/models missing expected models" \
"ids = {item['id'] for item in data['data']}
assert '${LLM_MODEL}' in ids
assert '${EMBEDDING_MODEL}' in ids"
pass "/v1/models returns ${LLM_MODEL} and ${EMBEDDING_MODEL}"

chat_body="${TMP_DIR}/chat.json"
chat_status="$(
  curl -sS -o "${chat_body}" -w "%{http_code}" \
    -H "Content-Type: application/json; charset=utf-8" \
    -H "Authorization: Bearer ${GATEWAY_API_KEY}" \
    --data-binary @- \
    "${GATEWAY_URL%/}/v1/chat/completions" <<JSON
{
  "model": "${LLM_MODEL}",
  "messages": [
    {
      "role": "user",
      "content": "Explain the role of a model gateway in one sentence. /no_think"
    }
  ],
  "stream": false,
  "temperature": 0.2,
  "max_tokens": 256
}
JSON
)"
require_status "${chat_status}" "200" "/v1/chat/completions" "${chat_body}"
json_check "${chat_body}" "/v1/chat/completions returned empty content" \
'content = data["choices"][0]["message"]["content"]
assert isinstance(content, str)
assert content.strip()'
pass "/v1/chat/completions returns text"

embedding_body="${TMP_DIR}/embedding.json"
embedding_status="$(
  curl -sS -o "${embedding_body}" -w "%{http_code}" \
    -H "Content-Type: application/json; charset=utf-8" \
    -H "Authorization: Bearer ${GATEWAY_API_KEY}" \
    --data-binary @- \
    "${GATEWAY_URL%/}/v1/embeddings" <<JSON
{
  "model": "${EMBEDDING_MODEL}",
  "input": "model gateway test"
}
JSON
)"
require_status "${embedding_status}" "200" "/v1/embeddings" "${embedding_body}"
json_check "${embedding_body}" "/v1/embeddings did not return a 1024-dimensional vector" \
'embedding = data["data"][0]["embedding"]
assert isinstance(embedding, list)
assert len(embedding) == 1024'
pass "/v1/embeddings returns 1024-dimensional vector"

bad_auth_body="${TMP_DIR}/bad_auth.json"
bad_auth_status="$(
  curl -sS -o "${bad_auth_body}" -w "%{http_code}" \
    -H "Authorization: Bearer definitely-wrong-key" \
    "${GATEWAY_URL%/}/v1/models"
)"
require_status "${bad_auth_status}" "401" "wrong API key" "${bad_auth_body}"
json_check "${bad_auth_body}" "wrong API key did not return unified auth error" \
'assert data["error"]["type"] == "authentication_error"
assert data["error"]["code"] == "invalid_api_key"'
pass "wrong API key returns 401"

echo "All stage 12 gateway checks passed for ${GATEWAY_URL}"
echo "Check the gateway terminal logs for request_id=${REQUEST_ID}."
