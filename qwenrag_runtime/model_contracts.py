"""Safe OpenAI-compatible contract checks for separately deployed model services."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import httpx

from .deployment import DeploymentConfig, EmbeddingServiceConfig, ModelServiceConfig, SecretsConfig


ServiceKind = Literal["llm", "embedding"]


class ModelContractError(RuntimeError):
    """A client-safe model service failure without a response body or secret."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ServiceContractResult:
    """A successful model service check; ``state`` supports later port reuse logic."""

    kind: ServiceKind
    expected_model: str
    state: Literal["reused"] = "reused"
    normalized_probe_vector: tuple[float, ...] | None = None


class ModelContractChecker:
    """Check readiness, model identity, and request semantics without starting processes."""

    def __init__(
        self,
        deployment: DeploymentConfig,
        secret_values: SecretsConfig,
        *,
        client: httpx.Client | None = None,
        ready_attempts: int = 3,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._deployment = deployment
        self._secrets = secret_values
        self._client = client or httpx.Client(timeout=httpx.Timeout(10.0))
        self._owns_client = client is None
        self._ready_attempts = max(1, ready_attempts)
        self._sleep = sleep or (lambda _: None)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def check_all(self, *, full: bool = True) -> tuple[ServiceContractResult, ...]:
        """Check LLM then Embedding, failing closed before any process action occurs."""
        try:
            return (
                self.check_llm(full=full),
                self.check_embedding(full=full),
            )
        finally:
            self.close()

    def check_llm(self, *, full: bool = True) -> ServiceContractResult:
        service = self._deployment.llm
        self._check_startup("llm", service, self._secrets.llm_upstream_api_key)
        if full:
            self._check_llm_response(service, self._secrets.llm_upstream_api_key)
            self._check_llm_stream(service, self._secrets.llm_upstream_api_key)
        return ServiceContractResult(kind="llm", expected_model=service.expected_model)

    def check_embedding(self, *, full: bool = True) -> ServiceContractResult:
        service = self._deployment.embedding
        self._check_startup("embedding", service, self._secrets.embedding_upstream_api_key)
        vector: tuple[float, ...] | None = None
        if full:
            vector = self._check_embedding_response(
                service, self._secrets.embedding_upstream_api_key
            )
        return ServiceContractResult(
            kind="embedding",
            expected_model=service.expected_model,
            normalized_probe_vector=vector,
        )

    def _check_startup(
        self, kind: ServiceKind, service: ModelServiceConfig, api_key: str | None
    ) -> None:
        self._check_readiness(kind, service, api_key)
        payload = self._request_json(
            kind, "GET", _join_url(service.base_url, "models"), api_key
        )
        data = payload.get("data")
        if not isinstance(data, list):
            raise ModelContractError("models_invalid", f"{_label(kind)} 模型列表格式不正确")
        model_ids = {
            item.get("id")
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if service.expected_model not in model_ids:
            raise ModelContractError(
                "model_mismatch",
                f"端口上的 {_label(kind)} 不是配置中的预期模型。",
            )

    def _check_readiness(
        self, kind: ServiceKind, service: ModelServiceConfig, api_key: str | None
    ) -> None:
        headers = _auth_headers(api_key)
        for attempt in range(self._ready_attempts):
            try:
                response = self._client.get(service.ready_url, headers=headers)
            except httpx.TimeoutException as exc:
                if attempt + 1 == self._ready_attempts:
                    raise ModelContractError("ready_timeout", f"{_label(kind)} 服务就绪检查超时") from exc
                self._sleep(0.2)
                continue
            except httpx.HTTPError as exc:
                raise ModelContractError("ready_connection_failed", f"无法连接 {_label(kind)} 服务") from exc
            if response.status_code == 200:
                return
            if response.status_code in {404, 405}:
                # Some OpenAI-compatible engines have no standalone health endpoint.
                return
            if response.status_code == 503 and attempt + 1 < self._ready_attempts:
                self._sleep(0.2)
                continue
            raise ModelContractError("ready_failed", f"{_label(kind)} 服务尚未就绪")

    def _check_llm_response(self, service: ModelServiceConfig, api_key: str | None) -> None:
        payload = self._request_json(
            "llm",
            "POST",
            _join_url(service.base_url, "chat/completions"),
            api_key,
            json_body={
                "model": service.expected_model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 8,
                "stream": False,
            },
        )
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelContractError("chat_invalid", "LLM 测试响应格式不正确")
        first = choices[0]
        if not isinstance(first, dict):
            raise ModelContractError("chat_invalid", "LLM 测试响应格式不正确")
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelContractError("chat_empty", "LLM 测试响应为空")

    def _check_llm_stream(self, service: ModelServiceConfig, api_key: str | None) -> None:
        url = _join_url(service.base_url, "chat/completions")
        try:
            with self._client.stream(
                "POST",
                url,
                headers=_auth_headers(api_key),
                json={
                    "model": service.expected_model,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_tokens": 8,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    raise ModelContractError("stream_http_error", "LLM 流式测试请求失败")
                if not any(
                    line.strip() == "data: [DONE]" for line in response.iter_lines()
                ):
                    raise ModelContractError("stream_incomplete", "LLM 流式响应未以 [DONE] 结束")
        except ModelContractError:
            raise
        except httpx.TimeoutException as exc:
            raise ModelContractError("stream_timeout", "LLM 流式测试超时") from exc
        except httpx.HTTPError as exc:
            raise ModelContractError("stream_connection_failed", "LLM 流式测试连接失败") from exc

    def _check_embedding_response(
        self, service: EmbeddingServiceConfig, api_key: str | None
    ) -> tuple[float, ...]:
        payload = self._request_json(
            "embedding",
            "POST",
            _join_url(service.base_url, "embeddings"),
            api_key,
            json_body={
                "model": service.expected_model,
                "input": ["QwenRAG embedding contract probe"],
            },
        )
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise ModelContractError("embedding_invalid", "Embedding 测试响应格式不正确")
        raw_vector = data[0].get("embedding")
        if not isinstance(raw_vector, list) or len(raw_vector) != service.expected_dimension:
            raise ModelContractError("embedding_dimension", "Embedding 向量维度与配置不一致")
        vector: list[float] = []
        for value in raw_vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ModelContractError("embedding_invalid", "Embedding 向量包含非数值元素")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ModelContractError("embedding_non_finite", "Embedding 向量包含非有限数值")
            vector.append(numeric)
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ModelContractError("embedding_norm", "Embedding 向量范数无效")
        return tuple(value / norm for value in vector)

    def _request_json(
        self,
        kind: ServiceKind,
        method: str,
        url: str,
        api_key: str | None,
        *,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        try:
            response = self._client.request(
                method, url, headers=_auth_headers(api_key), json=json_body
            )
        except httpx.TimeoutException as exc:
            raise ModelContractError("request_timeout", f"{_label(kind)} 服务请求超时") from exc
        except httpx.HTTPError as exc:
            raise ModelContractError("request_connection_failed", f"无法连接 {_label(kind)} 服务") from exc
        if response.status_code != 200:
            raise ModelContractError("request_http_error", f"{_label(kind)} 服务返回非成功状态")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelContractError("response_not_json", f"{_label(kind)} 服务未返回 JSON") from exc
        if not isinstance(payload, dict) or isinstance(payload.get("error"), dict):
            raise ModelContractError("response_invalid", f"{_label(kind)} 服务响应格式不正确")
        return payload


def _join_url(base_url: str, suffix: str) -> str:
    return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _label(kind: ServiceKind) -> str:
    return "LLM" if kind == "llm" else "Embedding"
