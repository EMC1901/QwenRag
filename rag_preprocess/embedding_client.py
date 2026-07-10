"""调用 OpenAI 兼容 embedding 服务生成向量。"""

import math
import os
import time
from dataclasses import dataclass

import requests


@dataclass
class EmbeddingResult:
    """单次 embedding 结果。"""

    success: bool
    vector: list[float] | None = None
    error_message: str | None = None
    retry_count: int = 0


def get_embedding_base_url() -> str:
    """读取 embedding 服务地址，默认使用本机 llama.cpp 服务。"""
    return os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8002/v1").rstrip("/")


def get_embedding_api_key() -> str:
    """读取 embedding API Key。llama.cpp 本地服务通常可设为 none。"""
    return os.getenv("EMBEDDING_API_KEY", "none")


def _headers(api_key: str | None = None) -> dict[str, str]:
    api_key = api_key if api_key is not None else get_embedding_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _embedding_endpoint(base_url: str | None = None) -> str:
    base = (base_url or get_embedding_base_url()).rstrip("/")
    return f"{base}/embeddings"


def _trust_env_proxy() -> bool:
    """是否沿用系统代理环境变量。内网 embedding 默认不走代理。"""
    value = os.getenv("EMBEDDING_TRUST_ENV_PROXY", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _post_embeddings(
    *,
    base_url: str | None,
    payload: dict,
    timeout: int,
):
    """发送 embedding 请求。默认绕过 HTTP_PROXY/HTTPS_PROXY。"""
    if _trust_env_proxy():
        return requests.post(
            _embedding_endpoint(base_url),
            json=payload,
            headers=_headers(),
            timeout=timeout,
        )

    session = requests.Session()
    session.trust_env = False
    try:
        return session.post(
            _embedding_endpoint(base_url),
            json=payload,
            headers=_headers(),
            timeout=timeout,
        )
    finally:
        session.close()


def _parse_embedding_response(data: dict, expected_count: int) -> list[EmbeddingResult]:
    items = data.get("data")
    if not isinstance(items, list):
        message = "embedding 响应缺少 data 列表"
        return [EmbeddingResult(success=False, error_message=message)
                for _ in range(expected_count)]

    ordered = sorted(items, key=lambda item: item.get("index", 0))
    results: list[EmbeddingResult] = []
    for item in ordered:
        vector = item.get("embedding")
        if isinstance(vector, list):
            results.append(EmbeddingResult(success=True, vector=vector))
        else:
            results.append(EmbeddingResult(
                success=False,
                error_message="embedding 响应项缺少 embedding 向量",
            ))

    while len(results) < expected_count:
        results.append(EmbeddingResult(
            success=False,
            error_message="embedding 响应数量少于请求数量",
        ))

    return results[:expected_count]


def embed_text(
    text: str,
    model: str = "qwen3-embedding-0.6b",
    *,
    base_url: str | None = None,
    timeout: int | None = None,
    max_retries: int = 2,
) -> EmbeddingResult:
    """调用 embedding 服务，返回一个向量。"""
    result = embed_batch(
        [text],
        model=model,
        batch_size=1,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return result[0] if result else EmbeddingResult(
        success=False,
        error_message="embedding_batch 未返回结果",
    )


def embed_batch(
    texts: list[str],
    model: str = "qwen3-embedding-0.6b",
    batch_size: int = 128,
    *,
    base_url: str | None = None,
    timeout: int | None = None,
    max_retries: int = 2,
) -> list[EmbeddingResult]:
    """批量向量化。"""
    results: list[EmbeddingResult] = []
    timeout = timeout or int(os.getenv("EMBEDDING_TIMEOUT", "120"))

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {"model": model, "input": batch}
        retry_count = 0
        last_error: str | None = None

        while retry_count <= max_retries:
            try:
                response = _post_embeddings(
                    base_url=base_url,
                    payload=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                batch_results = _parse_embedding_response(response.json(), len(batch))
                for result in batch_results:
                    result.retry_count = retry_count
                results.extend(batch_results)
                break
            except Exception as exc:
                last_error = str(exc)
                if retry_count >= max_retries:
                    results.extend([
                        EmbeddingResult(
                            success=False,
                            error_message=last_error,
                            retry_count=retry_count,
                        )
                        for _ in batch
                    ])
                    break
                sleep_seconds = min(2 ** retry_count, 8)
                time.sleep(sleep_seconds)
                retry_count += 1

    return results


def validate_embedding(vector: list[float], expected_dim: int) -> bool:
    """检查向量维度。"""
    if len(vector) != expected_dim:
        return False
    return all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in vector)


def normalize_embedding(vector: list[float]) -> list[float]:
    """L2 归一化向量，用于内积相似度。"""
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(x) / norm for x in vector]
