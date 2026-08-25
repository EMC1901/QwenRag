from __future__ import annotations

import math

import httpx
import pytest

from qwenrag_runtime.deployment import DeploymentConfig, SecretsConfig, default_deployment
from qwenrag_runtime.model_contracts import ModelContractChecker, ModelContractError


def _deployment() -> DeploymentConfig:
    payload = default_deployment().model_dump(mode="json")
    payload["embedding"]["expected_dimension"] = 3
    payload["rag"]["embedding_dimension"] = 3
    return DeploymentConfig.model_validate(payload)


def _secrets() -> SecretsConfig:
    return SecretsConfig(
        local_rag_api_key="local", gateway_api_key="gateway", llm_upstream_api_key="upstream"
    )


def _success_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/health"):
        return httpx.Response(200)
    if request.url.path.endswith("/models"):
        model = "qwen" if request.url.port == 8001 else "qwen3-embedding-0.6b"
        return httpx.Response(200, json={"data": [{"id": model}]})
    if request.url.path.endswith("/chat/completions"):
        if request.headers.get("authorization") != "Bearer upstream":
            return httpx.Response(401)
        if request.content and b'"stream":true' in request.content:
            return httpx.Response(200, content=b"data: {\"choices\": []}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(200, json={"data": [{"embedding": [3, 4, 0]}]})
    raise AssertionError(f"Unexpected request: {request.url}")


def test_full_contract_checks_model_identity_requests_stream_and_normalizes_vector() -> None:
    client = httpx.Client(transport=httpx.MockTransport(_success_handler))

    results = ModelContractChecker(_deployment(), _secrets(), client=client).check_all()

    assert [result.kind for result in results] == ["llm", "embedding"]
    vector = results[1].normalized_probe_vector
    assert vector is not None
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)
    assert vector == pytest.approx((0.6, 0.8, 0.0))


def test_wrong_model_blocks_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("health"):
            return httpx.Response(200)
        return httpx.Response(200, json={"data": [{"id": "wrong"}]})

    checker = ModelContractChecker(_deployment(), _secrets(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ModelContractError, match="预期") as error:
        checker.check_llm(full=False)
    assert error.value.code == "model_mismatch"


@pytest.mark.parametrize(
    ("embedding", "code"),
    [([1, 2], "embedding_dimension"), ([float("nan"), 0, 0], "embedding_non_finite")],
)
def test_invalid_embedding_vector_is_rejected(embedding: list[float], code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("health"):
            return httpx.Response(200)
        if request.url.path.endswith("models"):
            return httpx.Response(200, json={"data": [{"id": "qwen3-embedding-0.6b"}]})
        if any(not math.isfinite(value) for value in embedding):
            return httpx.Response(200, content=b'{"data":[{"embedding":[NaN,0,0]}]}')
        return httpx.Response(200, json={"data": [{"embedding": embedding}]})

    checker = ModelContractChecker(_deployment(), _secrets(), client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ModelContractError) as error:
        checker.check_embedding()
    assert error.value.code == code


def test_readiness_retry_timeout_html_and_error_body_are_handled_safely() -> None:
    attempts = 0

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path.endswith("health"):
            attempts += 1
            return httpx.Response(503 if attempts == 1 else 200)
        return httpx.Response(200, content=b"<html>response-secret</html>")

    checker = ModelContractChecker(
        _deployment(), _secrets(), client=httpx.Client(transport=httpx.MockTransport(retry_handler)), ready_attempts=2
    )
    with pytest.raises(ModelContractError) as error:
        checker.check_llm(full=False)
    assert attempts == 2
    assert error.value.code == "response_not_json"
    assert "response-secret" not in str(error.value)

    timeout_client = httpx.Client(transport=httpx.MockTransport(lambda _: (_ for _ in ()).throw(httpx.ReadTimeout("timeout"))))
    with pytest.raises(ModelContractError) as timeout:
        ModelContractChecker(_deployment(), _secrets(), client=timeout_client, ready_attempts=1).check_llm(full=False)
    assert timeout.value.code == "ready_timeout"
