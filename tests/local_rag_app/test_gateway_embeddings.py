"""Mock-only tests for the local application's model-gateway embedding client."""

import json

import httpx
import pytest

from local_rag_app.config import Settings
from local_rag_app.errors import LocalRagError
from local_rag_app.gateway_client import ModelGatewayClient


def embedding_settings(*, dimension: int = 3) -> Settings:
    """Create an isolated gateway configuration with a small test vector size."""
    return Settings(
        LOCAL_RAG_ANSWER_MODE="gateway",
        MODEL_GATEWAY_BASE_URL="http://gateway.test:8010/v1/",
        MODEL_GATEWAY_API_KEY="test-service-key",
        UPSTREAM_LLM_MODEL="qwen",
        UPSTREAM_EMBEDDING_MODEL="embed-test",
        RAG_EMBEDDING_DIM=dimension,
        _env_file=None,
    )


@pytest.mark.asyncio
async def test_create_embedding_uses_gateway_contract_and_returns_vector() -> None:
    """One query uses the configured gateway URL, credential, model, and request ID."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "embed-test",
                "data": [{"object": "embedding", "index": 0, "embedding": [1, "2", 3.0]}],
            },
            request=request,
        )

    client = ModelGatewayClient(
        embedding_settings(),
        transport=httpx.MockTransport(handler),
    )
    vector = await client.create_embedding(
        "private query must not be logged",
        request_id="embedding-request-1",
    )

    assert vector == [1.0, 2.0, 3.0]
    assert captured["url"] == "http://gateway.test:8010/v1/embeddings"
    assert captured["body"] == {
        "model": "embed-test",
        "input": "private query must not be logged",
    }
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer test-service-key"
    assert headers["x-request-id"] == "embedding-request-1"


@pytest.mark.asyncio
async def test_create_embedding_accepts_gateway_response_without_model_field() -> None:
    """Some compatible gateways omit the optional model echo in successful JSON."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            request=request,
        )

    client = ModelGatewayClient(
        embedding_settings(),
        transport=httpx.MockTransport(handler),
    )

    assert await client.create_embedding("query") == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"data": []},
        {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]},
        {"data": [{"embedding": "not-a-list"}]},
        {"data": [{"embedding": [0.1, 0.2]}]},
        {"data": [{"embedding": [0.0, 0.0, 0.0]}]},
        {"model": "other-embedding-model", "data": [{"embedding": [0.1, 0.2, 0.3]}]},
    ],
)
async def test_create_embedding_rejects_invalid_structures(payload: object) -> None:
    """Malformed data, wrong dimensions, zero vectors, and model mismatches fail safely."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    client = ModelGatewayClient(
        embedding_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LocalRagError) as error:
        await client.create_embedding("private query")

    assert error.value.code == "gateway_invalid_response"
    assert "private query" not in str(error.value)
    assert "test-service-key" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector",
    [
        [True, 0.2, 0.3],
        ["not-a-number", 0.2, 0.3],
    ],
)
async def test_create_embedding_rejects_non_finite_or_non_numeric_values(
    vector: list[object],
) -> None:
    """Vectors must be finite numeric values suitable for later normalization."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"embedding": vector}]},
            request=request,
        )

    client = ModelGatewayClient(
        embedding_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LocalRagError) as error:
        await client.create_embedding("private query")

    assert error.value.code == "gateway_invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize("special_number", ["NaN", "Infinity"])
async def test_create_embedding_rejects_non_finite_json_numbers(
    special_number: str,
) -> None:
    """Non-standard JSON numeric values must not enter the retrieval pipeline."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                '{"data":[{"embedding":['
                f"{special_number},0.2,0.3"
                "]}]}"
            ).encode("utf-8"),
            request=request,
        )

    client = ModelGatewayClient(
        embedding_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LocalRagError) as error:
        await client.create_embedding("private query")

    assert error.value.code == "gateway_invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [(401, "gateway_auth_failed"), (403, "gateway_auth_failed"), (500, "gateway_http_error")],
)
async def test_create_embedding_maps_gateway_http_errors(
    status_code: int,
    expected_code: str,
) -> None:
    """Embedding calls reuse the existing stable upstream HTTP error contract."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "private upstream body"}, request=request)

    client = ModelGatewayClient(
        embedding_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LocalRagError) as error:
        await client.create_embedding("private query")

    assert error.value.code == expected_code
    assert "private upstream body" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "expected_code"),
    [
        (httpx.ConnectError("private connection detail"), "gateway_connection_failed"),
        (httpx.ReadTimeout("private timeout detail"), "gateway_timeout"),
    ],
)
async def test_create_embedding_maps_transport_errors(
    exception: httpx.TransportError,
    expected_code: str,
) -> None:
    """Connection and timeout failures remain client-safe and stable."""
    def handler(_: httpx.Request) -> httpx.Response:
        raise exception

    client = ModelGatewayClient(
        embedding_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LocalRagError) as error:
        await client.create_embedding("private query")

    assert error.value.code == expected_code
    assert "private" not in str(error.value)


@pytest.mark.asyncio
async def test_create_embedding_records_only_embedding_upstream_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client identifies the embedding upstream without logging query content."""
    events: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        "local_rag_app.gateway_client.record_upstream_result",
        lambda name, status_code=None: events.append((name, status_code)),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            request=request,
        )

    client = ModelGatewayClient(
        embedding_settings(),
        transport=httpx.MockTransport(handler),
    )
    await client.create_embedding("private query")

    assert events == [
        ("model_gateway_embedding", None),
        ("model_gateway_embedding", 200),
    ]
