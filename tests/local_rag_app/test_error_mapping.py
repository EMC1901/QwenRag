"""Error-envelope tests for client-safe local application failures."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from local_rag_app.errors import (
    gateway_connection_error,
    missing_retrieval_query_error,
    rag_answer_generation_not_ready_error,
    rag_context_build_error,
    rag_decision_unavailable_error,
    rag_knowledge_base_unavailable_error,
    rag_retrieval_unavailable_error,
    rag_retrieval_not_ready_error,
    reference_display_error,
)
from local_rag_app.main import create_app


def test_invalid_json_uses_openai_style_400_error() -> None:
    """Malformed client JSON must not fall back to FastAPI's default 422 payload."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer none",
                "Content-Type": "application/json",
            },
            content=b"{not-json",
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "Request body must be valid JSON",
        "type": "invalid_request_error",
        "code": "invalid_json",
    }


def test_expected_gateway_error_uses_unified_502_response() -> None:
    """Expected upstream errors must not reveal their internal exception details."""
    app = create_app()

    @app.get("/_test/gateway-error")
    async def raise_gateway_error() -> None:
        raise gateway_connection_error()

    with TestClient(app) as client:
        response = client.get("/_test/gateway-error")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "gateway_connection_failed"


def test_rag_router_errors_use_documented_safe_503_contracts() -> None:
    """Before retrieval exists, router failures must be explicit and non-sensitive."""
    decision_error = rag_decision_unavailable_error()
    retrieval_error = rag_retrieval_not_ready_error()

    assert (
        decision_error.status_code,
        decision_error.error_type,
        decision_error.code,
        decision_error.message,
    ) == (
        503,
        "service_unavailable_error",
        "rag_decision_unavailable",
        "Unable to decide whether knowledge retrieval is required",
    )
    assert (
        retrieval_error.status_code,
        retrieval_error.error_type,
        retrieval_error.code,
        retrieval_error.message,
    ) == (
        503,
        "service_unavailable_error",
        "rag_retrieval_not_ready",
        "This question requires the local knowledge base, but retrieval is not ready",
    )


def test_local_retrieval_errors_use_stable_safe_contracts() -> None:
    """Stage-1 retrieval failures must not expose paths, queries, or tracebacks."""
    errors = [
        missing_retrieval_query_error(),
        rag_knowledge_base_unavailable_error(),
        rag_retrieval_unavailable_error(),
        rag_answer_generation_not_ready_error(),
        rag_context_build_error(),
        reference_display_error(),
    ]

    assert [
        (error.status_code, error.error_type, error.code, error.message)
        for error in errors
    ] == [
        (
            400,
            "invalid_request_error",
            "missing_retrieval_query",
            "A non-empty user query is required for knowledge retrieval",
        ),
        (
            503,
            "service_unavailable_error",
            "rag_knowledge_base_unavailable",
            "The local knowledge base is unavailable",
        ),
        (
            503,
            "service_unavailable_error",
            "rag_retrieval_unavailable",
            "Local knowledge retrieval failed",
        ),
        (
            503,
            "service_unavailable_error",
            "rag_answer_generation_not_ready",
            "Knowledge retrieval succeeded, but RAG answer generation is not ready",
        ),
        (
            503,
            "service_unavailable_error",
            "rag_context_build_failed",
            "The retrieved context could not be prepared safely",
        ),
        (
            503,
            "service_unavailable_error",
            "reference_display_failed",
            "The answer references could not be prepared safely",
        ),
    ]


def test_unexpected_error_is_a_safe_500_with_request_id() -> None:
    """Python exception text and tracebacks must never be sent to the client."""
    app = create_app()

    @app.get("/_test/unexpected-error")
    async def raise_unexpected_error() -> None:
        raise RuntimeError("private diagnostic detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            "/_test/unexpected-error",
            headers={"X-Request-ID": "safe-error-1"},
        )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "safe-error-1"
    assert response.json() == {
        "error": {
            "message": "Internal server error",
            "type": "internal_error",
            "code": "internal_server_error",
        }
    }
    assert "private diagnostic detail" not in response.text
