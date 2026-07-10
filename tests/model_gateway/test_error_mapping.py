"""Tests for unified model gateway error responses."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_gateway.errors import (
    GatewayError,
    authentication_error,
    error_response,
    gateway_error_response,
    internal_error,
    invalid_request_error,
    upstream_connection_error,
    upstream_http_error,
    upstream_timeout_error,
)
from model_gateway.main import app


def test_error_response_uses_openai_style_shape() -> None:
    response = error_response(
        status_code=400,
        message="Bad input",
        error_type="invalid_request_error",
        code="bad_input",
    )

    assert response.status_code == 400
    assert response.body == (
        b'{"error":{"message":"Bad input","type":"invalid_request_error",'
        b'"code":"bad_input"}}'
    )


def test_authentication_error_uses_unified_shape() -> None:
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "Unauthorized",
            "type": "authentication_error",
            "code": "invalid_api_key",
        }
    }


def test_gateway_error_exception_handler_returns_unified_shape() -> None:
    test_app = FastAPI()
    test_app.add_exception_handler(GatewayError, gateway_error_response)

    @test_app.get("/invalid")
    async def invalid_route() -> None:
        raise invalid_request_error("Missing model", "missing_model")

    response = TestClient(test_app).get("/invalid")

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Missing model",
            "type": "invalid_request_error",
            "code": "missing_model",
        }
    }


def test_error_factory_metadata() -> None:
    cases = [
        (authentication_error(), 401, "authentication_error", "invalid_api_key"),
        (
            invalid_request_error("Invalid JSON", "invalid_json"),
            400,
            "invalid_request_error",
            "invalid_json",
        ),
        (
            upstream_connection_error(),
            502,
            "upstream_error",
            "upstream_connection_failed",
        ),
        (upstream_timeout_error(), 504, "upstream_error", "upstream_timeout"),
        (upstream_http_error(503), 503, "upstream_error", "upstream_http_error"),
        (internal_error(), 500, "internal_error", "internal_server_error"),
    ]

    for exc, status_code, error_type, code in cases:
        assert exc.status_code == status_code
        assert exc.error_type == error_type
        assert exc.code == code
