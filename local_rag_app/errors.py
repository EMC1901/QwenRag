"""Expected-error types and OpenAI-compatible error responses."""

from fastapi import Request
from fastapi.responses import JSONResponse


class LocalRagError(Exception):
    """An expected failure that can be safely returned to an API client."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str,
        code: str,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code


def authentication_error() -> LocalRagError:
    """Create the standard response for a missing or invalid local API key."""
    return LocalRagError(
        "Unauthorized",
        status_code=401,
        error_type="authentication_error",
        code="invalid_api_key",
    )


def invalid_request_error(message: str, code: str) -> LocalRagError:
    """Create a stable 400 response for an invalid OpenAI-compatible request."""
    return LocalRagError(
        message,
        status_code=400,
        error_type="invalid_request_error",
        code=code,
    )


def service_not_ready_error(message: str, code: str) -> LocalRagError:
    """Report a configured mode whose implementation has not been added yet."""
    return LocalRagError(
        message,
        status_code=503,
        error_type="service_unavailable_error",
        code=code,
    )


def rag_decision_unavailable_error() -> LocalRagError:
    """Report that the application cannot safely choose a RAG route."""
    return service_not_ready_error(
        "Unable to decide whether knowledge retrieval is required",
        "rag_decision_unavailable",
    )


def rag_retrieval_not_ready_error() -> LocalRagError:
    """Report a private-data request before the retrieval stage is available."""
    return service_not_ready_error(
        "This question requires the local knowledge base, but retrieval is not ready",
        "rag_retrieval_not_ready",
    )


def gateway_connection_error() -> LocalRagError:
    """Hide model-gateway connection details behind a stable local API error."""
    return LocalRagError(
        "Model gateway connection failed",
        status_code=502,
        error_type="upstream_error",
        code="gateway_connection_failed",
    )


def gateway_timeout_error() -> LocalRagError:
    """Report an upstream model-gateway timeout without exposing internals."""
    return LocalRagError(
        "Model gateway request timed out",
        status_code=504,
        error_type="upstream_error",
        code="gateway_timeout",
    )


def gateway_auth_error() -> LocalRagError:
    """Report rejected service-to-service credentials as a local upstream error."""
    return LocalRagError(
        "Model gateway authentication failed",
        status_code=502,
        error_type="upstream_error",
        code="gateway_auth_failed",
    )


def gateway_http_error() -> LocalRagError:
    """Report a non-auth upstream HTTP failure without leaking its response body."""
    return LocalRagError(
        "Model gateway returned an error",
        status_code=502,
        error_type="upstream_error",
        code="gateway_http_error",
    )


def gateway_invalid_response_error() -> LocalRagError:
    """Report malformed gateway output instead of returning invalid OpenAI JSON."""
    return LocalRagError(
        "Model gateway returned an invalid response",
        status_code=502,
        error_type="upstream_error",
        code="gateway_invalid_response",
    )


def internal_server_error() -> LocalRagError:
    """Create a generic 500 response that contains no Python exception details."""
    return LocalRagError(
        "Internal server error",
        status_code=500,
        error_type="internal_error",
        code="internal_server_error",
    )


async def local_rag_error_response(
    request: Request,
    exc: LocalRagError,
) -> JSONResponse:
    """Render expected application failures in a stable OpenAI-like envelope."""
    request.state.error_code = exc.code
    return JSONResponse(
        status_code=exc.status_code,
        media_type="application/json; charset=utf-8",
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "code": exc.code,
            }
        },
    )


async def unexpected_error_response(
    request: Request,
    _: Exception,
) -> JSONResponse:
    """Hide unexpected exception details while preserving a traceable request ID."""
    return await local_rag_error_response(request, internal_server_error())
