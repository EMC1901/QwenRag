"""Error response helpers for the model gateway."""

from fastapi.responses import JSONResponse
from starlette.requests import Request


class GatewayError(Exception):
    """Base exception for errors intentionally returned by the gateway."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        error_type: str,
        code: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code


def error_payload(message: str, error_type: str, code: str) -> dict[str, dict[str, str]]:
    """Build an OpenAI-style error response payload."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


def error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str,
) -> JSONResponse:
    """Return a JSONResponse using the gateway's unified error format."""
    return JSONResponse(
        status_code=status_code,
        content=error_payload(message, error_type, code),
    )


def gateway_error_response(request: Request, exc: GatewayError) -> JSONResponse:
    """Convert a GatewayError into a FastAPI exception-handler response."""
    request.state.error_code = exc.code
    return error_response(
        status_code=exc.status_code,
        message=exc.message,
        error_type=exc.error_type,
        code=exc.code,
    )


def authentication_error(message: str = "Unauthorized") -> GatewayError:
    """Create an authentication error."""
    return GatewayError(
        status_code=401,
        message=message,
        error_type="authentication_error",
        code="invalid_api_key",
    )


def invalid_request_error(
    message: str,
    code: str = "invalid_request",
) -> GatewayError:
    """Create an invalid request error."""
    return GatewayError(
        status_code=400,
        message=message,
        error_type="invalid_request_error",
        code=code,
    )


def upstream_connection_error(
    message: str = "Upstream connection failed",
) -> GatewayError:
    """Create an upstream connection error."""
    return GatewayError(
        status_code=502,
        message=message,
        error_type="upstream_error",
        code="upstream_connection_failed",
    )


def upstream_timeout_error(message: str = "Upstream request timed out") -> GatewayError:
    """Create an upstream timeout error."""
    return GatewayError(
        status_code=504,
        message=message,
        error_type="upstream_error",
        code="upstream_timeout",
    )


def upstream_http_error(
    status_code: int,
    message: str = "Upstream HTTP error",
) -> GatewayError:
    """Create an upstream non-2xx HTTP error."""
    return GatewayError(
        status_code=status_code,
        message=message,
        error_type="upstream_error",
        code="upstream_http_error",
    )


def internal_error(message: str = "Internal server error") -> GatewayError:
    """Create an internal gateway error."""
    return GatewayError(
        status_code=500,
        message=message,
        error_type="internal_error",
        code="internal_server_error",
    )
