"""Logging and request ID setup for the model gateway."""

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.responses import Response

from model_gateway.config import Settings, get_settings
from qwenrag_runtime.logging_setup import configure_component_logging
from qwenrag_runtime.paths import get_runtime_paths

REQUEST_ID_HEADER = "X-Request-ID"
LOGGER_NAME = "model_gateway"


def configure_logging(settings: Settings | None = None) -> None:
    """Configure process-level logging for the gateway."""
    settings = settings or get_settings()
    configure_component_logging(
        LOGGER_NAME, settings.log_level, get_runtime_paths().log_root / "gateway"
    )


def add_request_logging_middleware(app: FastAPI) -> None:
    """Add middleware that manages request IDs and access logs."""
    logger = logging.getLogger(LOGGER_NAME)

    @app.middleware("http")
    async def request_logging_middleware(
        request: Request,
        call_next,
    ) -> Response:
        request_id = _resolve_request_id(request)
        request.state.request_id = request_id
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started_at) * 1000
            logger.exception(
                "request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f "
                "client_host=%s error_code=%s",
                request_id,
                request.method,
                request.url.path,
                500,
                duration_ms,
                request.client.host if request.client else "",
                "unhandled_exception",
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f "
            "client_host=%s error_code=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request.client.host if request.client else "",
            getattr(request.state, "error_code", ""),
        )
        return response


def get_request_id(request: Request) -> str:
    """Return the request ID assigned by middleware."""
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        return str(request_id)
    return _resolve_request_id(request)


def _resolve_request_id(request: Request) -> str:
    request_id = request.headers.get(REQUEST_ID_HEADER)
    if request_id and request_id.strip():
        return request_id.strip()
    return uuid4().hex
