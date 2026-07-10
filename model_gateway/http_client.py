"""Upstream HTTP client helpers for the model gateway."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
import logging
from time import perf_counter
from typing import Any

import httpx

from model_gateway.config import Settings, get_settings
from model_gateway.errors import (
    GatewayError,
    upstream_connection_error,
    upstream_http_error,
    upstream_timeout_error,
)
from model_gateway.logging_config import LOGGER_NAME

HOP_BY_HOP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "authorization",
}


def join_url(base_url: str, path: str) -> str:
    """Join an upstream base URL and API path without duplicate slashes."""
    normalized_base = base_url.strip().rstrip("/")
    normalized_path = path.strip()
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    return f"{normalized_base}{normalized_path}"


def build_timeout(settings: Settings | None = None) -> httpx.Timeout:
    """Build an httpx timeout from gateway settings."""
    settings = settings or get_settings()
    return httpx.Timeout(
        connect=settings.http_connect_timeout_seconds,
        read=settings.http_read_timeout_seconds,
        write=settings.http_write_timeout_seconds,
        pool=settings.http_pool_timeout_seconds,
    )


def build_upstream_headers(
    *,
    upstream_api_key: str | None = None,
    request_id: str | None = None,
    source_headers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build safe headers for an upstream model request."""
    headers = {"Content-Type": "application/json"}

    source_headers = source_headers or {}
    if not request_id:
        for key, value in source_headers.items():
            if key.lower() == "x-request-id":
                request_id = value
                break

    if request_id:
        headers["X-Request-ID"] = request_id
    if upstream_api_key:
        headers["Authorization"] = f"Bearer {upstream_api_key}"

    return headers


def filter_forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Remove connection and credential headers that must not be proxied."""
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _create_client(settings: Settings | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=build_timeout(settings))


async def call_json(
    upstream_name: str,
    base_url: str,
    path: str,
    body: Any | None = None,
    *,
    method: str = "POST",
    upstream_api_key: str | None = None,
    request_id: str | None = None,
    source_headers: Mapping[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> Any:
    """Call an upstream JSON endpoint and return the parsed JSON body."""
    close_client = client is None
    active_client = client or _create_client(settings)
    url = join_url(base_url, path)
    headers = build_upstream_headers(
        upstream_api_key=upstream_api_key,
        request_id=request_id,
        source_headers=source_headers,
    )
    started_at = perf_counter()

    try:
        response = await active_client.request(
            method.upper(),
            url,
            json=body if body is not None else None,
            headers=headers,
        )
        duration_ms = (perf_counter() - started_at) * 1000
        if not 200 <= response.status_code < 300:
            _log_upstream_response(
                upstream_name,
                url,
                request_id,
                response.status_code,
                duration_ms,
                error_code="upstream_http_error",
            )
            _raise_for_upstream_status(upstream_name, response)

        try:
            result = response.json()
        except ValueError as exc:
            _log_upstream_response(
                upstream_name,
                url,
                request_id,
                response.status_code,
                duration_ms,
                error_code="upstream_connection_failed",
            )
            raise upstream_connection_error(
                f"{upstream_name} upstream returned invalid JSON"
            ) from exc
        _log_upstream_response(
            upstream_name,
            url,
            request_id,
            response.status_code,
            duration_ms,
        )
        return result
    except httpx.TimeoutException as exc:
        duration_ms = (perf_counter() - started_at) * 1000
        _log_upstream_error(
            upstream_name,
            url,
            request_id,
            duration_ms,
            error_code="upstream_timeout",
        )
        raise upstream_timeout_error(f"{upstream_name} upstream request timed out") from exc
    except httpx.TransportError as exc:
        duration_ms = (perf_counter() - started_at) * 1000
        _log_upstream_error(
            upstream_name,
            url,
            request_id,
            duration_ms,
            error_code="upstream_connection_failed",
        )
        raise upstream_connection_error(
            f"{upstream_name} upstream connection failed"
        ) from exc
    finally:
        if close_client:
            await active_client.aclose()


@asynccontextmanager
async def stream(
    upstream_name: str,
    base_url: str,
    path: str,
    body: Any,
    *,
    upstream_api_key: str | None = None,
    request_id: str | None = None,
    source_headers: Mapping[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[httpx.Response]:
    """Stream a POST request to an upstream model endpoint."""
    close_client = client is None
    active_client = client or _create_client(settings)
    url = join_url(base_url, path)
    headers = build_upstream_headers(
        upstream_api_key=upstream_api_key,
        request_id=request_id,
        source_headers=source_headers,
    )
    started_at = perf_counter()

    try:
        async with active_client.stream("POST", url, json=body, headers=headers) as response:
            duration_ms = (perf_counter() - started_at) * 1000
            if not 200 <= response.status_code < 300:
                _log_upstream_response(
                    upstream_name,
                    url,
                    request_id,
                    response.status_code,
                    duration_ms,
                    error_code="upstream_http_error",
                )
                _raise_for_upstream_status(upstream_name, response)

            _log_upstream_response(
                upstream_name,
                url,
                request_id,
                response.status_code,
                duration_ms,
            )
            yield response
    except httpx.TimeoutException as exc:
        duration_ms = (perf_counter() - started_at) * 1000
        _log_upstream_error(
            upstream_name,
            url,
            request_id,
            duration_ms,
            error_code="upstream_timeout",
        )
        raise upstream_timeout_error(f"{upstream_name} upstream request timed out") from exc
    except httpx.TransportError as exc:
        duration_ms = (perf_counter() - started_at) * 1000
        _log_upstream_error(
            upstream_name,
            url,
            request_id,
            duration_ms,
            error_code="upstream_connection_failed",
        )
        raise upstream_connection_error(
            f"{upstream_name} upstream connection failed"
        ) from exc
    finally:
        if close_client:
            await active_client.aclose()


async def check_models(
    upstream_name: str,
    base_url: str,
    *,
    upstream_api_key: str | None = None,
    request_id: str | None = None,
    client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> Any:
    """Call an upstream /models endpoint and return its JSON response."""
    return await call_json(
        upstream_name,
        base_url,
        "/models",
        None,
        method="GET",
        upstream_api_key=upstream_api_key,
        request_id=request_id,
        client=client,
        settings=settings,
    )


def _raise_for_upstream_status(upstream_name: str, response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return

    raise upstream_http_error(
        response.status_code,
        f"{upstream_name} upstream returned HTTP {response.status_code}",
    )


def _log_upstream_response(
    upstream_name: str,
    upstream_url: str,
    request_id: str | None,
    status_code: int,
    duration_ms: float,
    *,
    error_code: str = "",
) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    log_level = logging.WARNING if error_code else logging.INFO
    logger.log(
        log_level,
        "request_id=%s upstream_name=%s upstream_url=%s "
        "upstream_status_code=%s duration_ms=%.2f error_code=%s",
        request_id or "",
        upstream_name,
        upstream_url,
        status_code,
        duration_ms,
        error_code,
    )


def _log_upstream_error(
    upstream_name: str,
    upstream_url: str,
    request_id: str | None,
    duration_ms: float,
    *,
    error_code: str,
) -> None:
    logging.getLogger(LOGGER_NAME).warning(
        "request_id=%s upstream_name=%s upstream_url=%s "
        "upstream_status_code=%s duration_ms=%.2f error_code=%s",
        request_id or "",
        upstream_name,
        upstream_url,
        "",
        duration_ms,
        error_code,
    )


__all__ = [
    "HOP_BY_HOP_HEADERS",
    "GatewayError",
    "build_timeout",
    "build_upstream_headers",
    "call_json",
    "check_models",
    "filter_forward_headers",
    "join_url",
    "stream",
]
