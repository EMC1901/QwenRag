"""Privacy-preserving request logging and tracing helpers."""

import logging
import re
import uuid
from contextvars import ContextVar
from time import perf_counter

from fastapi import FastAPI, Request, Response

from local_rag_app.config import get_settings


REQUEST_ID_HEADER = "X-Request-ID"
LOGGER_NAME = "local_rag_app"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_upstream_name: ContextVar[str | None] = ContextVar("upstream_name", default=None)
_upstream_status_code: ContextVar[int | None] = ContextVar(
    "upstream_status_code",
    default=None,
)
_rag_routing_context: ContextVar[dict[str, str] | None] = ContextVar(
    "rag_routing_context",
    default=None,
)
_RAG_ROUTES = frozenset(
    {
        "router_disabled",
        "direct_llm",
        "retrieval_pending",
        "decision_unavailable",
    }
)
_RAG_DECISION_SOURCES = frozenset({"", "llm", "fallback"})


def configure_logging(level: str) -> None:
    """Configure a named logger without ever enabling request-body logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger(LOGGER_NAME).setLevel(level)


def get_request_id() -> str | None:
    """Return the request ID associated with the current request, if any."""
    return _request_id.get()


def record_upstream_result(name: str, status_code: int | None = None) -> None:
    """Attach safe upstream diagnostic fields to the active request context."""
    _upstream_name.set(name)
    _upstream_status_code.set(status_code)


def record_rag_route(route: str, source: str = "") -> None:
    """Attach privacy-safe RAG routing fields to the active request context."""
    if route not in _RAG_ROUTES:
        raise ValueError("Unsupported RAG route")
    if source not in _RAG_DECISION_SOURCES:
        raise ValueError("Unsupported RAG decision source")
    context = _rag_routing_context.get()
    if context is not None:
        # Starlette may run an endpoint in a child task. Mutating this request-
        # scoped mapping keeps the safe routing result visible to middleware.
        context["route"] = route
        context["source"] = source


def _resolve_request_id(request: Request) -> str:
    """Use a safe caller ID when present, otherwise issue a new UUID."""
    supplied = request.headers.get(REQUEST_ID_HEADER)
    if supplied and _REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied
    return uuid.uuid4().hex


def add_request_id_middleware(app: FastAPI) -> None:
    """Attach request IDs and write one metadata-only log line per request."""

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        request_id = _resolve_request_id(request)
        request.state.request_id = request_id
        request_token = _request_id.set(request_id)
        upstream_name_token = _upstream_name.set(None)
        upstream_status_token = _upstream_status_code.set(None)
        rag_routing_token = _rag_routing_context.set(
            {"route": "router_disabled", "source": ""}
        )
        started_at = perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
        except Exception as exc:
            # ServerErrorMiddleware otherwise creates its response outside this
            # middleware, which would omit our request ID and audit-log fields.
            from local_rag_app.errors import unexpected_error_response

            response = await unexpected_error_response(request, exc)
        finally:
            duration_ms = (perf_counter() - started_at) * 1000
            status_code = response.status_code if response is not None else 500
            if response is not None:
                response.headers[REQUEST_ID_HEADER] = request_id

            client = request.client
            client_host = client.host if client is not None else ""
            error_code = getattr(request.state, "error_code", "")
            answer_mode = getattr(
                request.state,
                "answer_mode",
                get_settings().local_rag_answer_mode,
            )
            logging.getLogger(LOGGER_NAME).info(
                "request_id=%s method=%s path=%s status_code=%s duration_ms=%.2f "
                "client_host=%s answer_mode=%s rag_route=%s rag_decision_source=%s "
                "upstream_name=%s "
                "upstream_status_code=%s error_code=%s",
                request_id,
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                client_host,
                answer_mode,
                (_rag_routing_context.get() or {}).get("route", "router_disabled"),
                (_rag_routing_context.get() or {}).get("source", ""),
                _upstream_name.get() or "",
                _upstream_status_code.get() or "",
                error_code,
            )
            _upstream_status_code.reset(upstream_status_token)
            _upstream_name.reset(upstream_name_token)
            _rag_routing_context.reset(rag_routing_token)
            _request_id.reset(request_token)

        return response
