"""Privacy-preserving request logging and tracing helpers."""

import logging
import re
import uuid
from contextvars import ContextVar
from math import isfinite
from numbers import Real
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
_retrieval_context: ContextVar[dict[str, str] | None] = ContextVar(
    "retrieval_context",
    default=None,
)
_generation_context: ContextVar[dict[str, str] | None] = ContextVar(
    "generation_context",
    default=None,
)
_RAG_ROUTES = frozenset(
    {
        "router_disabled",
        "direct_llm",
        "retrieval_pending",
        "retrieval_ready",
        "retrieval_unavailable",
        "decision_unavailable",
        "rag_generation",
        "rag_no_evidence",
        "rag_generation_unavailable",
    }
)
_RAG_DECISION_SOURCES = frozenset({"", "llm", "fallback"})
_RETRIEVAL_MODES = frozenset({"vector", "hybrid"})


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


def record_retrieval_success(
    *,
    retrieval_mode: str,
    hit_count: int,
    vector_candidate_count: int,
    fts_candidate_count: int,
    duration_ms: float,
) -> None:
    """Attach aggregate retrieval metrics without retaining query or source contents."""
    if retrieval_mode not in _RETRIEVAL_MODES:
        raise ValueError("Unsupported retrieval mode")
    for value in (hit_count, vector_candidate_count, fts_candidate_count):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Retrieval counts must be non-negative integers")
    _record_retrieval_duration(duration_ms)
    context = _retrieval_context.get()
    if context is not None:
        context.update(
            {
                "mode": retrieval_mode,
                "hit_count": str(hit_count),
                "vector_candidate_count": str(vector_candidate_count),
                "fts_candidate_count": str(fts_candidate_count),
            }
        )


def record_retrieval_failure(*, duration_ms: float) -> None:
    """Attach only elapsed time when retrieval cannot produce a safe result."""
    _record_retrieval_duration(duration_ms)


def record_context_build_success(
    *,
    selected_hit_count: int,
    dropped_hit_count: int,
    estimated_input_tokens: int,
    estimated_evidence_tokens: int,
    estimated_history_tokens: int,
    truncated_hit_count: int,
) -> None:
    """Attach only numeric context-build aggregates to the active request."""
    values = {
        "selected_hit_count": selected_hit_count,
        "dropped_hit_count": dropped_hit_count,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_evidence_tokens": estimated_evidence_tokens,
        "estimated_history_tokens": estimated_history_tokens,
        "truncated_hit_count": truncated_hit_count,
    }
    for name, value in values.items():
        _require_nonnegative_int(value, f"Context {name}")
    context = _generation_context.get()
    if context is not None:
        context.update({name: str(value) for name, value in values.items()})


def record_generation_success(*, stream: bool, duration_ms: float) -> None:
    """Record safe generation mode and time after a completion or stream opens."""
    _record_generation_metrics(stream=stream, duration_ms=duration_ms)


def record_generation_failure(*, stream: bool, duration_ms: float) -> None:
    """Retain safe timing data when generation cannot produce a response."""
    _record_generation_metrics(stream=stream, duration_ms=duration_ms)


def _record_retrieval_duration(duration_ms: float) -> None:
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, Real)
        or not isfinite(duration_ms)
        or duration_ms < 0
    ):
        raise ValueError("Retrieval duration must be a finite non-negative number")
    context = _retrieval_context.get()
    if context is not None:
        context["duration_ms"] = f"{duration_ms:.2f}"


def _record_generation_metrics(*, stream: bool, duration_ms: float) -> None:
    if not isinstance(stream, bool):
        raise ValueError("Generation stream flag must be a boolean")
    _require_nonnegative_duration(duration_ms, "Generation duration")
    context = _generation_context.get()
    if context is not None:
        context.update(
            {
                "stream": str(stream).lower(),
                "duration_ms": f"{duration_ms:.2f}",
            }
        )


def _require_nonnegative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _require_nonnegative_duration(duration_ms: float, label: str) -> None:
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, Real)
        or not isfinite(duration_ms)
        or duration_ms < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")


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
        retrieval_token = _retrieval_context.set(
            {
                "mode": "",
                "hit_count": "",
                "vector_candidate_count": "",
                "fts_candidate_count": "",
                "duration_ms": "",
            }
        )
        generation_token = _generation_context.set(
            {
                "selected_hit_count": "",
                "dropped_hit_count": "",
                "estimated_input_tokens": "",
                "estimated_evidence_tokens": "",
                "estimated_history_tokens": "",
                "truncated_hit_count": "",
                "stream": "",
                "duration_ms": "",
            }
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
                "retrieval_mode=%s retrieval_hit_count=%s "
                "retrieval_vector_candidate_count=%s retrieval_fts_candidate_count=%s "
                "retrieval_duration_ms=%s context_selected_hit_count=%s "
                "context_dropped_hit_count=%s context_estimated_input_tokens=%s "
                "context_estimated_evidence_tokens=%s context_estimated_history_tokens=%s "
                "context_truncated_hit_count=%s generation_stream=%s "
                "generation_duration_ms=%s upstream_name=%s "
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
                (_retrieval_context.get() or {}).get("mode", ""),
                (_retrieval_context.get() or {}).get("hit_count", ""),
                (_retrieval_context.get() or {}).get("vector_candidate_count", ""),
                (_retrieval_context.get() or {}).get("fts_candidate_count", ""),
                (_retrieval_context.get() or {}).get("duration_ms", ""),
                (_generation_context.get() or {}).get("selected_hit_count", ""),
                (_generation_context.get() or {}).get("dropped_hit_count", ""),
                (_generation_context.get() or {}).get("estimated_input_tokens", ""),
                (_generation_context.get() or {}).get("estimated_evidence_tokens", ""),
                (_generation_context.get() or {}).get("estimated_history_tokens", ""),
                (_generation_context.get() or {}).get("truncated_hit_count", ""),
                (_generation_context.get() or {}).get("stream", ""),
                (_generation_context.get() or {}).get("duration_ms", ""),
                _upstream_name.get() or "",
                _upstream_status_code.get() or "",
                error_code,
            )
            _upstream_status_code.reset(upstream_status_token)
            _upstream_name.reset(upstream_name_token)
            _generation_context.reset(generation_token)
            _retrieval_context.reset(retrieval_token)
            _rag_routing_context.reset(rag_routing_token)
            _request_id.reset(request_token)

        return response
