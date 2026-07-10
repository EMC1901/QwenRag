"""HTTP routes for the model gateway."""

from collections.abc import AsyncIterator
from json import JSONDecodeError
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from model_gateway.auth import require_api_key
from model_gateway.config import Settings, get_settings
from model_gateway.errors import GatewayError, invalid_request_error
from model_gateway.http_client import call_json, check_models, stream as stream_upstream
from model_gateway.logging_config import get_request_id

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Return process-level health for the model gateway."""
    return {"status": "ok", "service": "model-gateway"}


@router.get("/health/upstreams", dependencies=[Depends(require_api_key)])
async def health_upstreams(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return health details for configured upstream model services."""
    llm = await _check_upstream(
        "llm",
        settings.llm_base_url,
        upstream_api_key=settings.llm_upstream_api_key,
        request_id=get_request_id(request),
        settings=settings,
    )
    embedding = await _check_upstream(
        "embedding",
        settings.embedding_base_url,
        upstream_api_key=settings.embedding_upstream_api_key,
        request_id=get_request_id(request),
        settings=settings,
    )
    overall_status = "ok" if llm["ok"] and embedding["ok"] else "degraded"

    return {
        "status": overall_status,
        "upstreams": {
            "llm": llm,
            "embedding": embedding,
        },
    }


@router.get("/v1/models", dependencies=[Depends(require_api_key)])
async def models(settings: Settings = Depends(get_settings)) -> dict:
    """Return configured models exposed by the gateway."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.llm_model,
                "object": "model",
                "owned_by": "model-gateway",
            },
            {
                "id": settings.embedding_model,
                "object": "model",
                "owned_by": "model-gateway",
            },
        ],
    }


@router.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> Response:
    """Proxy chat completion requests to the LLM upstream."""
    body = await _read_json_body(request)
    _validate_chat_completion_body(body)

    if body.get("stream") is True:
        return await _open_chat_stream(request, body, settings)

    result = await call_json(
        "llm",
        settings.llm_base_url,
        "/chat/completions",
        body,
        upstream_api_key=settings.llm_upstream_api_key or None,
        request_id=get_request_id(request),
        source_headers=request.headers,
        settings=settings,
    )
    return JSONResponse(content=result)


@router.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
async def embeddings(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """Proxy embedding requests to the embedding upstream."""
    body = await _read_json_body(request)
    _validate_embeddings_body(body)

    result = await call_json(
        "embedding",
        settings.embedding_base_url,
        "/embeddings",
        body,
        upstream_api_key=settings.embedding_upstream_api_key or None,
        request_id=get_request_id(request),
        source_headers=request.headers,
        settings=settings,
    )
    return JSONResponse(content=result)


async def _check_upstream(
    upstream_name: str,
    base_url: str,
    *,
    upstream_api_key: str,
    request_id: str,
    settings: Settings,
) -> dict[str, object]:
    result: dict[str, object] = {"ok": False, "base_url": base_url}
    try:
        await check_models(
            upstream_name,
            base_url,
            upstream_api_key=upstream_api_key or None,
            request_id=request_id,
            settings=settings,
        )
    except GatewayError as exc:
        result["error"] = exc.message
        result["code"] = exc.code
        return result
    except Exception:
        result["error"] = "Unexpected upstream health check failure"
        result["code"] = "internal_server_error"
        return result

    result["ok"] = True
    return result


async def _read_json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except (JSONDecodeError, ValueError) as exc:
        raise invalid_request_error(
            "Request body must be valid JSON",
            "invalid_json",
        ) from exc

    if not isinstance(body, dict):
        raise invalid_request_error(
            "Request body must be a JSON object",
            "invalid_json",
        )
    return body


def _validate_chat_completion_body(body: dict[str, Any]) -> None:
    if "model" not in body:
        raise invalid_request_error(
            "Missing required field: model",
            "missing_model",
        )
    if "messages" not in body:
        raise invalid_request_error(
            "Missing required field: messages",
            "missing_messages",
        )
    if not isinstance(body["messages"], list):
        raise invalid_request_error(
            "messages must be an array",
            "invalid_messages",
        )


def _validate_embeddings_body(body: dict[str, Any]) -> None:
    if "model" not in body:
        raise invalid_request_error(
            "Missing required field: model",
            "missing_model",
        )
    if "input" not in body:
        raise invalid_request_error(
            "Missing required field: input",
            "missing_input",
        )


async def _open_chat_stream(
    request: Request,
    body: dict[str, Any],
    settings: Settings,
) -> StreamingResponse:
    upstream_context = stream_upstream(
        "llm",
        settings.llm_base_url,
        "/chat/completions",
        body,
        upstream_api_key=settings.llm_upstream_api_key or None,
        request_id=get_request_id(request),
        source_headers=request.headers,
        settings=settings,
    )
    upstream_response = await upstream_context.__aenter__()
    media_type = upstream_response.headers.get("content-type", "text/event-stream")

    return StreamingResponse(
        _iter_upstream_stream(upstream_context, upstream_response),
        media_type=media_type,
    )


async def _iter_upstream_stream(
    upstream_context,
    upstream_response,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream_response.aiter_bytes():
            yield chunk
    except BaseException as exc:
        await upstream_context.__aexit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        await upstream_context.__aexit__(None, None, None)
