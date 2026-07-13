"""HTTP routes for the Windows-local OpenAI-compatible RAG application."""

from json import JSONDecodeError
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from local_rag_app.answer_service import get_answer_service
from local_rag_app.auth import require_local_api_key
from local_rag_app.config import Settings, get_settings
from local_rag_app.errors import invalid_request_error
from local_rag_app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    ModelCard,
    ModelListResponse,
)


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return process-level health without probing the model gateway."""
    return HealthResponse(status="ok", service="local-rag-app")


@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    dependencies=[Depends(require_local_api_key)],
)
async def list_models(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelListResponse:
    """Expose the single local business model to OpenAI-compatible clients."""
    return ModelListResponse(data=[ModelCard(id=settings.local_rag_model)])


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(require_local_api_key)],
)
async def chat_completions(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Return an OpenAI-compatible completion selected by the local answer service."""
    request.state.answer_mode = settings.local_rag_answer_mode
    body = await _read_json_object(request)
    completion_request = _parse_chat_completion_request(
        body,
        expected_model=settings.local_rag_model,
    )
    answer_service = getattr(request.app.state, "answer_service", None)
    if answer_service is None:
        answer_service = get_answer_service(settings)

    if completion_request.stream:
        event_stream = await answer_service.stream(completion_request)
        return StreamingResponse(
            event_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    response = await answer_service.complete(completion_request)
    return JSONResponse(
        content=response.model_dump(mode="json"),
        media_type="application/json; charset=utf-8",
    )


async def _read_json_object(request: Request) -> dict:
    try:
        body = await request.json()
    except (JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
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


def _parse_chat_completion_request(
    body: dict,
    *,
    expected_model: str,
) -> ChatCompletionRequest:
    """Validate the first-release text-only chat request contract with stable codes."""
    if "model" not in body:
        raise invalid_request_error("Missing required field: model", "missing_model")
    if body["model"] != expected_model:
        raise invalid_request_error(
            f"Unsupported model: {body['model']}",
            "unsupported_model",
        )
    if "messages" not in body:
        raise invalid_request_error("Missing required field: messages", "missing_messages")
    if not isinstance(body["messages"], list) or not body["messages"]:
        raise invalid_request_error(
            "messages must be a non-empty array",
            "invalid_messages",
        )
    for message in body["messages"]:
        if not isinstance(message, dict):
            raise invalid_request_error(
                "Each message must be an object",
                "invalid_messages",
            )
        if message.get("role") not in {"system", "user", "assistant"}:
            raise invalid_request_error(
                "Each message must use role system, user, or assistant",
                "invalid_messages",
            )
        if not isinstance(message.get("content"), str):
            raise invalid_request_error(
                "Each message content must be a string in the first release",
                "invalid_messages",
            )
    if "stream" in body and not isinstance(body["stream"], bool):
        raise invalid_request_error("stream must be a boolean", "invalid_stream")

    try:
        return ChatCompletionRequest.model_validate(body)
    except ValidationError as exc:
        raise invalid_request_error(
            "Invalid chat completion request",
            "invalid_request",
        ) from exc
