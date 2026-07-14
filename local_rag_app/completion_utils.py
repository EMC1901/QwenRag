"""Reusable local OpenAI-compatible fixed completion helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from time import time
from uuid import uuid4

from local_rag_app.schemas import (
    AssistantMessage,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChunkChoice,
    CompletionChoice,
    Delta,
    Usage,
)


def build_fixed_completion(model: str, content: str) -> ChatCompletionResponse:
    """Create a normal local completion without pretending token usage is exact."""
    completion_id, created = new_completion_metadata()
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[CompletionChoice(message=AssistantMessage(content=content))],
        usage=Usage(),
    )


async def iter_fixed_sse_events(model: str, content: str) -> AsyncIterator[bytes]:
    """Yield the standard role, content, finish, and terminal SSE sequence."""
    completion_id, created = new_completion_metadata()
    yield encode_sse_event(
        ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChunkChoice(delta=Delta(role="assistant"))],
        )
    )
    yield encode_sse_event(
        ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChunkChoice(delta=Delta(content=content))],
        )
    )
    yield encode_sse_event(
        ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChunkChoice(delta=Delta(), finish_reason="stop")],
        )
    )
    yield b"data: [DONE]\n\n"


def encode_sse_event(chunk: ChatCompletionChunk) -> bytes:
    """Render one local completion chunk using standard SSE data framing."""
    return f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")


def new_completion_metadata() -> tuple[str, int]:
    """Return a local OpenAI-compatible identifier and Unix creation time."""
    return f"chatcmpl-local-{uuid4().hex}", int(time())
