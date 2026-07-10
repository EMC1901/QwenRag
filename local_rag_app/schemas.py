"""Request and response schemas for the local OpenAI-compatible API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr


class HealthResponse(BaseModel):
    """Reserved response shape for the stage 3 process-health endpoint."""

    status: str
    service: str


class ModelCard(BaseModel):
    """Minimal OpenAI-compatible model descriptor exposed to Chatbox."""

    id: str
    object: str = "model"
    owned_by: str = "local-rag-app"


class ModelListResponse(BaseModel):
    """OpenAI-compatible response returned by ``GET /v1/models``."""

    object: str = "list"
    data: list[ModelCard]


class ChatMessage(BaseModel):
    """The text-only message subset supported by the first local API release."""

    model_config = ConfigDict(extra="allow", strict=True)

    role: Literal["system", "user", "assistant"]
    content: StrictStr


class ChatCompletionRequest(BaseModel):
    """Accepted OpenAI chat-completion fields, preserving unknown safe options."""

    model_config = ConfigDict(extra="allow", strict=True)

    model: StrictStr
    messages: list[ChatMessage]
    stream: StrictBool = False


class AssistantMessage(BaseModel):
    """Assistant message included in a non-streaming completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str


class CompletionChoice(BaseModel):
    """One non-streaming OpenAI-compatible completion choice."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    """Token usage reported by deterministic stub responses."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible non-streaming chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class Delta(BaseModel):
    """Partial assistant output inside a streaming completion chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    """One SSE completion delta."""

    index: int = 0
    delta: Delta
    finish_reason: Literal["stop"] | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible chunk emitted when ``stream=true``."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
