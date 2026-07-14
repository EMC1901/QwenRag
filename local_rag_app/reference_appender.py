"""Append deterministic references to normal and streaming chat completions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from json import JSONDecodeError, dumps, loads
from typing import Any

from local_rag_app.errors import gateway_invalid_response_error
from local_rag_app.reference_models import ReferenceBuildResult
from local_rag_app.schemas import ChatCompletionResponse


_MAX_SSE_EVENT_BYTES = 1024 * 1024


def join_answer_and_references(answer: str, section_text: str) -> str:
    """Join answer text and a prepared reference section with stable spacing."""
    trimmed_answer = answer.rstrip()
    if not trimmed_answer:
        return section_text
    return f"{trimmed_answer}\n\n{section_text}"


def append_references_to_completion(
    response: ChatCompletionResponse,
    reference: ReferenceBuildResult,
) -> ChatCompletionResponse:
    """Return a deep-copied completion whose choices include one reference section."""
    if not response.choices:
        raise gateway_invalid_response_error()

    appended = response.model_copy(deep=True)
    for choice in appended.choices:
        choice.message.content = join_answer_and_references(
            choice.message.content,
            reference.section_text,
        )
    return appended


@dataclass(frozen=True)
class _CompletionStreamMetadata:
    """Fields required to emit one local completion chunk in an existing stream."""

    completion_id: str
    created: int
    model: str
    choice_index: int


class _SSEEventBuffer:
    """Turn arbitrary upstream byte chunks into complete LF- or CRLF-delimited events."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        """Append one network chunk and return every complete SSE event it contains."""
        if not isinstance(chunk, bytes):
            raise gateway_invalid_response_error()
        self._buffer.extend(chunk)
        events: list[bytes] = []
        while True:
            boundary = self._find_boundary()
            if boundary is None:
                break
            end, width = boundary
            events.append(bytes(self._buffer[: end + width]))
            del self._buffer[: end + width]
        if len(self._buffer) > _MAX_SSE_EVENT_BYTES:
            raise gateway_invalid_response_error()
        return events

    def finish(self) -> None:
        """Reject a closed stream that leaves an incomplete non-whitespace event."""
        if self._buffer.strip():
            raise gateway_invalid_response_error()

    def _find_boundary(self) -> tuple[int, int] | None:
        lf_index = self._buffer.find(b"\n\n")
        crlf_index = self._buffer.find(b"\r\n\r\n")
        candidates = [
            (index, width)
            for index, width in ((lf_index, 2), (crlf_index, 4))
            if index >= 0
        ]
        return min(candidates, default=None)


class SSEReferenceAppender:
    """Insert one safe reference delta before an OpenAI-compatible SSE stream ends."""

    def __init__(self, reference: ReferenceBuildResult) -> None:
        self._reference = reference
        self._metadata: _CompletionStreamMetadata | None = None
        self._has_answer_content = False
        self._reference_appended = False
        self._done = False

    async def iterate(self, upstream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Wrap one already-opened upstream stream without buffering its full answer."""
        buffer = _SSEEventBuffer()
        async for chunk in upstream:
            for event in buffer.feed(chunk):
                async for output in self._process_event(event):
                    yield output

        buffer.finish()
        if not self._done:
            raise gateway_invalid_response_error()

    async def _process_event(self, event: bytes) -> AsyncIterator[bytes]:
        if self._done:
            raise gateway_invalid_response_error()

        data = self._event_data(event)
        if data is None:
            yield event
            return

        if data == "[DONE]":
            if self._metadata is None:
                raise gateway_invalid_response_error()
            if not self._reference_appended:
                yield self._reference_event()
                self._reference_appended = True
            yield event
            self._done = True
            return

        payload = self._json_payload(data)
        metadata, finish_reason, has_content = self._metadata_from_payload(payload)
        self._metadata = metadata
        self._has_answer_content = self._has_answer_content or has_content

        if finish_reason is not None:
            if self._reference_appended:
                raise gateway_invalid_response_error()
            yield self._reference_event()
            self._reference_appended = True
        yield event

    @staticmethod
    def _event_data(event: bytes) -> str | None:
        try:
            lines = event.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise gateway_invalid_response_error() from exc

        values: list[str] = []
        for line in lines:
            if line.startswith("data:"):
                value = line[5:]
                values.append(value[1:] if value.startswith(" ") else value)
        if not values:
            return None
        return "\n".join(values)

    @staticmethod
    def _json_payload(data: str) -> dict[str, Any]:
        try:
            payload = loads(data)
        except (JSONDecodeError, ValueError) as exc:
            raise gateway_invalid_response_error() from exc
        if not isinstance(payload, dict):
            raise gateway_invalid_response_error()
        return payload

    @staticmethod
    def _metadata_from_payload(
        payload: dict[str, Any],
    ) -> tuple[_CompletionStreamMetadata, str | None, bool]:
        completion_id = payload.get("id")
        created = payload.get("created")
        model = payload.get("model")
        choices = payload.get("choices")
        if (
            not isinstance(completion_id, str)
            or not completion_id
            or isinstance(created, bool)
            or not isinstance(created, int)
            or not isinstance(model, str)
            or not model
            or not isinstance(choices, list)
            or len(choices) != 1
        ):
            raise gateway_invalid_response_error()

        choice = choices[0]
        if not isinstance(choice, dict):
            raise gateway_invalid_response_error()
        choice_index = choice.get("index")
        delta = choice.get("delta")
        finish_reason = choice.get("finish_reason")
        if (
            isinstance(choice_index, bool)
            or not isinstance(choice_index, int)
            or not isinstance(delta, dict)
            or (
                finish_reason is not None
                and not isinstance(finish_reason, str)
            )
        ):
            raise gateway_invalid_response_error()

        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise gateway_invalid_response_error()
        return (
            _CompletionStreamMetadata(
                completion_id=completion_id,
                created=created,
                model=model,
                choice_index=choice_index,
            ),
            finish_reason,
            bool(content),
        )

    def _reference_event(self) -> bytes:
        if self._metadata is None:
            raise gateway_invalid_response_error()
        content = self._reference.section_text
        if self._has_answer_content:
            content = f"\n\n{content}"
        payload = {
            "id": self._metadata.completion_id,
            "object": "chat.completion.chunk",
            "created": self._metadata.created,
            "model": self._metadata.model,
            "choices": [
                {
                    "index": self._metadata.choice_index,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def append_references_to_sse_stream(
    upstream: AsyncIterator[bytes],
    reference: ReferenceBuildResult,
) -> AsyncIterator[bytes]:
    """Return an SSE stream that emits one reference section before completion ends."""
    return SSEReferenceAppender(reference).iterate(upstream)
