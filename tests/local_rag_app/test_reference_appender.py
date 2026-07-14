"""Tests for normal and streaming RAG reference completion appending."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from local_rag_app.errors import LocalRagError
from local_rag_app.reference_appender import (
    append_references_to_completion,
    append_references_to_sse_stream,
    join_answer_and_references,
)
from local_rag_app.reference_models import ReferenceBuildResult, ReferenceFile
from local_rag_app.schemas import (
    AssistantMessage,
    ChatCompletionResponse,
    CompletionChoice,
    Usage,
)


REFERENCE = ReferenceBuildResult(
    section_text=(
        "参考文件：\n"
        "[1] 项目说明书.docx\n"
        "    位置：第一章 / 段落 1-3\n"
        "    对应资料：[资料1]"
    ),
    files=[
        ReferenceFile(
            reference_no=1,
            display_name="项目说明书.docx",
            locations=["第一章 / 段落 1-3"],
            evidence_nos=[1],
        )
    ],
    selected_hit_count=1,
    location_count=1,
)


def _response(*, choices: list[CompletionChoice] | None = None) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-source-test",
        created=1_783_950_000,
        model="local-rag",
        choices=choices
        if choices is not None
        else [CompletionChoice(message=AssistantMessage(content="模型回答\n\n"))],
        usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


def test_join_answer_and_references_has_stable_spacing() -> None:
    """A normal answer has one blank line before the deterministic reference block."""
    assert join_answer_and_references("回答\n\n", REFERENCE.section_text) == (
        "回答\n\n" + REFERENCE.section_text
    )
    assert join_answer_and_references("", REFERENCE.section_text) == REFERENCE.section_text


def test_append_references_copies_response_and_preserves_openai_fields() -> None:
    """Appending sources changes only copied message content, not gateway metadata."""
    response = _response()
    original_dump = response.model_dump()

    appended = append_references_to_completion(response, REFERENCE)

    assert appended is not response
    assert appended.choices[0].message.content == "模型回答\n\n" + REFERENCE.section_text
    assert appended.id == response.id
    assert appended.created == response.created
    assert appended.model == response.model
    assert appended.choices[0].finish_reason == response.choices[0].finish_reason
    assert appended.usage == response.usage
    assert response.model_dump() == original_dump


def test_append_references_updates_every_choice_once() -> None:
    """Even an unexpected multi-choice upstream completion keeps each choice usable."""
    response = _response(
        choices=[
            CompletionChoice(message=AssistantMessage(content="答案一")),
            CompletionChoice(message=AssistantMessage(content="答案二")),
        ]
    )

    appended = append_references_to_completion(response, REFERENCE)

    assert [choice.message.content for choice in appended.choices] == [
        "答案一\n\n" + REFERENCE.section_text,
        "答案二\n\n" + REFERENCE.section_text,
    ]
    assert all(choice.message.content.count("参考文件：") == 1 for choice in appended.choices)


def test_append_references_rejects_an_invalid_completion_without_choices() -> None:
    """A malformed gateway completion must not become a misleading RAG answer."""
    response = _response(choices=[])

    with pytest.raises(LocalRagError) as error:
        append_references_to_completion(response, REFERENCE)

    assert error.value.code == "gateway_invalid_response"


def _sse_event(payload: dict, *, ending: str = "\n\n") -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}{ending}".encode("utf-8")


def _chunk(
    *,
    delta: dict[str, str | None],
    finish_reason: str | None = None,
    completion_id: str = "chatcmpl-stream-test",
    created: int = 1_783_950_000,
    model: str = "local-rag",
    index: int = 0,
    ending: str = "\n\n",
) -> bytes:
    return _sse_event(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": index,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        },
        ending=ending,
    )


async def _upstream(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream])


def _events(body: bytes) -> list[bytes]:
    return [event for event in body.split(b"\n\n") if event]


def _json_event(event: bytes) -> dict:
    return json.loads(event.decode("utf-8").removeprefix("data: "))


@pytest.mark.asyncio
async def test_sse_appender_inserts_references_before_finish_and_done() -> None:
    """A standard stream keeps its original completion identity and end order."""
    upstream = _upstream(
        [
            _chunk(delta={"role": "assistant", "content": None}),
            _chunk(delta={"content": "模型回答"}),
            _chunk(delta={}, finish_reason="stop"),
            b"data: [DONE]\n\n",
        ]
    )

    body = await _collect(append_references_to_sse_stream(upstream, REFERENCE))
    events = _events(body)
    payloads = [_json_event(event) for event in events[:-1]]

    assert events[-1] == b"data: [DONE]"
    assert len(payloads) == 4
    assert payloads[2]["choices"][0]["delta"]["content"] == (
        "\n\n" + REFERENCE.section_text
    )
    assert payloads[2]["id"] == "chatcmpl-stream-test"
    assert payloads[2]["created"] == 1_783_950_000
    assert payloads[2]["model"] == "local-rag"
    assert payloads[2]["choices"][0]["index"] == 0
    assert payloads[3]["choices"][0]["finish_reason"] == "stop"
    reconstructed = "".join(
        payload["choices"][0]["delta"].get("content") or ""
        for payload in payloads
    )
    assert reconstructed == "模型回答\n\n" + REFERENCE.section_text


@pytest.mark.asyncio
@pytest.mark.parametrize("split_mode", ["one_chunk", "one_byte"])
async def test_sse_appender_handles_arbitrary_byte_boundaries(split_mode: str) -> None:
    """SSE parsing is independent of network chunk boundaries, including UTF-8 bytes."""
    source = b"".join(
        [
            _chunk(delta={"role": "assistant", "content": None}),
            _chunk(delta={"content": "中文回答"}),
            _chunk(delta={}, finish_reason="stop"),
            b"data: [DONE]\n\n",
        ]
    )
    chunks = [source] if split_mode == "one_chunk" else [bytes([value]) for value in source]

    body = await _collect(append_references_to_sse_stream(_upstream(chunks), REFERENCE))

    assert body.count("参考文件：".encode("utf-8")) == 1
    assert body.index("参考文件：".encode("utf-8")) < body.index(b'"finish_reason": "stop"')
    assert body.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_sse_appender_handles_crlf_comments_and_done_without_finish() -> None:
    """Comments pass through and a reference still precedes DONE when finish is absent."""
    role = _chunk(delta={"role": "assistant", "content": None}, ending="\r\n\r\n")
    content = _chunk(delta={"content": "回答"}, ending="\r\n\r\n")
    upstream = _upstream([b": keep-alive\r\n\r\n", role, content, b"data: [DONE]\r\n\r\n"])

    body = await _collect(append_references_to_sse_stream(upstream, REFERENCE))

    assert body.startswith(b": keep-alive\r\n\r\n")
    assert body.count("参考文件：".encode("utf-8")) == 1
    assert body.index("参考文件：".encode("utf-8")) < body.index(b"data: [DONE]")
    assert body.endswith(b"data: [DONE]\r\n\r\n")


@pytest.mark.asyncio
async def test_sse_appender_uses_no_leading_blank_lines_without_answer_content() -> None:
    """A role-only upstream stream emits a clean reference block before its finish chunk."""
    upstream = _upstream(
        [
            _chunk(delta={"role": "assistant", "content": None}),
            _chunk(delta={}, finish_reason="stop"),
            b"data: [DONE]\n\n",
        ]
    )

    body = await _collect(append_references_to_sse_stream(upstream, REFERENCE))
    payloads = [_json_event(event) for event in _events(body)[:-1]]

    assert payloads[1]["choices"][0]["delta"]["content"] == REFERENCE.section_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        [b"data: not-json\n\n"],
        [b"data: [DONE]\n\n"],
        [_chunk(delta={"content": "partial"})],
        [b"x" * (1024 * 1024 + 1)],
    ],
)
async def test_sse_appender_rejects_invalid_or_incomplete_upstream_streams(
    chunks: list[bytes],
) -> None:
    """Malformed JSON, missing metadata, missing DONE, and oversized events fail closed."""
    with pytest.raises(LocalRagError) as error:
        await _collect(append_references_to_sse_stream(_upstream(chunks), REFERENCE))

    assert error.value.code == "gateway_invalid_response"


@pytest.mark.asyncio
async def test_sse_appender_rejects_data_after_done_and_duplicate_finish() -> None:
    """The wrapper must not quietly accept a second terminal sequence or extra data."""
    after_done = _upstream(
        [
            _chunk(delta={"content": "回答"}),
            _chunk(delta={}, finish_reason="stop"),
            b"data: [DONE]\n\n",
            _chunk(delta={"content": "unexpected"}),
        ]
    )
    duplicate_finish = _upstream(
        [
            _chunk(delta={"content": "回答"}),
            _chunk(delta={}, finish_reason="stop"),
            _chunk(delta={}, finish_reason="stop"),
            b"data: [DONE]\n\n",
        ]
    )

    for upstream in (after_done, duplicate_finish):
        with pytest.raises(LocalRagError) as error:
            await _collect(append_references_to_sse_stream(upstream, REFERENCE))
        assert error.value.code == "gateway_invalid_response"


@pytest.mark.asyncio
async def test_sse_appender_preserves_upstream_iterator_errors() -> None:
    """Transport-layer errors remain visible to the caller instead of being rewritten."""

    async def failing_upstream() -> AsyncIterator[bytes]:
        yield _chunk(delta={"content": "partial"})
        raise RuntimeError("transport interrupted")

    with pytest.raises(RuntimeError, match="transport interrupted"):
        await _collect(append_references_to_sse_stream(failing_upstream(), REFERENCE))
