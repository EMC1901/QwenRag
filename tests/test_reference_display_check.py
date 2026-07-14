"""Offline tests for the HTTP reference-display acceptance checker."""

from __future__ import annotations

import json

import pytest

from scripts.check_reference_display import (
    EXIT_ARGUMENT_ERROR,
    EXIT_RUNTIME_FAILURE,
    EXIT_SUCCESS,
    parse_args,
    run_check,
)


PRIVATE_QUERY = "PRIVATE-QUERY-SECRET"
PRIVATE_KEY = "PRIVATE-KEY-SECRET"
PRIVATE_FILE = "PRIVATE-FILENAME-SECRET.docx"
PRIVATE_LOCATION = "PRIVATE-LOCATION-SECRET"
PRIVATE_ANSWER = "PRIVATE-ANSWER-SECRET"
REFERENCE = (
    "参考文件：\n"
    f"[1] {PRIVATE_FILE}\n"
    f"    位置：{PRIVATE_LOCATION}\n"
    "    对应资料：[资料1]"
)


class FakeResponse:
    """Minimal requests-compatible response with JSON and SSE test payloads."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_payload: object | None = None,
        lines: list[bytes | str] | None = None,
        content_type: str = "application/json",
    ) -> None:
        self.status_code = status_code
        self._json_payload = json_payload
        self._lines = lines or []
        self.headers = {"Content-Type": content_type}

    def json(self) -> object:
        if isinstance(self._json_payload, Exception):
            raise self._json_payload
        return self._json_payload

    def iter_lines(self):
        yield from self._lines


class FakePost:
    """Return prepared responses and retain safe-to-assert request parameters."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def _completion(answer: str) -> FakeResponse:
    return FakeResponse(
        json_payload={"choices": [{"message": {"content": answer}}]},
    )


def _sse_event(delta: dict, finish_reason: str | None = None) -> bytes:
    return (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "created": 1,
                "model": "local-rag",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }
                ],
            },
            ensure_ascii=False,
        )
    ).encode("utf-8")


def _stream(*, reference_before_finish: bool = True, done: bool = True) -> FakeResponse:
    lines: list[bytes] = [
        _sse_event({"role": "assistant"}),
        _sse_event({"content": PRIVATE_ANSWER}),
    ]
    if reference_before_finish:
        lines.append(_sse_event({"content": "\n\n" + REFERENCE}))
    lines.append(_sse_event({}, "stop"))
    if not reference_before_finish:
        lines.append(_sse_event({"content": "\n\n" + REFERENCE}))
    if done:
        lines.append(b"data: [DONE]")
    return FakeResponse(lines=lines, content_type="text/event-stream; charset=utf-8")


def _run(*, post: FakePost, **overrides):
    lines: list[str] = []
    values = {
        "base_url": "http://127.0.0.1:18080/",
        "query": PRIVATE_QUERY,
        "api_key": PRIVATE_KEY,
        "non_stream_only": False,
        "stream_only": False,
        "show_answer": False,
        "post": post,
        "output": lines.append,
    }
    values.update(overrides)
    return run_check(**values), lines


def test_both_modes_succeed_without_printing_private_content_by_default() -> None:
    post = FakePost([_completion(PRIVATE_ANSWER + "\n\n" + REFERENCE), _stream()])

    code, lines = _run(post=post)

    output = "\n".join(lines)
    assert code == EXIT_SUCCESS
    assert len(post.calls) == 2
    assert post.calls[0]["url"] == "http://127.0.0.1:18080/v1/chat/completions"
    assert post.calls[0]["json"]["stream"] is False
    assert post.calls[1]["json"]["stream"] is True
    assert post.calls[0]["timeout"] == post.calls[1]["timeout"] == 300
    assert post.calls[0]["headers"]["Authorization"] == f"Bearer {PRIVATE_KEY}"
    assert "[PASS] non-stream response returned one reference section" in output
    assert "[PASS] stream reference delta appeared before finish_reason" in output
    assert "Reference display check passed." in output
    for secret in (PRIVATE_QUERY, PRIVATE_KEY, PRIVATE_FILE, PRIVATE_LOCATION, PRIVATE_ANSWER):
        assert secret not in output


def test_empty_query_and_missing_api_key_return_argument_error_without_http_calls() -> None:
    post = FakePost([])

    empty_query_code, empty_query_lines = _run(post=post, query="   ")
    missing_key_code, missing_key_lines = _run(post=post, api_key=" ")

    assert empty_query_code == missing_key_code == EXIT_ARGUMENT_ERROR
    assert post.calls == []
    assert "--query" in "\n".join(empty_query_lines)
    assert "LOCAL_RAG_API_KEY" in "\n".join(missing_key_lines)


@pytest.mark.parametrize("status_code", [401, 502, 503, 504])
def test_http_errors_fail_without_printing_response_body(status_code: int) -> None:
    post = FakePost([FakeResponse(status_code=status_code)])

    code, lines = _run(post=post, non_stream_only=True)

    assert code == EXIT_RUNTIME_FAILURE
    assert f"HTTP {status_code}" in "\n".join(lines)


@pytest.mark.parametrize(
    "answer",
    [
        PRIVATE_ANSWER,
        PRIVATE_ANSWER + "\n\n" + REFERENCE + "\n\n" + REFERENCE,
    ],
)
def test_non_stream_requires_exactly_one_reference_section(answer: str) -> None:
    post = FakePost([_completion(answer)])

    code, lines = _run(post=post, non_stream_only=True)

    assert code == EXIT_RUNTIME_FAILURE
    assert "one valid reference section" in "\n".join(lines)


def test_stream_rejects_reference_after_finish_missing_done_and_invalid_json() -> None:
    cases = [
        _stream(reference_before_finish=False),
        _stream(done=False),
        FakeResponse(
            lines=[b"data: not-json", b"data: [DONE]"],
            content_type="text/event-stream",
        ),
    ]

    for response in cases:
        code, lines = _run(post=FakePost([response]), stream_only=True)

        assert code == EXIT_RUNTIME_FAILURE
        assert "[FAIL]" in "\n".join(lines)


def test_show_answer_prints_warning_before_the_complete_private_content() -> None:
    post = FakePost([_completion(PRIVATE_ANSWER + "\n\n" + REFERENCE)])

    code, lines = _run(post=post, non_stream_only=True, show_answer=True)

    assert code == EXIT_SUCCESS
    assert lines[0].startswith("[WARNING]")
    assert any(PRIVATE_ANSWER in line for line in lines)
    assert any(PRIVATE_FILE in line for line in lines)


def test_argument_parser_does_not_accept_an_api_key_option() -> None:
    parsed = parse_args(
        [
            "--base-url",
            "http://127.0.0.1:18080",
            "--query",
            "question",
            "--stream-only",
        ]
    )

    assert parsed.stream_only is True
    assert "api_key" not in vars(parsed)
