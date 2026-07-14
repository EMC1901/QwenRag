"""Privacy-safe HTTP acceptance check for visible RAG reference files."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urlparse

import requests


EXIT_SUCCESS = 0
EXIT_RUNTIME_FAILURE = 1
EXIT_ARGUMENT_ERROR = 2
_REFERENCE_HEADING = "参考文件："
_EVIDENCE_LABEL = "对应资料："
_FILE_ITEM_PATTERN = re.compile(r"^\[\d+\]\s+\S", re.MULTILINE)
# A complete RAG request includes routing, embedding, local retrieval, and LLM
# generation.  Its wall-clock time can exceed the gateway read timeout, so the
# local acceptance client must leave room for the entire pipeline.
_REQUEST_TIMEOUT_SECONDS = 300


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only non-sensitive checker inputs and explicit display switches."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate normal and streaming local-RAG reference display without "
            "printing private answers or source metadata by default."
        ),
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Local application base URL, for example http://127.0.0.1:18080.",
    )
    parser.add_argument("--query", required=True, help="A non-empty RAG acceptance query.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--non-stream-only",
        action="store_true",
        help="Check only the normal JSON completion path.",
    )
    modes.add_argument(
        "--stream-only",
        action="store_true",
        help="Check only the SSE completion path.",
    )
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Print complete answers and source text; this may expose private data.",
    )
    return parser.parse_args(argv)


def run_check(
    *,
    base_url: str,
    query: str,
    api_key: str | None,
    non_stream_only: bool,
    stream_only: bool,
    show_answer: bool,
    post: Callable[..., Any] = requests.post,
    output: Callable[[str], None] = print,
) -> int:
    """Run HTTP acceptance checks and return documented, privacy-safe exit codes."""
    normalized_base_url = _normalize_base_url(base_url)
    if normalized_base_url is None:
        output("[FAIL] --base-url must be an absolute http(s) URL.")
        return EXIT_ARGUMENT_ERROR
    normalized_query = query.strip()
    if not normalized_query:
        output("[FAIL] --query must contain non-whitespace text.")
        return EXIT_ARGUMENT_ERROR
    if not api_key or not api_key.strip():
        output("[FAIL] LOCAL_RAG_API_KEY environment variable is required.")
        return EXIT_ARGUMENT_ERROR
    if non_stream_only and stream_only:
        output("[FAIL] --non-stream-only and --stream-only cannot be used together.")
        return EXIT_ARGUMENT_ERROR

    if show_answer:
        output(
            "[WARNING] Full answers and reference metadata may contain private data; "
            "do not paste this output into tickets, chat, or logs."
        )

    url = f"{normalized_base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    if not stream_only:
        answer = _check_non_stream(
            url=url,
            headers=headers,
            query=normalized_query,
            post=post,
            output=output,
        )
        if answer is None:
            return EXIT_RUNTIME_FAILURE
        if show_answer:
            output(f"non_stream_answer={answer}")

    if not non_stream_only:
        answer = _check_stream(
            url=url,
            headers=headers,
            query=normalized_query,
            post=post,
            output=output,
        )
        if answer is None:
            return EXIT_RUNTIME_FAILURE
        if show_answer:
            output(f"stream_answer={answer}")

    output("Reference display check passed.")
    return EXIT_SUCCESS


def _check_non_stream(
    *,
    url: str,
    headers: dict[str, str],
    query: str,
    post: Callable[..., Any],
    output: Callable[[str], None],
) -> str | None:
    response = _post(
        post,
        url=url,
        headers=headers,
        query=query,
        stream=False,
        output=output,
        mode="non-stream",
    )
    if response is None:
        return None
    if getattr(response, "status_code", None) != 200:
        output(f"[FAIL] non-stream request returned HTTP {_safe_status(response)}.")
        return None
    try:
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        output("[FAIL] non-stream response was not a valid chat completion.")
        return None
    if not isinstance(answer, str) or not answer.strip():
        output("[FAIL] non-stream response contained an empty answer.")
        return None
    file_count = _validate_reference_text(answer)
    if file_count is None:
        output("[FAIL] non-stream response did not contain one valid reference section.")
        return None

    output("[PASS] non-stream response returned one reference section")
    output(f"[PASS] non-stream response listed {file_count} reference files")
    return answer


def _check_stream(
    *,
    url: str,
    headers: dict[str, str],
    query: str,
    post: Callable[..., Any],
    output: Callable[[str], None],
) -> str | None:
    response = _post(
        post,
        url=url,
        headers=headers,
        query=query,
        stream=True,
        output=output,
        mode="stream",
    )
    if response is None:
        return None
    if getattr(response, "status_code", None) != 200:
        output(f"[FAIL] stream request returned HTTP {_safe_status(response)}.")
        return None
    content_type = str(getattr(response, "headers", {}).get("Content-Type", "")).lower()
    if not content_type.startswith("text/event-stream"):
        output("[FAIL] stream response did not use text/event-stream.")
        return None

    try:
        answer_before_finish, answer, saw_done = _decode_stream(response.iter_lines())
    except (AttributeError, TypeError, UnicodeDecodeError, ValueError):
        output("[FAIL] stream response was not valid OpenAI-compatible SSE.")
        return None
    if not saw_done:
        output("[FAIL] stream response ended without a final [DONE] event.")
        return None
    if _validate_reference_text(answer_before_finish) is None:
        output("[FAIL] stream reference section did not appear before finish_reason.")
        return None
    file_count = _validate_reference_text(answer)
    if file_count is None:
        output("[FAIL] reconstructed stream did not contain one valid reference section.")
        return None

    output("[PASS] stream events were valid and ended with [DONE]")
    output("[PASS] stream reference delta appeared before finish_reason")
    output("[PASS] reconstructed stream contained one reference section")
    output(f"[PASS] stream response listed {file_count} reference files")
    return answer


def _post(
    post: Callable[..., Any],
    *,
    url: str,
    headers: dict[str, str],
    query: str,
    stream: bool,
    output: Callable[[str], None],
    mode: str,
) -> Any | None:
    try:
        return post(
            url,
            headers=headers,
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": query}],
                "stream": stream,
            },
            stream=stream,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except (OSError, requests.RequestException):
        output(
            f"[FAIL] {mode} request could not reach the local application "
            "or did not finish before the acceptance timeout."
        )
        return None


def _decode_stream(lines: Iterable[bytes | str]) -> tuple[str, str, bool]:
    """Return content before finish, complete content, and terminal-DONE status."""
    answer_parts: list[str] = []
    answer_before_finish: str | None = None
    saw_finish = False
    saw_done = False

    for raw_line in lines:
        line = _decode_sse_line(raw_line)
        if saw_done and line.strip():
            raise ValueError("data after DONE")
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].lstrip()
        if payload_text == "[DONE]":
            if saw_done:
                raise ValueError("duplicate DONE")
            saw_done = True
            continue
        if saw_done:
            raise ValueError("data after DONE")

        try:
            payload = json.loads(payload_text)
            choices = payload["choices"]
            choice = choices[0]
            delta = choice["delta"]
            finish_reason = choice["finish_reason"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid SSE JSON") from exc
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(delta, dict):
            raise ValueError("invalid SSE choice")
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("invalid SSE content")
        if content:
            if saw_finish:
                raise ValueError("content after finish")
            answer_parts.append(content)
        if finish_reason is not None:
            if not isinstance(finish_reason, str) or saw_finish:
                raise ValueError("invalid SSE finish")
            saw_finish = True
            answer_before_finish = "".join(answer_parts)

    if not saw_done or not saw_finish or answer_before_finish is None:
        return "", "".join(answer_parts), False
    return answer_before_finish, "".join(answer_parts), True


def _decode_sse_line(raw_line: bytes | str) -> str:
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8")
    if isinstance(raw_line, str):
        return raw_line
    raise ValueError("invalid SSE line type")


def _validate_reference_text(answer: str) -> int | None:
    """Validate the minimum user-visible source contract without printing it."""
    if answer.count(_REFERENCE_HEADING) != 1:
        return None
    file_count = len(_FILE_ITEM_PATTERN.findall(answer))
    if file_count < 1 or _EVIDENCE_LABEL not in answer:
        return None
    return file_count


def _normalize_base_url(value: str) -> str | None:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return normalized


def _safe_status(response: Any) -> str:
    status = getattr(response, "status_code", "unknown")
    return str(status) if isinstance(status, int) else "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    """Read the local API key from the environment and execute the checker."""
    args = parse_args(argv)
    return run_check(
        base_url=args.base_url,
        query=args.query,
        api_key=os.getenv("LOCAL_RAG_API_KEY"),
        non_stream_only=args.non_stream_only,
        stream_only=args.stream_only,
        show_answer=args.show_answer,
    )


if __name__ == "__main__":
    sys.exit(main())
