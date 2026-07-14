"""Run a privacy-safe real-environment acceptance check for RAG generation."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_rag_app.config import Settings, get_settings
from local_rag_app.context_builder import ContextBuildError, ContextBuilder
from local_rag_app.context_models import ContextBuildResult
from local_rag_app.errors import LocalRagError
from local_rag_app.knowledge_base import KnowledgeBase, KnowledgeBaseLoadError
from local_rag_app.rag_generation import RagGenerationService
from local_rag_app.retrieval import LocalRetriever
from local_rag_app.retrieval_models import RetrievalHit, RetrievalResult
from local_rag_app.schemas import ChatCompletionRequest, ChatCompletionResponse


EXIT_SUCCESS = 0
EXIT_RUNTIME_FAILURE = 1
EXIT_ARGUMENT_ERROR = 2
_PREVIEW_LIMIT = 100


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit switches for output that may contain private data."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate local retrieval, context assembly, and RAG generation "
            "without printing private content by default."
        ),
    )
    parser.add_argument("--query", required=True, help="One non-empty RAG question.")
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Print the model answer; it may contain private data.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print a bounded context preview; it may contain private data.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Validate the streaming generation path and its final [DONE] event.",
    )
    return parser.parse_args(argv)


def run_check(
    query: str,
    *,
    show_answer: bool,
    show_context: bool,
    stream: bool,
    settings: Settings,
    knowledge_base: KnowledgeBase | Any,
    retriever: LocalRetriever | Any,
    context_builder: ContextBuilder | Any,
    generation_service: RagGenerationService | Any,
    output: Callable[[str], None] = print,
) -> int:
    """Run one end-to-end RAG check with documented, privacy-safe exit codes."""
    normalized_query = query.strip()
    if not normalized_query:
        output("[FAIL] --query must contain non-whitespace text.")
        return EXIT_ARGUMENT_ERROR
    if not _has_rag_generation_configuration(settings):
        output("[FAIL] RAG answer-generation configuration is incomplete or disabled.")
        return EXIT_RUNTIME_FAILURE

    try:
        knowledge_base.load()
    except (KnowledgeBaseLoadError, OSError, ValueError, RuntimeError):
        output("[FAIL] knowledge-base assets are unavailable or inconsistent.")
        return EXIT_RUNTIME_FAILURE
    output("[PASS] knowledge-base assets are consistent")

    try:
        retrieval_result = asyncio.run(retriever.retrieve(normalized_query))
    except (LocalRagError, OSError, ValueError, RuntimeError):
        output("[FAIL] embedding or local retrieval failed.")
        return EXIT_RUNTIME_FAILURE
    output(
        "[PASS] retrieval returned "
        f"{len(retrieval_result.hits)} hits in {retrieval_result.retrieval_mode} mode"
    )
    if not retrieval_result.hits:
        output("[FAIL] retrieval returned no hits for the acceptance query.")
        return EXIT_RUNTIME_FAILURE

    request = ChatCompletionRequest.model_validate(
        {
            "model": settings.local_rag_model,
            "messages": [{"role": "user", "content": normalized_query}],
            "stream": stream,
        }
    )
    try:
        context = context_builder.build(request, retrieval_result)
    except (ContextBuildError, LocalRagError, ValueError, RuntimeError):
        output("[FAIL] retrieved context could not be prepared safely.")
        return EXIT_RUNTIME_FAILURE
    output(
        "[PASS] context selected "
        f"{len(context.selected_hits)} hits and dropped {context.dropped_hit_count}"
    )
    output(
        "[PASS] estimated input is "
        f"{context.estimated_input_tokens} / {settings.rag_max_input_tokens} tokens"
    )

    if show_context:
        output(
            "[WARNING] Context previews may contain private data; do not paste this "
            "output into tickets, chat, or logs."
        )
        for selected_hit in context.selected_hits:
            output(_format_context_preview(selected_hit.hit, selected_hit.text_for_prompt))

    try:
        answer = (
            asyncio.run(_stream_answer(generation_service, request, retrieval_result))
            if stream
            else _complete_answer(generation_service, request, retrieval_result)
        )
    except (LocalRagError, OSError, ValueError, RuntimeError):
        output("[FAIL] RAG answer generation failed.")
        return EXIT_RUNTIME_FAILURE
    if not answer.strip():
        output("[FAIL] RAG answer generation returned empty text.")
        return EXIT_RUNTIME_FAILURE
    output("[PASS] generation returned non-empty text")

    if show_answer:
        output(
            "[WARNING] The model answer may contain private data; do not paste this "
            "output into tickets, chat, or logs."
        )
        output(f"answer={answer}")

    output("RAG generation check passed.")
    return EXIT_SUCCESS


def _has_rag_generation_configuration(settings: Settings) -> bool:
    """Ensure the standalone check exercises the enabled production RAG path."""
    return bool(
        settings.local_rag_answer_mode == "gateway"
        and settings.enable_rag_router
        and settings.enable_local_retrieval
        and settings.enable_rag_answer_generation
        and settings.model_gateway_base_url
        and settings.model_gateway_api_key
        and settings.upstream_llm_model
        and settings.upstream_embedding_model
    )


def _complete_answer(
    generation_service: RagGenerationService | Any,
    request: ChatCompletionRequest,
    retrieval_result: RetrievalResult | Any,
) -> str:
    """Generate one normal answer and reject malformed empty completion choices."""
    response: ChatCompletionResponse = asyncio.run(
        generation_service.complete(request, retrieval_result)
    )
    if not response.choices:
        return ""
    return response.choices[0].message.content


async def _stream_answer(
    generation_service: RagGenerationService | Any,
    request: ChatCompletionRequest,
    retrieval_result: RetrievalResult | Any,
) -> str:
    """Consume standard OpenAI SSE and return text only after a [DONE] terminator."""
    event_stream: AsyncIterator[bytes] = await generation_service.stream(
        request,
        retrieval_result,
    )
    raw_events = b"".join([event async for event in event_stream])
    return _decode_sse_answer(raw_events)


def _decode_sse_answer(raw_events: bytes) -> str:
    """Parse the minimal SSE subset required for an OpenAI-compatible response."""
    answer_parts: list[str] = []
    saw_done = False
    for event in raw_events.decode("utf-8").split("\n\n"):
        data_lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            parsed = json.loads(payload)
            content = parsed["choices"][0]["delta"].get("content")
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid OpenAI-compatible SSE payload") from exc
        if isinstance(content, str):
            answer_parts.append(content)
    if not saw_done:
        raise ValueError("stream ended without [DONE]")
    return "".join(answer_parts)


def _format_context_preview(hit: RetrievalHit, text_for_prompt: str) -> str:
    """Render only bounded source metadata and text after explicit user opt-in."""
    title = hit.title or hit.doc_title or Path(hit.relative_path).name
    location = hit.section_path or hit.article_range or hit.article_no or ""
    return (
        f"evidence={hit.rank} title={_single_line(title)} "
        f"section={_single_line(location)} preview={_preview(text_for_prompt)}"
    )


def _preview(text: str) -> str:
    """Return a bounded, one-line private-content preview after opt-in only."""
    return _single_line(text)[:_PREVIEW_LIMIT]


def _single_line(value: str) -> str:
    """Keep opt-in terminal output to one line without exposing a source path."""
    return " ".join(value.split())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone checker and convert expected setup failures to exit codes."""
    args = parse_args(argv)
    try:
        settings = get_settings()
    except (OSError, ValueError):
        print("[FAIL] RAG answer-generation configuration is invalid or unavailable.")
        return EXIT_RUNTIME_FAILURE

    knowledge_base = KnowledgeBase(settings)
    retriever = LocalRetriever(settings, knowledge_base=knowledge_base)
    context_builder = ContextBuilder(settings)
    generation_service = RagGenerationService(
        settings,
        context_builder=context_builder,
    )
    return run_check(
        args.query,
        show_answer=args.show_answer,
        show_context=args.show_context,
        stream=args.stream,
        settings=settings,
        knowledge_base=knowledge_base,
        retriever=retriever,
        context_builder=context_builder,
        generation_service=generation_service,
    )


if __name__ == "__main__":
    sys.exit(main())
