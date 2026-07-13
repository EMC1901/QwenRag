"""Run a privacy-safe real-environment acceptance check for local retrieval."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_rag_app.config import Settings, get_settings
from local_rag_app.errors import LocalRagError
from local_rag_app.knowledge_base import KnowledgeBase, KnowledgeBaseLoadError
from local_rag_app.retrieval import LocalRetriever
from local_rag_app.retrieval_models import RetrievalHit, RetrievalResult


EXIT_SUCCESS = 0
EXIT_RUNTIME_FAILURE = 1
EXIT_ARGUMENT_ERROR = 2
_PREVIEW_LIMIT = 100


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse only the query and the explicit private-data preview switch."""
    parser = argparse.ArgumentParser(
        description="Validate local RAG retrieval without printing source text by default.",
    )
    parser.add_argument("--query", required=True, help="One non-empty retrieval query.")
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Print up to 100 characters of each matched chunk; may expose private data.",
    )
    return parser.parse_args(argv)


def run_check(
    query: str,
    *,
    show_preview: bool,
    settings: Settings,
    knowledge_base: KnowledgeBase | Any,
    retriever: LocalRetriever | Any,
    output: Callable[[str], None] = print,
) -> int:
    """Validate assets and retrieve one query, returning a documented process code."""
    normalized_query = query.strip()
    if not normalized_query:
        output("[FAIL] --query must contain non-whitespace text.")
        return EXIT_ARGUMENT_ERROR
    if not _has_gateway_embedding_configuration(settings):
        output("[FAIL] Model gateway embedding configuration is incomplete.")
        return EXIT_RUNTIME_FAILURE

    try:
        knowledge_base.load()
    except (KnowledgeBaseLoadError, OSError, ValueError, RuntimeError):
        output("[FAIL] knowledge-base assets are unavailable or inconsistent.")
        return EXIT_RUNTIME_FAILURE
    output("[PASS] knowledge-base assets are consistent")

    try:
        result = asyncio.run(retriever.retrieve(normalized_query))
    except LocalRagError:
        output("[FAIL] embedding or local retrieval failed.")
        return EXIT_RUNTIME_FAILURE

    output(f"[PASS] embedding model returned {result.embedding_dim} dimensions")
    output(
        "[PASS] vector search returned "
        f"{result.vector_candidate_count} candidates"
    )
    output(
        "[PASS] retrieval returned "
        f"{len(result.hits)} hits in {result.retrieval_mode} mode"
    )
    if not result.hits:
        output("[FAIL] retrieval returned no hits.")
        return EXIT_RUNTIME_FAILURE

    if show_preview:
        output(
            "[WARNING] Source previews may contain private data; do not paste this output "
            "into tickets, chat, or logs."
        )
    for hit in result.hits:
        output(_format_hit_summary(hit))
        if show_preview:
            output(f"   preview={_preview(hit.chunk_text)}")

    output("Local retrieval check passed.")
    return EXIT_SUCCESS


def _has_gateway_embedding_configuration(settings: Settings) -> bool:
    """Ensure this standalone checker has the upstream values retrieval actually needs."""
    return bool(
        settings.model_gateway_base_url
        and settings.model_gateway_api_key
        and settings.upstream_embedding_model
    )


def _format_hit_summary(hit: RetrievalHit) -> str:
    """Render source metadata without chunk text or a full source path."""
    title = hit.title or hit.doc_title or Path(hit.relative_path).name
    location = hit.section_path or hit.article_range or hit.article_no or ""
    return (
        f"{hit.rank}. title={_single_line(title)} "
        f"section={_single_line(location)} "
        f"score={hit.final_score:.4f} source={hit.matched_by}"
    )


def _preview(text: str) -> str:
    """Return a bounded one-line preview only after the caller explicitly opts in."""
    return _single_line(text)[:_PREVIEW_LIMIT]


def _single_line(value: str) -> str:
    """Keep a metadata field bounded to one terminal line without changing source data."""
    return " ".join(value.split())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone checker and convert expected setup failures to exit codes."""
    args = parse_args(argv)
    try:
        settings = get_settings()
    except (OSError, ValueError):
        print("[FAIL] Local retrieval configuration is invalid or unavailable.")
        return EXIT_RUNTIME_FAILURE

    knowledge_base = KnowledgeBase(settings)
    retriever = LocalRetriever(settings, knowledge_base=knowledge_base)
    return run_check(
        args.query,
        show_preview=args.show_preview,
        settings=settings,
        knowledge_base=knowledge_base,
        retriever=retriever,
    )


if __name__ == "__main__":
    sys.exit(main())
