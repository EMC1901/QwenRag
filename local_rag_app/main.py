"""FastAPI entrypoint for the Windows-local RAG application."""

import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from local_rag_app.config import get_settings
from local_rag_app.errors import (
    LocalRagError,
    local_rag_error_response,
    unexpected_error_response,
)
from local_rag_app.logging_config import add_request_id_middleware, configure_logging
from local_rag_app import routes


LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load one shared knowledge base before reporting application readiness."""
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.answer_service = routes.get_answer_service(settings)
    knowledge_base = _get_knowledge_base(app.state.answer_service)
    app.state.knowledge_base_status = (
        "loading" if knowledge_base is not None else "not_required"
    )
    if knowledge_base is not None:
        try:
            await asyncio.to_thread(knowledge_base.load)
        except Exception:
            app.state.knowledge_base_status = "failed"
            LOGGER.exception("The local knowledge base failed to load during startup")
        else:
            app.state.knowledge_base_status = "ready"
    try:
        yield
    finally:
        _close_knowledge_base(app.state.answer_service)


def _get_knowledge_base(answer_service: object) -> object | None:
    """Return the knowledge base already owned by the active RAG retriever."""
    retriever = getattr(answer_service, "_retriever", None)
    return getattr(retriever, "_knowledge_base", None)


def _close_knowledge_base(answer_service: object) -> None:
    """Release SQLite/FAISS references before the supervised process exits."""
    knowledge_base = _get_knowledge_base(answer_service)
    close = getattr(knowledge_base, "close", None)
    if callable(close):
        close()


def create_app() -> FastAPI:
    """Create the local application with its stage 3 routes and safeguards."""
    app = FastAPI(
        title="QwenRag Local RAG App",
        version="0.1.0",
        description="Windows-local OpenAI-compatible entrypoint for QwenRag.",
        lifespan=lifespan,
    )
    app.add_exception_handler(LocalRagError, local_rag_error_response)
    app.add_exception_handler(Exception, unexpected_error_response)
    add_request_id_middleware(app)
    app.include_router(routes.router)
    return app


app = create_app()
