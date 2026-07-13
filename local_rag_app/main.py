"""FastAPI entrypoint for the Windows-local RAG application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from local_rag_app.config import get_settings
from local_rag_app.errors import (
    LocalRagError,
    local_rag_error_response,
    unexpected_error_response,
)
from local_rag_app.logging_config import add_request_id_middleware, configure_logging
from local_rag_app import routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create one answer service before this process starts accepting requests."""
    settings = get_settings()
    configure_logging(settings.log_level)
    app.state.answer_service = routes.get_answer_service(settings)
    yield


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
