"""FastAPI entrypoint for the server-side model gateway."""

from fastapi import FastAPI

from model_gateway.errors import GatewayError, gateway_error_response
from model_gateway.logging_config import (
    add_request_logging_middleware,
    configure_logging,
)
from model_gateway.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="QwenRag Model Gateway",
        version="0.1.0",
        description="OpenAI-compatible gateway for LLM and embedding services.",
    )
    configure_logging()
    add_request_logging_middleware(app)
    app.add_exception_handler(GatewayError, gateway_error_response)
    app.include_router(router)
    return app


app = create_app()
