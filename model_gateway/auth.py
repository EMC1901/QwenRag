"""Authentication helpers for the model gateway."""

from typing import Annotated

from fastapi import Depends, Header

from model_gateway.config import Settings, get_settings
from model_gateway.errors import authentication_error


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise authentication_error()

    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        raise authentication_error()

    return token.strip()


async def require_api_key(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    """Require a configured gateway API key unless auth is disabled."""
    if settings.gateway_allow_no_auth:
        return

    token = _extract_bearer_token(authorization)
    if token not in settings.gateway_api_keys:
        raise authentication_error()
