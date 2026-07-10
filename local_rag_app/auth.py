"""Local API-key authentication for OpenAI-compatible endpoints."""

import secrets
from typing import Annotated

from fastapi import Depends, Header

from local_rag_app.config import Settings, get_settings
from local_rag_app.errors import authentication_error


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Return a Bearer token only when the Authorization header is well formed."""
    if authorization is None:
        return None
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        return None
    return token.strip()


def _matches_allowed_key(candidate: str, allowed_keys: list[str]) -> bool:
    """Compare keys without short-circuiting on character-by-character matches."""
    candidate_bytes = candidate.encode("utf-8")
    return any(
        secrets.compare_digest(candidate_bytes, allowed_key.encode("utf-8"))
        for allowed_key in allowed_keys
    )


async def require_local_api_key(
    authorization: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    """Require a configured local key unless local authentication is disabled."""
    if settings.local_rag_allow_no_auth:
        return

    token = _extract_bearer_token(authorization)
    if token is None or not _matches_allowed_key(token, settings.local_rag_api_keys):
        raise authentication_error()
