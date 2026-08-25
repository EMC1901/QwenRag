"""Safe readiness checks for the supervisor-facing local RAG endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

import httpx

from local_rag_app.config import Settings

KnowledgeBaseStatus = Literal["loading", "ready", "failed", "not_required"]


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    checks: dict[str, str]

    def payload(self) -> dict[str, object]:
        return {"status": "ready" if self.ready else "not_ready", "checks": self.checks}


async def check_readiness(
    settings: Settings,
    *,
    knowledge_base_status: KnowledgeBaseStatus,
) -> ReadinessResult:
    """Check cached component status; never reload assets or expose details."""
    checks = {"config": "ok", "knowledge_base": "failed", "model_gateway": "failed", "embedding_contract": "failed"}
    parsed = urlsplit(settings.model_gateway_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        checks["config"] = "failed"
        return ReadinessResult(False, checks)

    if knowledge_base_status == "loading":
        checks["knowledge_base"] = "loading"
        checks["embedding_contract"] = "pending"
        return ReadinessResult(False, checks)
    if knowledge_base_status == "failed":
        return ReadinessResult(False, checks)
    checks["knowledge_base"] = "ok"
    checks["embedding_contract"] = "ok"

    try:
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(health_url)
        if response.status_code != 200:
            return ReadinessResult(False, checks)
        checks["model_gateway"] = "ok"
    except httpx.HTTPError:
        return ReadinessResult(False, checks)
    return ReadinessResult(True, checks)
