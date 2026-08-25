"""Supervisor readiness checks must fail closed without leaking diagnostics."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from local_rag_app.main import create_app
from local_rag_app import main, readiness, routes


class _Client:
    def __init__(self, status_code: int) -> None:
        self._status_code = status_code
        self.requested_url = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def get(self, url: str):
        self.requested_url = url
        return SimpleNamespace(status_code=self._status_code)


def _settings(url: str = "http://127.0.0.1:8010/v1") -> SimpleNamespace:
    return SimpleNamespace(model_gateway_base_url=url)


def test_readiness_requires_knowledge_base_contract_and_gateway(monkeypatch) -> None:
    client = _Client(200)
    monkeypatch.setattr(readiness.httpx, "AsyncClient", lambda **kwargs: client)

    result = asyncio.run(
        readiness.check_readiness(_settings(), knowledge_base_status="ready")
    )

    assert result.ready is True
    assert result.payload() == {
        "status": "ready",
        "checks": {
            "config": "ok",
            "knowledge_base": "ok",
            "model_gateway": "ok",
            "embedding_contract": "ok",
        },
    }
    assert client.requested_url == "http://127.0.0.1:8010/health"


def test_readiness_fails_closed_for_cached_knowledge_base_failure() -> None:
    result = asyncio.run(
        readiness.check_readiness(_settings(), knowledge_base_status="failed")
    )

    assert result.ready is False
    assert result.checks == {
        "config": "ok",
        "knowledge_base": "failed",
        "model_gateway": "failed",
        "embedding_contract": "failed",
    }


def test_readiness_reports_cached_loading_state_without_reloading_assets() -> None:
    result = asyncio.run(
        readiness.check_readiness(_settings(), knowledge_base_status="loading")
    )

    assert result.ready is False
    assert result.checks["knowledge_base"] == "loading"
    assert result.checks["embedding_contract"] == "pending"


def test_readiness_rejects_missing_gateway_configuration() -> None:
    result = asyncio.run(
        readiness.check_readiness(
            _settings(""),
            knowledge_base_status="ready",
        )
    )

    assert result.ready is False
    assert result.checks["config"] == "failed"


def test_ready_endpoint_uses_503_and_safe_check_names() -> None:
    async def not_ready(_settings):
        return readiness.ReadinessResult(
            False,
            {
                "config": "ok",
                "knowledge_base": "failed",
                "model_gateway": "failed",
                "embedding_contract": "failed",
            },
        )

    app = create_app()
    app.state.readiness_checker = not_ready
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "path" not in response.text.lower()
    assert "key" not in response.text.lower()


def test_lifespan_loads_and_reuses_one_rag_knowledge_base(monkeypatch) -> None:
    class _KnowledgeBase:
        def __init__(self) -> None:
            self.load_calls = 0
            self.close_calls = 0

        def load(self) -> None:
            self.load_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    knowledge_base = _KnowledgeBase()
    service = SimpleNamespace(
        _retriever=SimpleNamespace(_knowledge_base=knowledge_base)
    )
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(log_level="INFO"),
    )
    monkeypatch.setattr(main, "configure_logging", lambda _level: None)
    monkeypatch.setattr(routes, "get_answer_service", lambda _settings: service)

    async def exercise() -> None:
        app = create_app()
        async with main.lifespan(app):
            assert app.state.knowledge_base_status == "ready"
            assert knowledge_base.load_calls == 1
        assert knowledge_base.close_calls == 1

    asyncio.run(exercise())
