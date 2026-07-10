"""Tests for the stage 3 local process-health endpoint."""

from fastapi.testclient import TestClient

from local_rag_app.main import create_app


def test_health_returns_process_status_and_request_id() -> None:
    """Health must not require credentials and must support request tracing."""
    with TestClient(create_app()) as client:
        response = client.get("/health", headers={"X-Request-ID": "health-check-1"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "local-rag-app"}
    assert response.headers["X-Request-ID"] == "health-check-1"


def test_health_generates_request_id_when_client_does_not_supply_one() -> None:
    """Every response needs a trace ID even when Chatbox does not send one."""
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 32
