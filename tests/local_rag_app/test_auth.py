"""Authentication tests that cover the local API policy in stage 7."""

from fastapi.testclient import TestClient

from local_rag_app.main import create_app


def test_auth_can_be_disabled_only_by_explicit_configuration(monkeypatch) -> None:
    """A development override permits local calls only when deliberately enabled."""
    monkeypatch.setenv("LOCAL_RAG_ALLOW_NO_AUTH", "true")

    with TestClient(create_app()) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "local-rag"


def test_malformed_authorization_header_is_unauthorized() -> None:
    """Only the Bearer scheme may access protected local OpenAI endpoints."""
    with TestClient(create_app()) as client:
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Basic not-a-bearer-token"},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
