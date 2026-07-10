"""Tests for the stage 3 OpenAI-compatible model-list endpoint."""

from fastapi.testclient import TestClient

from local_rag_app.main import create_app


def test_models_returns_only_the_local_rag_business_model() -> None:
    """Chatbox should see local-rag, not upstream server model identifiers."""
    with TestClient(create_app()) as client:
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer none"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "local-rag",
                "object": "model",
                "owned_by": "local-rag-app",
            }
        ],
    }


def test_models_rejects_missing_api_key_with_openai_style_error() -> None:
    """The endpoint must not reveal its model list to unauthenticated callers."""
    with TestClient(create_app()) as client:
        response = client.get(
            "/v1/models",
            headers={"X-Request-ID": "missing-key-1"},
        )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "Unauthorized",
            "type": "authentication_error",
            "code": "invalid_api_key",
        }
    }
    assert response.headers["X-Request-ID"] == "missing-key-1"


def test_models_rejects_wrong_api_key() -> None:
    """A syntactically correct but unknown Bearer token is still unauthorized."""
    with TestClient(create_app()) as client:
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key"},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
