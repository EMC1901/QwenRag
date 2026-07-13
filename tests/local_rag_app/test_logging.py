"""Privacy and diagnostic-field tests for stage 7 request logging."""

import logging

from fastapi.testclient import TestClient

from local_rag_app.logging_config import LOGGER_NAME
from local_rag_app.main import create_app


def test_request_log_has_diagnostic_fields_without_prompt_or_key(monkeypatch, caplog) -> None:
    """Logging must make failures traceable without storing private conversation data."""
    private_prompt = "客户私有问题：合同编号-SECRET-123"
    local_key = "local-api-key-SECRET"
    monkeypatch.setenv("LOCAL_RAG_API_KEYS", local_key)
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {local_key}",
                "X-Request-ID": "privacy-log-1",
            },
            json={
                "model": "local-rag",
                "messages": [{"role": "user", "content": private_prompt}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "request_id=privacy-log-1" in log_text
    assert "method=POST" in log_text
    assert "path=/v1/chat/completions" in log_text
    assert "status_code=200" in log_text
    assert "answer_mode=stub" in log_text
    assert "rag_route=router_disabled" in log_text
    assert "rag_decision_source= " in log_text
    assert "retrieval_mode= " in log_text
    assert "retrieval_hit_count= " in log_text
    assert "retrieval_duration_ms= " in log_text
    assert "error_code=" in log_text
    assert private_prompt not in log_text
    assert local_key not in log_text
    assert "Authorization" not in log_text


def test_authentication_failure_is_logged_with_error_code(caplog) -> None:
    """Failed auth is diagnosable through the code, never the supplied key value."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with TestClient(create_app()) as client:
        response = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrong-key-SECRET"},
        )

    assert response.status_code == 401
    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "status_code=401" in log_text
    assert "error_code=invalid_api_key" in log_text
    assert "wrong-key-SECRET" not in log_text
