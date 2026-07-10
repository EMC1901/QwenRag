"""Tests for stage 5 Server-Sent Events in stub mode."""

import json

from fastapi.testclient import TestClient

from local_rag_app.answer_service import STUB_ANSWER
from local_rag_app.main import create_app


def test_stub_stream_returns_sse_chunks_and_done_marker() -> None:
    """A streaming client must receive valid events and a terminal [DONE] marker."""
    request = {
        "model": "local-rag",
        "messages": [{"role": "user", "content": "流式测试"}],
        "stream": True,
    }

    with TestClient(create_app()) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer none"},
            json=request,
        ) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache"
    events = [event for event in body.split("\n\n") if event]
    assert events[-1] == "data: [DONE]"
    assert len(events) == 4

    chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    assert chunks[1]["choices"][0]["delta"] == {"role": None, "content": STUB_ANSWER}
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
