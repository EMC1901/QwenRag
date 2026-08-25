"""Minimal local OpenAI-compatible model server used only by automated tests."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    server_version = "QwenRAGFakeModel/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/v1/models":
            self._json({"data": [{"id": self.server.model_name}]})  # type: ignore[attr-defined]
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": {"message": "invalid json"}}, HTTPStatus.BAD_REQUEST)
            return
        if self.path == "/v1/chat/completions" and self.server.kind == "llm":  # type: ignore[attr-defined]
            if request.get("stream"):
                body = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                content = "OK"
                if request.get("max_tokens") == 128 and any(
                    isinstance(message, dict) and message.get("role") == "system"
                    for message in request.get("messages", [])
                ):
                    content = '{"need_rag": false, "reason_code": "general_knowledge"}'
                self._json(
                    {
                        "id": "fake-completion",
                        "object": "chat.completion",
                        "created": 0,
                        "model": request.get("model", self.server.model_name),  # type: ignore[attr-defined]
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                )
        elif self.path == "/v1/embeddings" and self.server.kind == "embedding":  # type: ignore[attr-defined]
            dimension = self.server.dimension  # type: ignore[attr-defined]
            self._json({"data": [{"index": 0, "embedding": [1.0, *([0.0] * (dimension - 1))]}]})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("llm", "embedding"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dimension", type=int, default=8)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.kind = args.kind  # type: ignore[attr-defined]
    server.model_name = args.model  # type: ignore[attr-defined]
    server.dimension = args.dimension  # type: ignore[attr-defined]
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
