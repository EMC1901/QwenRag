"""HTTP client for the server-side model gateway used by gateway answer mode."""

from collections.abc import AsyncIterator
from json import JSONDecodeError, dumps, loads
from typing import Any

import httpx

from local_rag_app.config import Settings
from local_rag_app.errors import (
    LocalRagError,
    gateway_auth_error,
    gateway_connection_error,
    gateway_http_error,
    gateway_invalid_response_error,
    gateway_timeout_error,
)
from local_rag_app.logging_config import record_upstream_result
from local_rag_app.schemas import ChatCompletionRequest


class ModelGatewayClient:
    """Call the configured model gateway without exposing local-client credentials."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def complete_chat(
        self,
        request: ChatCompletionRequest,
        *,
        local_model: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Call the non-streaming model-gateway chat endpoint and parse its JSON."""
        record_upstream_result("model_gateway")
        try:
            async with self._new_client() as client:
                response = await client.post(
                    self._chat_url,
                    json=self._build_upstream_body(request),
                    headers=self._headers(request_id),
                )
        except httpx.TimeoutException as exc:
            raise gateway_timeout_error() from exc
        except httpx.TransportError as exc:
            raise gateway_connection_error() from exc

        record_upstream_result("model_gateway", response.status_code)
        self._raise_for_status(response)
        try:
            payload = response.json()
        except (JSONDecodeError, ValueError) as exc:
            raise gateway_invalid_response_error() from exc
        if not isinstance(payload, dict):
            raise gateway_invalid_response_error()
        payload["model"] = local_model
        return payload

    async def open_chat_stream(
        self,
        request: ChatCompletionRequest,
        *,
        local_model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Open a gateway SSE response before local response headers are sent."""
        record_upstream_result("model_gateway")
        client = self._new_client()
        stream_context = client.stream(
            "POST",
            self._chat_url,
            json=self._build_upstream_body(request),
            headers=self._headers(request_id),
        )
        try:
            response = await stream_context.__aenter__()
        except httpx.TimeoutException as exc:
            await client.aclose()
            raise gateway_timeout_error() from exc
        except httpx.TransportError as exc:
            await client.aclose()
            raise gateway_connection_error() from exc

        record_upstream_result("model_gateway", response.status_code)
        try:
            self._raise_for_status(response)
        except LocalRagError:
            await stream_context.__aexit__(None, None, None)
            await client.aclose()
            raise

        return _GatewaySSEStream(
            response=response,
            stream_context=stream_context,
            client=client,
            local_model=local_model,
        )

    @property
    def _chat_url(self) -> str:
        return f"{self._settings.model_gateway_base_url}/chat/completions"

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self._transport,
            timeout=httpx.Timeout(
                connect=self._settings.http_connect_timeout_seconds,
                read=self._settings.http_read_timeout_seconds,
                write=self._settings.http_write_timeout_seconds,
                pool=self._settings.http_pool_timeout_seconds,
            ),
        )

    def _build_upstream_body(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Keep client options but replace the local business model with qwen."""
        body = request.model_dump(mode="json")
        body["model"] = self._settings.upstream_llm_model
        return body

    def _headers(self, request_id: str | None) -> dict[str, str]:
        """Use only the configured service credential for the model-gateway request."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._settings.model_gateway_api_key}",
        }
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if 200 <= response.status_code < 300:
            return
        if response.status_code in {401, 403}:
            raise gateway_auth_error()
        raise gateway_http_error()


class _GatewaySSEStream:
    """Own an open upstream response and rewrite only SSE JSON model identifiers."""

    def __init__(
        self,
        *,
        response: httpx.Response,
        stream_context: Any,
        client: httpx.AsyncClient,
        local_model: str,
    ) -> None:
        self._response = response
        self._stream_context = stream_context
        self._client = client
        self._local_model = local_model

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        try:
            async for line in self._response.aiter_lines():
                yield self._rewrite_sse_line(line)
        except httpx.TimeoutException as exc:
            raise gateway_timeout_error() from exc
        except httpx.TransportError as exc:
            raise gateway_connection_error() from exc
        finally:
            await self._stream_context.__aexit__(None, None, None)
            await self._client.aclose()

    def _rewrite_sse_line(self, line: str) -> bytes:
        """Preserve SSE framing while exposing local-rag instead of the upstream model."""
        if not line.startswith("data:"):
            return f"{line}\n".encode("utf-8")

        data = line[5:].lstrip()
        if data == "[DONE]":
            return b"data: [DONE]\n"
        try:
            payload = loads(data)
        except (JSONDecodeError, ValueError) as exc:
            raise gateway_invalid_response_error() from exc
        if not isinstance(payload, dict):
            raise gateway_invalid_response_error()
        payload["model"] = self._local_model
        return f"data: {dumps(payload, ensure_ascii=False)}\n".encode("utf-8")
