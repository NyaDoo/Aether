"""
Rust execution runtime 客户端主入口。

旧的 `rust_executor_client.py` 仍然保留，作为兼容入口。
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config.settings import config
from src.services.request.execution_runtime_plan import ExecutionPlan


class ExecutionRuntimeClientError(RuntimeError):
    """Rust execution runtime 客户端错误。"""


@dataclass(slots=True)
class ExecutionRuntimeSyncResult:
    status_code: int
    response_json: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    provider_response_json: Any = None
    response_body_bytes: bytes | None = None


@dataclass(slots=True)
class ExecutionRuntimeStreamResult:
    status_code: int
    headers: dict[str, str]
    byte_iterator: AsyncIterator[bytes]
    response_ctx: Any


class _ExecutionRuntimeManagedStreamContext:
    def __init__(self, client: httpx.AsyncClient, response_ctx: Any) -> None:
        self._client = client
        self._response_ctx = response_ctx
        self._closed = False

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._response_ctx.__aexit__(exc_type, exc, tb)
        finally:
            await self._client.aclose()


class ExecutionRuntimeClient:
    """Python 控制面访问 Rust execution runtime 的轻量客户端。"""

    def __init__(
        self,
        *,
        transport: str | None = None,
        base_url: str | None = None,
        socket_path: str | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.transport = (transport or config.execution_runtime_transport).strip().lower()
        self.base_url = (base_url or config.execution_runtime_base_url).strip()
        self.socket_path = (socket_path or config.execution_runtime_socket_path).strip()
        self.request_timeout = (
            request_timeout
            if request_timeout is not None
            else config.execution_runtime_request_timeout
        )

    def _build_client(self, *, streaming: bool = False) -> httpx.AsyncClient:
        if streaming:
            timeout = httpx.Timeout(
                connect=min(self.request_timeout, 30.0),
                read=None,
                write=self.request_timeout,
                pool=self.request_timeout,
            )
        else:
            timeout = httpx.Timeout(self.request_timeout)
        if self.transport == "unix_socket":
            if not self.socket_path:
                raise ExecutionRuntimeClientError(
                    "EXECUTION_RUNTIME_SOCKET_PATH is required for unix_socket"
                )
            transport = httpx.AsyncHTTPTransport(uds=self.socket_path, retries=0)
            return httpx.AsyncClient(
                transport=transport,
                base_url=self.base_url,
                timeout=timeout,
            )

        if self.transport != "tcp":
            raise ExecutionRuntimeClientError(
                f"Unsupported execution-runtime transport: {self.transport}"
            )

        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            transport=httpx.AsyncHTTPTransport(retries=0),
        )

    async def execute_sync_json(self, plan: ExecutionPlan) -> ExecutionRuntimeSyncResult:
        async with self._build_client() as client:
            response = await client.post(
                "/v1/execute/sync",
                json=plan.to_payload(),
            )
            response.raise_for_status()
            payload = response.json()

        status_code = int(payload.get("status_code") or 200)
        headers = payload.get("headers") or {}
        if not isinstance(headers, dict):
            raise ExecutionRuntimeClientError("Execution runtime response headers must be an object")

        response_json = payload.get("response_json")
        provider_response_json = payload.get("provider_response_json")
        body_payload = payload.get("body")
        body_bytes_b64 = None
        if response_json is None and isinstance(body_payload, dict):
            response_json = body_payload.get("json_body")
            body_bytes_b64 = body_payload.get("body_bytes_b64")

        response_body_bytes: bytes | None = None
        if body_bytes_b64 is not None:
            if not isinstance(body_bytes_b64, str):
                raise ExecutionRuntimeClientError(
                    "Execution runtime body_bytes_b64 must be a string"
                )
            try:
                response_body_bytes = base64.b64decode(body_bytes_b64)
            except Exception as exc:  # noqa: BLE001
                raise ExecutionRuntimeClientError(
                    "Execution runtime body_bytes_b64 must be valid base64"
                ) from exc

        return ExecutionRuntimeSyncResult(
            status_code=status_code,
            response_json=response_json,
            headers={str(k): str(v) for k, v in headers.items()},
            provider_response_json=provider_response_json,
            response_body_bytes=response_body_bytes,
        )

    async def execute_stream(self, plan: ExecutionPlan) -> ExecutionRuntimeStreamResult:
        client = self._build_client(streaming=True)
        response_ctx = client.stream(
            "POST",
            "/v1/execute/stream",
            json=plan.to_payload(),
        )
        try:
            response = await response_ctx.__aenter__()
            response.raise_for_status()
            line_iter = response.aiter_lines()
            headers_frame = await self._read_first_stream_frame(line_iter)
            payload = headers_frame.get("payload")
            if not isinstance(payload, dict) or payload.get("kind") != "headers":
                raise ExecutionRuntimeClientError(
                    "Execution runtime stream must start with headers frame"
                )

            status_code = int(payload.get("status_code") or 200)
            headers = payload.get("headers") or {}
            if not isinstance(headers, dict):
                raise ExecutionRuntimeClientError(
                    "Execution runtime stream headers must be an object"
                )

            async def _byte_iter() -> AsyncIterator[bytes]:
                async for line in line_iter:
                    if not line:
                        continue
                    frame = self._decode_stream_frame(line)
                    frame_payload = frame["payload"]
                    kind = str(frame_payload.get("kind") or "").strip().lower()
                    if kind == "data":
                        chunk_b64 = frame_payload.get("chunk_b64")
                        if isinstance(chunk_b64, str):
                            if chunk_b64:
                                try:
                                    yield base64.b64decode(chunk_b64)
                                except Exception as exc:  # noqa: BLE001
                                    raise ExecutionRuntimeClientError(
                                        "Execution runtime stream chunk_b64 must be valid base64"
                                    ) from exc
                            continue

                        text = frame_payload.get("text")
                        if isinstance(text, str):
                            if text:
                                yield text.encode("utf-8")
                            continue

                    if kind == "error":
                        error = frame_payload.get("error") or {}
                        message = str(error.get("message") or "execution runtime stream error")
                        raise httpx.ReadError(message)

                    if kind == "telemetry":
                        continue

                    if kind == "eof":
                        break

                    raise ExecutionRuntimeClientError(
                        f"Unexpected execution runtime stream frame kind: {kind}"
                    )

            return ExecutionRuntimeStreamResult(
                status_code=status_code,
                headers={str(k): str(v) for k, v in headers.items()},
                byte_iterator=_byte_iter(),
                response_ctx=_ExecutionRuntimeManagedStreamContext(client, response_ctx),
            )
        except Exception:
            try:
                await response_ctx.__aexit__(None, None, None)
            finally:
                await client.aclose()
            raise

    @staticmethod
    def _decode_stream_frame(line: str) -> dict[str, Any]:
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExecutionRuntimeClientError(
                "Execution runtime stream frame must be valid JSON"
            ) from exc
        if not isinstance(frame, dict):
            raise ExecutionRuntimeClientError("Execution runtime stream frame must be an object")
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            raise ExecutionRuntimeClientError(
                "Execution runtime stream frame payload must be an object"
            )
        return frame

    async def _read_first_stream_frame(
        self,
        line_iter: AsyncIterator[str],
    ) -> dict[str, Any]:
        async for line in line_iter:
            if not line:
                continue
            return self._decode_stream_frame(line)
        raise ExecutionRuntimeClientError(
            "Execution runtime stream ended before headers frame"
        )


# Compatibility aliases for older call sites still using executor terminology.
RustExecutorClientError = ExecutionRuntimeClientError
RustExecutorSyncResult = ExecutionRuntimeSyncResult
RustExecutorStreamResult = ExecutionRuntimeStreamResult
RustExecutorClient = ExecutionRuntimeClient

__all__ = [
    "ExecutionRuntimeClient",
    "ExecutionRuntimeClientError",
    "ExecutionRuntimeStreamResult",
    "ExecutionRuntimeSyncResult",
    "RustExecutorClient",
    "RustExecutorClientError",
    "RustExecutorStreamResult",
    "RustExecutorSyncResult",
]
