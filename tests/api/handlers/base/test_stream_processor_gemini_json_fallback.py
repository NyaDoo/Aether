import json
from typing import AsyncIterator, Optional

import pytest

from src.api.handlers.base.parsers import get_parser_for_format
from src.api.handlers.base.stream_context import StreamContext
from src.api.handlers.base.stream_processor import StreamProcessor
from src.core.api_format.conversion import register_default_normalizers


class _DummyResponseCtx:
    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


class _DummyHTTPClient:
    async def aclose(self) -> None:
        return None


async def _iter_bytes(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_stream_processor_converts_gemini_json_lines_without_data_prefix() -> None:
    register_default_normalizers()

    ctx = StreamContext(model="gemini-test", api_format="OPENAI")
    ctx.provider_api_format = "GEMINI"
    ctx.client_api_format = "OPENAI"
    ctx.needs_conversion = True
    ctx.request_id = "req_test"
    ctx.mapped_model = "gemini-test"

    # Simulate Gemini JSON-array/chunks stream: wrapper lines + two JSON objects.
    chunk1 = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello"}], "role": "model"},
            }
        ]
    }
    chunk2 = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello world"}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
    }

    upstream_lines = [
        b"[\n",
        (json.dumps(chunk1) + ",\n").encode("utf-8"),
        (json.dumps(chunk2) + "\n").encode("utf-8"),
        b"]\n",
    ]

    processor = StreamProcessor(
        request_id="req_test",
        default_parser=get_parser_for_format("OPENAI"),
    )

    out = b""
    async for b in processor.create_response_stream(
        ctx=ctx,
        byte_iterator=_iter_bytes(upstream_lines),
        response_ctx=_DummyResponseCtx(),
        http_client=_DummyHTTPClient(),  # type: ignore[arg-type]
        prefetched_chunks=[],
        start_time=None,
    ):
        out += b

    text = out.decode("utf-8", errors="replace")
    data_lines = [ln for ln in text.splitlines() if ln.startswith("data: ")]

    # OpenAI termination marker should be present (StreamProcessor will append if upstream doesn't send it).
    assert "data: [DONE]" in data_lines

    # Parse JSON events (excluding [DONE]) and validate we have expected deltas.
    events = [json.loads(ln[6:]) for ln in data_lines if ln != "data: [DONE]"]

    delta_contents: list[str] = []
    for evt in events:
        for choice in evt.get("choices", []) or []:
            delta = choice.get("delta") or {}
            if "content" in delta and delta["content"]:
                delta_contents.append(delta["content"])

    assert "Hello" in "".join(delta_contents)
    assert " world" in "".join(delta_contents)
