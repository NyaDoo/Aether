"""SSE 解析辅助函数"""

from __future__ import annotations

import json
from typing import Any

from src.core.logger import logger


def _parse_sse_data_line(line: str) -> tuple[Any | None, str]:
    """
    解析标准 SSE data 行

    Args:
        line: 以 "data:" 开头的 SSE 行

    Returns:
        (parsed_json, status) 元组：
        - (parsed_dict, "ok") - 解析成功
        - (None, "empty") - 内容为空
        - (None, "invalid") - JSON 解析失败，调用方应透传原始行
    """
    data_content = line[5:].strip()
    if not data_content:
        return None, "empty"
    try:
        return json.loads(data_content), "ok"
    except json.JSONDecodeError:
        return None, "invalid"


def _parse_sse_event_data_line(line: str) -> tuple[Any | None, str]:
    """
    解析 event + data 同行格式（如 "event: xxx data: {...}"）

    Args:
        line: 以 "event:" 开头且包含 " data:" 的 SSE 行

    Returns:
        (parsed_json, status) 元组
    """
    _event_part, data_part = line.split(" data:", 1)
    data_content = data_part.strip()
    try:
        return json.loads(data_content), "ok"
    except json.JSONDecodeError:
        return None, "invalid"


def _parse_gemini_json_array_line(line: str) -> tuple[Any | None, str]:
    """
    解析 Gemini JSON-array 格式的裸 JSON 行

    Gemini 流式响应可能是 JSON 数组格式，每行是数组元素。

    Args:
        line: 原始行（可能是 "[", "]", ",", 或 JSON 对象）

    Returns:
        (parsed_json, status) 元组
    """
    stripped = line.strip()
    if stripped in ("", "[", "]", ","):
        return None, "skip"

    candidate = stripped.lstrip(",").rstrip(",").strip()
    try:
        return json.loads(candidate), "ok"
    except json.JSONDecodeError:
        logger.debug("Gemini JSON-array line skip: {}", stripped[:50])
        return None, "invalid"


def _format_converted_events_to_sse(
    converted_events: list[dict[str, Any]],
    client_format: str,
) -> list[str]:
    """
    将转换后的事件格式化为 SSE 行

    Args:
        converted_events: 转换后的事件列表
        client_format: 客户端 API 格式

    Returns:
        SSE 行列表（每个元素是完整的 SSE 事件，包含尾部空行）
    """
    result: list[str] = []
    client_fmt = str(client_format or "").strip().lower()
    needs_event_line = client_fmt.startswith("claude:") or client_fmt == "openai:cli"

    for evt in converted_events:
        payload = json.dumps(evt, ensure_ascii=False)
        if needs_event_line:
            evt_type = evt.get("type") if isinstance(evt, dict) else None
            if isinstance(evt_type, str) and evt_type:
                # Claude 格式：event + data + 空行
                result.append(f"event: {evt_type}\ndata: {payload}\n")
            else:
                result.append(f"data: {payload}\n")
        else:
            # OpenAI 格式：data + 空行
            result.append(f"data: {payload}\n")

    return result
