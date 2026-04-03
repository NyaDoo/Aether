"""
Gemini 图像生成模型请求适配

- 图像生成模型不支持 tools / system_instruction，需要移除
- responseModalities / responseMimeType 与 imageConfig 冲突，需要移除
"""

from typing import Any

from src.core.video_utils import is_image_gen_model

__all__ = ["is_image_gen_model", "adapt_request_for_image_gen"]


def adapt_request_for_image_gen(body: dict[str, Any]) -> dict[str, Any]:
    """为图像生成模型清理不兼容字段"""
    # 移除图像生成不支持的顶层字段
    for key in ("tools", "tool_config", "toolConfig", "system_instruction", "systemInstruction"):
        if key in body:
            body.pop(key)

    # 处理 generationConfig
    gc_key = "generationConfig" if "generationConfig" in body else "generation_config"
    gc = body.get(gc_key)
    if not isinstance(gc, dict):
        gc = {}
        body[gc_key] = gc

    # 移除与图像生成冲突的字段
    for key in (
        "responseMimeType",
        "response_mime_type",
        "responseModalities",
        "response_modalities",
    ):
        gc.pop(key, None)

    # 设置输出模态
    gc["responseModalities"] = ["TEXT", "IMAGE"]

    return body
