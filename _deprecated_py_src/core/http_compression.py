"""HTTP 压缩相关辅助函数。"""

from __future__ import annotations


def normalize_content_encoding(value: str | None) -> str | None:
    """标准化 Content-Encoding 值（仅做清洗，不做兼容扩展）。"""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def is_gzip_content_encoding(value: str | None) -> bool:
    """判断 Content-Encoding 是否为 gzip。"""
    return normalize_content_encoding(value) == "gzip"


def accepts_gzip(accept_encoding: str | None) -> bool:
    """判断 Accept-Encoding 是否可接受 gzip。"""
    if not isinstance(accept_encoding, str):
        return False

    for item in accept_encoding.split(","):
        token_and_params = [part.strip() for part in item.split(";") if part.strip()]
        if not token_and_params:
            continue

        encoding = token_and_params[0].lower()
        if encoding not in {"gzip", "*"}:
            continue

        quality = 1.0
        for param in token_and_params[1:]:
            if not param.lower().startswith("q="):
                continue
            try:
                quality = float(param[2:])
            except ValueError:
                quality = 0.0
            break

        if quality > 0:
            return True

    return False
