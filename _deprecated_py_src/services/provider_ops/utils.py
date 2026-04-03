"""
Provider Ops 通用工具函数
"""


def extract_cookie_value(cookie_string: str, key: str) -> str:
    """
    从 Cookie 字符串中提取指定 key 的值

    支持两种输入格式：
    1. 完整 Cookie: "key=xxx; other=yyy; ..."
    2. 仅值: "MTc2ODc4..."

    Args:
        cookie_string: Cookie 字符串或直接的值
        key: 要提取的 Cookie key

    Returns:
        对应的值
    """
    if f"{key}=" in cookie_string:
        for part in cookie_string.split(";"):
            part = part.strip()
            if part.startswith(f"{key}="):
                return part[len(key) + 1 :]
    return cookie_string.strip()
