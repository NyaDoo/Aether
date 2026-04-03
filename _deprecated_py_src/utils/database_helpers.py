"""
数据库方言兼容性辅助函数
"""

from typing import Any

from sqlalchemy import func


def escape_like_pattern(pattern: str) -> str:
    """
    转义 SQL LIKE 语句中的特殊字符（%、_、\\）

    Args:
        pattern: 原始搜索模式

    Returns:
        转义后的模式，可安全用于 LIKE 查询（需配合 escape="\\\\"）

    Examples:
        >>> escape_like_pattern("hello_world%test")
        'hello\\\\_world\\\\%test'
    """
    return pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def safe_truncate_escaped(escaped: str, max_len: int) -> str:
    """
    安全截断已转义的字符串，避免截断在转义序列中间

    转义后的字符串中，反斜杠总是成对出现（\\\\）或作为转义符（\\%, \\_）。
    如果在某个位置截断导致末尾有奇数个反斜杠，说明截断发生在转义序列中间，
    需要去掉最后一个反斜杠以保持转义完整性。

    Args:
        escaped: 已经过 escape_like_pattern 处理的字符串
        max_len: 最大长度

    Returns:
        截断后的字符串，保证不会破坏转义序列
    """
    if len(escaped) <= max_len:
        return escaped

    truncated = escaped[:max_len]

    # 统计末尾连续的反斜杠数量
    trailing_backslashes = 0
    for i in range(len(truncated) - 1, -1, -1):
        if truncated[i] == "\\":
            trailing_backslashes += 1
        else:
            break

    # 如果末尾反斜杠数量为奇数，说明截断在转义序列中间
    # 需要去掉最后一个反斜杠
    if trailing_backslashes % 2 == 1:
        truncated = truncated[:-1]

    return truncated


def date_trunc_portable(dialect_name: str, interval: str, column: Any) -> Any:
    """
    跨数据库的日期截断函数

    Args:
        dialect_name: 数据库方言名称 ('postgresql', 'sqlite', 'mysql')
        interval: 时间间隔 ('day', 'week', 'month', 'year')
        column: 日期列

    Returns:
        SQLAlchemy ClauseElement

    Raises:
        NotImplementedError: 不支持的数据库方言

    Examples:
        >>> # PostgreSQL
        >>> period_func = date_trunc_portable('postgresql', 'week', Usage.created_at)
        >>> # 等价于: func.date_trunc('week', Usage.created_at)

        >>> # SQLite
        >>> period_func = date_trunc_portable('sqlite', 'month', Usage.created_at)
        >>> # 等价于: func.strftime("%Y-%m", Usage.created_at)
    """
    if dialect_name == "postgresql":
        # PostgreSQL 使用 date_trunc 函数
        return func.date_trunc(interval, column)

    elif dialect_name == "sqlite":
        # SQLite 使用 strftime 函数
        format_map = {
            "year": "%Y",
            "month": "%Y-%m",
            "week": "%Y-%W",
            "day": "%Y-%m-%d",
        }
        if interval not in format_map:
            raise ValueError(f"Unsupported interval for SQLite: {interval}")
        return func.strftime(format_map[interval], column)

    elif dialect_name == "mysql":
        # MySQL 使用 date_format 函数
        format_map = {
            "year": "%Y",
            "month": "%Y-%m",
            "day": "%Y-%m-%d",
        }
        if interval not in format_map:
            raise ValueError(f"Unsupported interval for MySQL: {interval}")
        return func.date_format(column, format_map[interval])

    else:
        raise NotImplementedError(
            f"Unsupported database dialect: {dialect_name}. "
            f"Supported dialects: postgresql, sqlite, mysql"
        )
