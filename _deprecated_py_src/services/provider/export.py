"""OAuth Key 导出：provider-specific export builders.

每个 Provider adapter 在 ``register_all()`` 时注册自己的 ``build_export_data``，
导出端点通过 ``build_export_data()`` 分发。

未注册的 provider_type 使用默认实现（strip null + 临时字段）。
"""

from __future__ import annotations

from typing import Any, Callable

# 所有 provider 共用的临时/无用字段
_DEFAULT_SKIP_KEYS = frozenset(
    {
        "access_token",
        "expires_at",
        "updated_at",
        "token_type",
        "scope",
    }
)

ExportBuilder = Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]

_BUILDERS: dict[str, ExportBuilder] = {}


def register_export_builder(provider_type: str, builder: ExportBuilder) -> None:
    _BUILDERS[provider_type] = builder


def _default_builder(
    auth_config: dict[str, Any],
    upstream_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """默认导出：去掉 null、空字符串、临时字段。"""
    return {
        k: v
        for k, v in auth_config.items()
        if k not in _DEFAULT_SKIP_KEYS and v is not None and v != ""
    }


def build_export_data(
    provider_type: str,
    auth_config: dict[str, Any],
    upstream_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """调用 provider-specific builder 构建导出数据。"""
    builder = _BUILDERS.get(provider_type, _default_builder)
    return builder(auth_config, upstream_metadata)
