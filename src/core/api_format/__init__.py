"""
API 格式核心模块

统一管理 API 格式相关的枚举、元数据、工具函数等。

模块组成：
- enums.py: APIFormat 枚举定义
- metadata.py: 格式元数据定义（别名、路径、认证等）
- headers.py: 请求头处理（构建、过滤、脱敏）
- utils.py: 工具函数（is_cli_format, get_base_format 等）
- detection.py: 格式检测（从请求头、响应内容检测格式）
"""
from src.core.api_format.detection import (
    detect_cli_format_from_path,
    detect_format_and_key_from_starlette,
    detect_format_from_request,
    detect_format_from_response,
)
from src.core.api_format.enums import APIFormat
from src.core.api_format.headers import (
    CORE_REDACT_HEADERS,
    HOP_BY_HOP_HEADERS,
    RESPONSE_DROP_HEADERS,
    SENSITIVE_HEADERS,
    UPSTREAM_DROP_HEADERS,
    HeaderBuilder,
    build_adapter_base_headers,
    build_adapter_headers,
    build_upstream_headers,
    detect_capabilities,
    extract_client_api_key,
    extract_client_api_key_with_query,
    extract_set_headers_from_rules,
    filter_response_headers,
    get_adapter_protected_keys,
    get_extra_headers_from_endpoint,
    get_header_value,
    merge_headers_with_protection,
    normalize_headers,
    redact_headers_for_log,
)
from src.core.api_format.metadata import (
    API_FORMAT_DEFINITIONS,
    ApiFormatDefinition,
    get_api_format_definition,
    get_auth_config,
    get_default_path,
    get_extra_headers,
    get_local_path,
    get_protected_keys,
    is_cli_api_format,
    list_api_format_definitions,
    register_api_format_definition,
    resolve_api_format,
    resolve_api_format_alias,
)
from src.core.api_format.utils import (
    get_base_format,
    is_cli_format,
    is_convertible_format,
    is_same_format,
    normalize_format,
)

__all__ = [
    # Enums
    "APIFormat",
    # Metadata
    "ApiFormatDefinition",
    "API_FORMAT_DEFINITIONS",
    "get_api_format_definition",
    "list_api_format_definitions",
    "resolve_api_format",
    "resolve_api_format_alias",
    "register_api_format_definition",
    "get_default_path",
    "get_local_path",
    "get_auth_config",
    "get_extra_headers",
    "get_protected_keys",
    "is_cli_api_format",
    # Utils
    "is_cli_format",
    "get_base_format",
    "normalize_format",
    "is_same_format",
    "is_convertible_format",
    # Headers
    "UPSTREAM_DROP_HEADERS",
    "CORE_REDACT_HEADERS",
    "HOP_BY_HOP_HEADERS",
    "RESPONSE_DROP_HEADERS",
    "SENSITIVE_HEADERS",
    "normalize_headers",
    "get_header_value",
    "extract_client_api_key",
    "extract_client_api_key_with_query",
    "detect_capabilities",
    "HeaderBuilder",
    "build_upstream_headers",
    "merge_headers_with_protection",
    "filter_response_headers",
    "redact_headers_for_log",
    "build_adapter_base_headers",
    "build_adapter_headers",
    "get_adapter_protected_keys",
    "extract_set_headers_from_rules",
    "get_extra_headers_from_endpoint",
    # Detection
    "detect_format_from_request",
    "detect_format_and_key_from_starlette",
    "detect_format_from_response",
    "detect_cli_format_from_path",
]
