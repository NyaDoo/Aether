from __future__ import annotations

from copy import deepcopy
from typing import Any

MODEL_TEST_DEBUG_KEY = "model_test_debug"
MODEL_TEST_DEBUG_ATTR = "_model_test_debug"


def normalize_model_test_debug_payload(debug_payload: Any) -> dict[str, Any] | None:
    if not isinstance(debug_payload, dict):
        return None

    normalized: dict[str, Any] = {}
    for key in (
        "request_url",
        "request_headers",
        "request_body",
        "response_headers",
        "response_body",
    ):
        value = debug_payload.get(key)
        if value is None:
            continue
        normalized[key] = deepcopy(value)

    return normalized or None


def set_candidate_model_test_debug(candidate: Any, debug_payload: Any) -> None:
    normalized = normalize_model_test_debug_payload(debug_payload)
    if normalized is None:
        return
    setattr(candidate, MODEL_TEST_DEBUG_ATTR, normalized)


def get_candidate_model_test_debug(candidate: Any) -> dict[str, Any] | None:
    return normalize_model_test_debug_payload(getattr(candidate, MODEL_TEST_DEBUG_ATTR, None))


def merge_model_test_debug(
    extra_data: dict[str, Any] | None,
    debug_payload: Any,
) -> dict[str, Any] | None:
    normalized = normalize_model_test_debug_payload(debug_payload)
    if normalized is None:
        return extra_data

    merged = dict(extra_data or {})
    merged[MODEL_TEST_DEBUG_KEY] = normalized
    return merged


def get_model_test_debug_from_extra_data(extra_data: Any) -> dict[str, Any] | None:
    if not isinstance(extra_data, dict):
        return None
    return normalize_model_test_debug_payload(extra_data.get(MODEL_TEST_DEBUG_KEY))
