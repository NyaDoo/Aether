from __future__ import annotations

from src.api.handlers.base.utils import get_format_converter_registry
from src.services.provider.adapters.antigravity.constants import DUMMY_THOUGHT_SIGNATURE
from src.services.provider.adapters.antigravity.signature_cache import signature_cache


def _reset_sig_cache() -> None:
    # Module-global cache; tests must isolate state.
    signature_cache.clear()


def test_antigravity_converts_claude_thinking_block_to_gemini_thought_part_prefers_payload_sig() -> (
    None
):
    _reset_sig_cache()

    req = {
        "model": "gemini-3-pro",
        "max_tokens": 64,
        "thinking": {"type": "enabled", "budget_tokens": 1000},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "t1", "signature": "sig-1"},
                    {"type": "text", "text": "ok"},
                ],
            },
        ],
    }

    registry = get_format_converter_registry()
    out = registry.convert_request(req, "claude:chat", "gemini:chat", target_variant="antigravity")

    assert isinstance(out.get("contents"), list)
    model_turn = out["contents"][1]
    assert model_turn["role"] == "model"
    parts = model_turn["parts"]
    assert parts[0]["thought"] is True
    assert parts[0]["text"] == "t1"
    assert parts[0]["thoughtSignature"] == "sig-1"


def test_antigravity_uses_dummy_signature_for_gemini_when_missing() -> None:
    _reset_sig_cache()

    req = {
        "model": "gemini-3-pro",
        "max_tokens": 64,
        "thinking": {"type": "enabled", "budget_tokens": 1000},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "t2"},
                    {"type": "text", "text": "ok"},
                ],
            },
        ],
    }

    registry = get_format_converter_registry()
    out = registry.convert_request(req, "claude:chat", "gemini:chat", target_variant="antigravity")

    parts = out["contents"][1]["parts"]
    assert parts[0]["thought"] is True
    assert parts[0]["text"] == "t2"
    assert parts[0]["thoughtSignature"] == DUMMY_THOUGHT_SIGNATURE


def test_antigravity_drops_unsigned_thinking_for_non_gemini_models() -> None:
    _reset_sig_cache()

    req = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "thinking": {"type": "enabled", "budget_tokens": 1000},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "t3"},
                    {"type": "text", "text": "ok"},
                ],
            },
        ],
    }

    registry = get_format_converter_registry()
    out = registry.convert_request(req, "claude:chat", "gemini:chat", target_variant="antigravity")

    parts = out["contents"][1]["parts"]
    assert all(p.get("thought") is not True for p in parts)


def test_antigravity_inserts_dummy_thought_for_last_assistant_when_thinking_enabled() -> None:
    _reset_sig_cache()

    req = {
        "model": "gemini-3-pro",
        "max_tokens": 64,
        "thinking": {"type": "enabled", "budget_tokens": 1000},
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "prefill"}]},
        ],
    }

    registry = get_format_converter_registry()
    out = registry.convert_request(req, "claude:chat", "gemini:chat", target_variant="antigravity")

    parts = out["contents"][1]["parts"]
    assert parts[0]["thought"] is True
    assert parts[0]["thoughtSignature"] == DUMMY_THOUGHT_SIGNATURE
    assert parts[1]["text"] == "prefill"
