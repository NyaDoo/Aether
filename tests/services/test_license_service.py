from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.services.license import LicenseService


def _signed_license(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    signature = private_key.sign(canonical.encode("utf-8"))
    return (
        {**payload, "signature": base64.urlsafe_b64encode(signature).decode("ascii")},
        base64.urlsafe_b64encode(public_key).decode("ascii"),
    )


def test_signed_license_payload_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "license_id": "lic_test",
        "customer": "Example",
        "edition": "pro",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "features": ["proxy", "analytics"],
        "limits": {"max_users": 10},
    }
    license_data, public_key = _signed_license(payload)
    monkeypatch.setenv("AETHER_LICENSE_PUBLIC_KEY", public_key)

    parsed_payload, signature = LicenseService._split_payload_and_signature(license_data)
    LicenseService._verify_signature(parsed_payload, signature)
    status = LicenseService.get_status_for_payload(parsed_payload)

    assert status.licensed is True
    assert status.demo_mode is False
    assert status.license_id == "lic_test"
    assert status.features == ["proxy", "analytics"]


def test_invalid_license_signature_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "license_id": "lic_test",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    }
    license_data, public_key = _signed_license(payload)
    license_data["license_id"] = "tampered"
    monkeypatch.setenv("AETHER_LICENSE_PUBLIC_KEY", public_key)

    parsed_payload, signature = LicenseService._split_payload_and_signature(license_data)
    with pytest.raises(ValueError, match="license_signature_invalid"):
        LicenseService._verify_signature(parsed_payload, signature)


def test_expired_license_enters_demo_mode() -> None:
    payload = {
        "license_id": "lic_old",
        "expires_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    }

    status = LicenseService.get_status_for_payload(payload)

    assert status.licensed is False
    assert status.demo_mode is True
    assert status.mode == "expired"
