from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.models.database import Base, SystemConfig
from src.services.license import (
    LICENSE_CONFIG_KEY,
    LICENSE_TIME_ANCHOR_CONFIG_KEY,
    LicenseService,
)


MACHINE_FINGERPRINT = "test-machine-fingerprint"


@pytest.fixture(autouse=True)
def _stable_license_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AETHER_LICENSE", raising=False)
    monkeypatch.delenv("AETHER_LICENSE_FILE", raising=False)
    monkeypatch.setattr(LicenseService, "_ntp_cache", None)
    monkeypatch.setattr(LicenseService, "_ntp_failure_cache", None)
    monkeypatch.setattr(LicenseService, "_machine_profile_cache", None)
    monkeypatch.setattr(
        LicenseService,
        "get_machine_fingerprint",
        classmethod(lambda cls, profile=None: MACHINE_FINGERPRINT),
    )
    monkeypatch.setattr(
        LicenseService,
        "_get_cached_ntp_time",
        classmethod(lambda cls: (None, "ntp_unavailable")),
    )


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[SystemConfig.__table__])
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


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


def test_signed_license_payload_is_accepted(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "license_id": "lic_test",
        "customer": "Example",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "machine_fingerprint": MACHINE_FINGERPRINT,
    }
    license_data, public_key = _signed_license(payload)
    monkeypatch.setenv("AETHER_LICENSE_PUBLIC_KEY", public_key)

    parsed_payload, signature = LicenseService._split_payload_and_signature(license_data)
    LicenseService._verify_signature(parsed_payload, signature)
    status = LicenseService.get_status_for_payload(parsed_payload, db=db_session)

    assert status.licensed is True
    assert status.demo_mode is False
    assert status.license_id == "lic_test"
    assert status.edition == "full"
    assert status.features == ["all"]
    assert status.limits == {}
    assert status.machine_bound is True
    dumped = status.model_dump()
    assert dumped["reason"] is None
    assert dumped["mode"] == "licensed"
    assert "time_source" not in dumped
    assert "ntp_error" not in dumped


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


def test_expired_license_enters_demo_mode(db_session: Session) -> None:
    payload = {
        "license_id": "lic_old",
        "expires_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "machine_fingerprint": MACHINE_FINGERPRINT,
    }

    status = LicenseService.get_status_for_payload(payload, db=db_session)

    assert status.licensed is False
    assert status.demo_mode is True
    assert status.mode == "expired"


def test_stored_license_expires_on_status_check(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "license_id": "lic_stored_old",
        "expires_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "machine_fingerprint": MACHINE_FINGERPRINT,
    }
    license_data, public_key = _signed_license(payload)
    monkeypatch.setenv("AETHER_LICENSE_PUBLIC_KEY", public_key)
    db_session.add(
        SystemConfig(
            key=LICENSE_CONFIG_KEY,
            value=license_data,
            description="许可证授权信息",
        )
    )
    db_session.commit()

    status = LicenseService.get_status(db_session)
    dumped = status.model_dump()

    assert status.licensed is False
    assert status.reason == "license_expired"
    assert dumped["licensed"] is False
    assert dumped["mode"] == "unlicensed"
    assert dumped["expires_at"] is None


def test_license_requires_machine_fingerprint(db_session: Session) -> None:
    payload = {
        "license_id": "lic_unbound",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    }

    status = LicenseService.get_status_for_payload(payload, db=db_session)

    assert status.licensed is False
    assert status.reason == "machine_fingerprint_missing"
    assert status.model_dump()["mode"] == "unlicensed"


def test_machine_fingerprint_mismatch_is_rejected(db_session: Session) -> None:
    payload = {
        "license_id": "lic_wrong_machine",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "machine_fingerprint": "other-machine",
    }

    status = LicenseService.get_status_for_payload(payload, db=db_session)

    assert status.licensed is False
    assert status.reason == "machine_fingerprint_mismatch"


def test_database_time_anchor_prevents_expiry_bypass(db_session: Session) -> None:
    future_anchor = datetime.now(timezone.utc) + timedelta(days=60)
    db_session.add(
        SystemConfig(
            key=LICENSE_TIME_ANCHOR_CONFIG_KEY,
            value={"max_seen_at": future_anchor.isoformat()},
            description="许可证时间校验锚点",
        )
    )
    db_session.commit()
    payload = {
        "license_id": "lic_anchor_expired",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "machine_fingerprint": MACHINE_FINGERPRINT,
    }

    status = LicenseService.get_status_for_payload(payload, db=db_session)

    assert status.licensed is False
    assert status.reason == "license_expired"
    assert status.model_dump()["expires_at"] is None


def test_clock_rollback_is_rejected(db_session: Session) -> None:
    future_anchor = datetime.now(timezone.utc) + timedelta(days=1)
    db_session.add(
        SystemConfig(
            key=LICENSE_TIME_ANCHOR_CONFIG_KEY,
            value={"max_seen_at": future_anchor.isoformat()},
            description="许可证时间校验锚点",
        )
    )
    db_session.commit()
    payload = {
        "license_id": "lic_rollback",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "machine_fingerprint": MACHINE_FINGERPRINT,
    }

    status = LicenseService.get_status_for_payload(payload, db=db_session)

    assert status.licensed is False
    assert status.reason == "license_clock_rollback"


def test_activate_and_deactivate_database_license(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "license_id": "lic_db",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        "machine_fingerprint": MACHINE_FINGERPRINT,
    }
    license_data, public_key = _signed_license(payload)
    monkeypatch.setenv("AETHER_LICENSE_PUBLIC_KEY", public_key)

    activated = LicenseService.activate(db_session, license_data)
    assert activated.licensed is True
    assert db_session.query(SystemConfig).filter(SystemConfig.key == LICENSE_CONFIG_KEY).first()

    deactivated = LicenseService.deactivate(db_session)

    assert deactivated.licensed is False
    assert deactivated.reason == "license_missing"
    assert db_session.query(SystemConfig).filter(SystemConfig.key == LICENSE_CONFIG_KEY).first() is None


def test_machine_binding_response_only_exposes_fingerprint() -> None:
    binding = LicenseService.get_machine_binding()

    assert binding == {
        "fingerprint": MACHINE_FINGERPRINT,
        "fingerprint_version": "aether-machine-v1",
    }
