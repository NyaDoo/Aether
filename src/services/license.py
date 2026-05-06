"""License validation and storage service."""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from sqlalchemy.orm import Session

from src.core.logger import logger
from src.models.database import SystemConfig
from src.services.system.config import SystemConfigService

LICENSE_CONFIG_KEY = "license_payload"


@dataclass(frozen=True)
class LicenseStatus:
    licensed: bool
    demo_mode: bool
    mode: str
    reason: str | None = None
    license_id: str | None = None
    customer: str | None = None
    edition: str | None = None
    expires_at: str | None = None
    issued_at: str | None = None
    features: list[str] | None = None
    limits: dict[str, Any] | None = None
    instance_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "licensed": self.licensed,
            "demo_mode": self.demo_mode,
            "mode": self.mode,
            "reason": self.reason,
            "license_id": self.license_id,
            "customer": self.customer,
            "edition": self.edition,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
            "features": self.features or [],
            "limits": self.limits or {},
            "instance_id": self.instance_id,
        }


class LicenseService:
    """Validate signed licenses and persist activated license payloads."""

    @classmethod
    def get_status(cls, db: Session) -> LicenseStatus:
        raw_license = cls._load_license_payload(db)
        if raw_license is None:
            return LicenseStatus(
                licensed=False,
                demo_mode=True,
                mode="unlicensed",
                reason="license_missing",
            )

        try:
            payload, signature = cls._split_payload_and_signature(raw_license)
        except ValueError as exc:
            return LicenseStatus(
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason=str(exc),
            )

        try:
            cls._verify_signature(payload, signature)
        except ValueError as exc:
            return LicenseStatus(
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason=str(exc),
            )

        try:
            expires_at = cls._parse_datetime(payload.get("expires_at"))
        except ValueError as exc:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason=str(exc),
            )
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="expired",
                reason="license_expired",
            )

        configured_instance_id = cls._configured_instance_id()
        license_instance_id = cls._optional_str(payload.get("instance_id"))
        if license_instance_id and configured_instance_id != license_instance_id:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason="instance_mismatch",
            )

        return cls._status_from_payload(
            payload,
            licensed=True,
            demo_mode=False,
            mode="licensed",
            reason=None,
        )

    @classmethod
    def activate(cls, db: Session, license_data: Any) -> LicenseStatus:
        normalized = cls._normalize_license_data(license_data)
        payload, signature = cls._split_payload_and_signature(normalized)
        cls._verify_signature(payload, signature)

        status = cls.get_status_for_payload(payload)
        if not status.licensed:
            raise ValueError(status.reason or "license_invalid")

        SystemConfigService.set_config(
            db,
            LICENSE_CONFIG_KEY,
            normalized,
            description="许可证授权信息",
        )
        return cls.get_status(db)

    @classmethod
    def get_status_for_payload(cls, payload: dict[str, Any]) -> LicenseStatus:
        try:
            expires_at = cls._parse_datetime(payload.get("expires_at"))
        except ValueError as exc:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason=str(exc),
            )
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="expired",
                reason="license_expired",
            )

        configured_instance_id = cls._configured_instance_id()
        license_instance_id = cls._optional_str(payload.get("instance_id"))
        if license_instance_id and configured_instance_id != license_instance_id:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason="instance_mismatch",
            )

        return cls._status_from_payload(payload, licensed=True, demo_mode=False, mode="licensed")

    @classmethod
    def _load_license_payload(cls, db: Session) -> Any | None:
        env_license = os.getenv("AETHER_LICENSE")
        if env_license:
            return cls._normalize_license_data(env_license)

        env_file = os.getenv("AETHER_LICENSE_FILE")
        if env_file:
            try:
                return cls._normalize_license_data(Path(env_file).read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("读取许可证文件失败: {}", exc)
                return None

        row = db.query(SystemConfig).filter(SystemConfig.key == LICENSE_CONFIG_KEY).first()
        return row.value if row else None

    @staticmethod
    def _normalize_license_data(license_data: Any) -> dict[str, Any]:
        if isinstance(license_data, dict):
            return license_data
        if isinstance(license_data, str):
            stripped = license_data.strip()
            if not stripped:
                raise ValueError("license_empty")
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("license_json_invalid") from exc
            if not isinstance(parsed, dict):
                raise ValueError("license_payload_invalid")
            return parsed
        raise ValueError("license_payload_invalid")

    @staticmethod
    def _split_payload_and_signature(license_data: dict[str, Any]) -> tuple[dict[str, Any], str]:
        signature = license_data.get("signature")
        if not isinstance(signature, str) or not signature.strip():
            raise ValueError("license_signature_missing")

        if isinstance(license_data.get("payload"), dict):
            payload = dict(license_data["payload"])
        else:
            payload = {k: v for k, v in license_data.items() if k != "signature"}

        if not payload:
            raise ValueError("license_payload_invalid")
        return payload, signature.strip()

    @classmethod
    def _verify_signature(cls, payload: dict[str, Any], signature: str) -> None:
        public_key = cls._load_public_key()
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        try:
            public_key.verify(cls._decode_signature(signature), canonical.encode("utf-8"))
        except (InvalidSignature, ValueError) as exc:
            raise ValueError("license_signature_invalid") from exc

    @staticmethod
    def _load_public_key() -> Ed25519PublicKey:
        key_text = os.getenv("AETHER_LICENSE_PUBLIC_KEY", "").strip()
        if not key_text:
            raise ValueError("license_public_key_missing")

        try:
            if "BEGIN PUBLIC KEY" in key_text:
                loaded = serialization.load_pem_public_key(
                    key_text.replace("\\n", "\n").encode("utf-8")
                )
                if not isinstance(loaded, Ed25519PublicKey):
                    raise ValueError("license_public_key_invalid")
                return loaded
            return Ed25519PublicKey.from_public_bytes(LicenseService._decode_base64(key_text))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("license_public_key_invalid") from exc

    @staticmethod
    def _decode_signature(value: str) -> bytes:
        try:
            return LicenseService._decode_base64(value)
        except ValueError:
            try:
                return bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError("license_signature_invalid") from exc

    @staticmethod
    def _decode_base64(value: str) -> bytes:
        normalized = value.strip()
        padding = "=" * (-len(normalized) % 4)
        for decoder in (base64.urlsafe_b64decode, base64.b64decode):
            try:
                return decoder((normalized + padding).encode("ascii"))
            except (binascii.Error, ValueError):
                continue
        raise ValueError("base64_invalid")

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        if not isinstance(value, str):
            raise ValueError("license_datetime_invalid")
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _configured_instance_id() -> str | None:
        return (
            os.getenv("AETHER_LICENSE_INSTANCE_ID")
            or os.getenv("AETHER_INSTANCE_ID")
            or None
        )

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _status_from_payload(
        cls,
        payload: dict[str, Any],
        *,
        licensed: bool,
        demo_mode: bool,
        mode: str,
        reason: str | None = None,
    ) -> LicenseStatus:
        features = payload.get("features")
        limits = payload.get("limits")
        return LicenseStatus(
            licensed=licensed,
            demo_mode=demo_mode,
            mode=mode,
            reason=reason,
            license_id=cls._optional_str(payload.get("license_id")),
            customer=cls._optional_str(payload.get("customer")),
            edition=cls._optional_str(payload.get("edition")),
            expires_at=cls._optional_str(payload.get("expires_at")),
            issued_at=cls._optional_str(payload.get("issued_at")),
            features=[str(item) for item in features] if isinstance(features, list) else [],
            limits=limits if isinstance(limits, dict) else {},
            instance_id=cls._optional_str(payload.get("instance_id")),
        )
