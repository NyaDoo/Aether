"""License validation and storage service."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import platform
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
LICENSE_TIME_ANCHOR_CONFIG_KEY = "license_time_anchor"
LICENSE_NTP_SERVER = "ntp.aliyun.com"
MACHINE_FINGERPRINT_VERSION = "aether-machine-v1"


@dataclass(frozen=True)
class LicenseRecord:
    payload: Any
    source: str


@dataclass(frozen=True)
class TimeCheckResult:
    now: datetime
    source: str
    anchor_at: datetime | None = None
    ntp_at: datetime | None = None
    ntp_error: str | None = None
    reason: str | None = None


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
    license_source: str | None = None
    machine_fingerprint: str | None = None
    machine_bound: bool = False
    can_deactivate: bool = False
    time_anchor_at: str | None = None
    time_source: str | None = None
    ntp_checked_at: str | None = None
    ntp_error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        if not self.licensed:
            return {
                "licensed": False,
                "demo_mode": self.demo_mode,
                "mode": "unlicensed",
                "reason": None,
                "license_id": None,
                "customer": None,
                "edition": None,
                "expires_at": None,
                "issued_at": None,
                "features": [],
                "limits": {},
                "instance_id": None,
                "machine_fingerprint": self.machine_fingerprint,
                "can_deactivate": self.can_deactivate,
            }
        return {
            "licensed": self.licensed,
            "demo_mode": self.demo_mode,
            "mode": "licensed",
            "reason": None,
            "license_id": self.license_id,
            "customer": self.customer,
            "edition": self.edition,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
            "features": self.features or [],
            "limits": self.limits or {},
            "instance_id": self.instance_id,
            "machine_fingerprint": self.machine_fingerprint,
            "can_deactivate": self.can_deactivate,
        }


class LicenseService:
    """Validate signed licenses and persist activated license payloads."""

    _ntp_cache: tuple[datetime, float] | None = None
    _ntp_failure_cache: tuple[str, float] | None = None
    _machine_profile_cache: dict[str, Any] | None = None

    @classmethod
    def get_status(cls, db: Session) -> LicenseStatus:
        time_check = cls._check_time(db)
        record = cls._load_license_record(db)
        if record is None:
            return LicenseStatus(
                licensed=False,
                demo_mode=True,
                mode="unlicensed",
                reason="license_missing",
                license_source=None,
                machine_fingerprint=cls.get_machine_fingerprint(),
                can_deactivate=False,
                time_anchor_at=(
                    time_check.anchor_at.isoformat() if time_check.anchor_at is not None else None
                ),
                time_source=time_check.source,
                ntp_checked_at=time_check.ntp_at.isoformat() if time_check.ntp_at else None,
                ntp_error=time_check.ntp_error,
            )

        try:
            payload, signature = cls._split_payload_and_signature(record.payload)
        except ValueError as exc:
            return LicenseStatus(
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason=str(exc),
                license_source=record.source,
                machine_fingerprint=cls.get_machine_fingerprint(),
                can_deactivate=record.source == "database",
                time_anchor_at=(
                    time_check.anchor_at.isoformat() if time_check.anchor_at is not None else None
                ),
                time_source=time_check.source,
                ntp_checked_at=time_check.ntp_at.isoformat() if time_check.ntp_at else None,
                ntp_error=time_check.ntp_error,
            )

        try:
            cls._verify_signature(payload, signature)
        except ValueError as exc:
            return LicenseStatus(
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason=str(exc),
                license_source=record.source,
                machine_fingerprint=cls.get_machine_fingerprint(),
                can_deactivate=record.source == "database",
                time_anchor_at=(
                    time_check.anchor_at.isoformat() if time_check.anchor_at is not None else None
                ),
                time_source=time_check.source,
                ntp_checked_at=time_check.ntp_at.isoformat() if time_check.ntp_at else None,
                ntp_error=time_check.ntp_error,
            )

        return cls.get_status_for_payload(
            payload,
            db=db,
            license_source=record.source,
            time_check=time_check,
        )

    @classmethod
    def activate(cls, db: Session, license_data: Any) -> LicenseStatus:
        if cls._load_env_license_record() is not None:
            raise ValueError("license_env_managed")

        normalized = cls._normalize_license_data(license_data)
        payload, signature = cls._split_payload_and_signature(normalized)
        cls._verify_signature(payload, signature)

        status = cls.get_status_for_payload(payload, db=db, license_source="database")
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
    def deactivate(cls, db: Session) -> LicenseStatus:
        if cls._load_env_license_record() is not None:
            raise ValueError("license_env_managed")
        SystemConfigService.delete_config(db, LICENSE_CONFIG_KEY)
        return cls.get_status(db)

    @classmethod
    def get_status_for_payload(
        cls,
        payload: dict[str, Any],
        db: Session | None = None,
        license_source: str | None = None,
        time_check: TimeCheckResult | None = None,
    ) -> LicenseStatus:
        try:
            expires_at = cls._parse_datetime(payload.get("expires_at"))
        except ValueError as exc:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason=str(exc),
                license_source=license_source,
            )

        time_check = time_check or cls._check_time(db)
        if expires_at is not None and expires_at <= time_check.now:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="expired",
                reason="license_expired",
                license_source=license_source,
                time_check=time_check,
            )
        if time_check.reason:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason=time_check.reason,
                license_source=license_source,
                time_check=time_check,
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
                license_source=license_source,
                time_check=time_check,
            )

        expected_machine_fingerprint = cls._payload_machine_fingerprint(payload)
        current_machine_fingerprint = cls.get_machine_fingerprint()
        if not expected_machine_fingerprint:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason="machine_fingerprint_missing",
                license_source=license_source,
                time_check=time_check,
            )
        if expected_machine_fingerprint != current_machine_fingerprint:
            return cls._status_from_payload(
                payload,
                licensed=False,
                demo_mode=True,
                mode="invalid",
                reason="machine_fingerprint_mismatch",
                license_source=license_source,
                time_check=time_check,
            )

        return cls._status_from_payload(
            payload,
            licensed=True,
            demo_mode=False,
            mode="licensed",
            license_source=license_source,
            time_check=time_check,
        )

    @classmethod
    def _load_license_record(cls, db: Session) -> LicenseRecord | None:
        env_record = cls._load_env_license_record()
        if env_record is not None:
            return env_record

        row = db.query(SystemConfig).filter(SystemConfig.key == LICENSE_CONFIG_KEY).first()
        return LicenseRecord(row.value, "database") if row else None

    @classmethod
    def _load_env_license_record(cls) -> LicenseRecord | None:
        env_license = os.getenv("AETHER_LICENSE")
        if env_license:
            return LicenseRecord(cls._normalize_license_data(env_license), "environment")

        env_file = os.getenv("AETHER_LICENSE_FILE")
        if env_file:
            try:
                return LicenseRecord(
                    cls._normalize_license_data(Path(env_file).read_text(encoding="utf-8")),
                    "file",
                )
            except Exception as exc:
                logger.warning("读取许可证文件失败: {}", exc)
                return None
        return None

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

    @classmethod
    def get_machine_binding(cls) -> dict[str, Any]:
        profile = cls.get_machine_profile()
        return {
            "fingerprint": cls.get_machine_fingerprint(profile),
            "fingerprint_version": MACHINE_FINGERPRINT_VERSION,
        }

    @classmethod
    def get_machine_fingerprint(cls, profile: dict[str, Any] | None = None) -> str:
        profile = profile or cls.get_machine_profile()
        canonical = json.dumps(
            profile,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def get_machine_profile(cls) -> dict[str, Any]:
        if cls._machine_profile_cache is not None:
            return cls._machine_profile_cache

        network = cls._collect_network_profile()
        profile_data = {
            "fingerprint_version": MACHINE_FINGERPRINT_VERSION,
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "machine_ids": cls._collect_machine_ids(),
            "network": network,
            "hardware": cls._collect_hardware_profile(),
        }
        cls._machine_profile_cache = profile_data
        return profile_data

    @staticmethod
    def _collect_machine_ids() -> list[str]:
        values: list[str] = []
        for path in (
            "/etc/machine-id",
            "/var/lib/dbus/machine-id",
            "/etc/hostid",
        ):
            try:
                value = Path(path).read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if value:
                values.append(value)
        return sorted(set(values))

    @staticmethod
    def _collect_network_profile() -> dict[str, Any]:
        macs: set[str] = set()
        ips: set[str] = set()

        node = uuid.getnode()
        if (node >> 40) % 2 == 0:
            macs.add(":".join(f"{(node >> shift) & 0xff:02x}" for shift in range(40, -1, -8)))

        try:
            import psutil

            for _interface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    address = str(getattr(addr, "address", "") or "").strip()
                    if not address:
                        continue
                    family = getattr(addr, "family", None)
                    if family in {socket.AF_INET, socket.AF_INET6}:
                        if not address.startswith(("127.", "::1", "fe80:")):
                            ips.add(address.split("%", 1)[0])
                    elif family is not None and "AF_LINK" in str(family):
                        normalized = address.replace("-", ":").lower()
                        if normalized and normalized != "00:00:00:00:00:00":
                            macs.add(normalized)
        except Exception as exc:
            logger.debug("采集网络指纹信息失败: {}", exc)

        return {"ips": sorted(ips), "macs": sorted(macs)}

    @staticmethod
    def _collect_hardware_profile() -> dict[str, Any]:
        hardware: dict[str, Any] = {"cpu_count": os.cpu_count() or 0}
        try:
            import psutil

            hardware["memory_total_bytes"] = int(psutil.virtual_memory().total)
            hardware["disk_total_bytes"] = int(psutil.disk_usage("/").total)
        except Exception as exc:
            logger.debug("采集硬件指纹信息失败: {}", exc)
        return hardware

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

    @classmethod
    def _check_time(cls, db: Session | None) -> TimeCheckResult:
        local_now = datetime.now(timezone.utc)
        if db is None:
            return TimeCheckResult(
                now=local_now,
                source="local",
                reason="license_time_anchor_unavailable",
            )

        ntp_now, ntp_error = cls._get_cached_ntp_time()
        anchor_at = cls._load_time_anchor(db)

        tolerance = cls._clock_rollback_tolerance()
        if anchor_at is not None and local_now + tolerance < anchor_at:
            result = TimeCheckResult(
                now=max(anchor_at, ntp_now or anchor_at),
                source="database_anchor",
                anchor_at=anchor_at,
                ntp_at=ntp_now,
                ntp_error=ntp_error,
                reason="license_clock_rollback",
            )
            cls._save_time_anchor(db, local_now, ntp_now, result)
            return result

        candidates = [local_now]
        if ntp_now is not None:
            candidates.append(ntp_now)
        if anchor_at is not None:
            candidates.append(anchor_at)

        effective_now = max(candidates)
        source_parts = ["local"]
        if ntp_now is not None and effective_now == ntp_now:
            source_parts = ["ntp"]
        if anchor_at is not None and effective_now == anchor_at:
            source_parts = ["database_anchor"]

        result = TimeCheckResult(
            now=effective_now,
            source="+".join(source_parts),
            anchor_at=anchor_at,
            ntp_at=ntp_now,
            ntp_error=ntp_error,
        )
        cls._save_time_anchor(db, local_now, ntp_now, result)
        return result

    @classmethod
    def _load_time_anchor(cls, db: Session | None) -> datetime | None:
        if db is None:
            return None
        row = (
            db.query(SystemConfig)
            .filter(SystemConfig.key == LICENSE_TIME_ANCHOR_CONFIG_KEY)
            .first()
        )
        if row is None or not isinstance(row.value, dict):
            return None
        try:
            return cls._parse_datetime(row.value.get("max_seen_at"))
        except Exception:
            return None

    @classmethod
    def _save_time_anchor(
        cls,
        db: Session | None,
        local_now: datetime,
        ntp_now: datetime | None,
        result: TimeCheckResult,
    ) -> None:
        if db is None:
            return

        max_seen_at = max([dt for dt in (local_now, ntp_now, result.anchor_at, result.now) if dt])
        value = {
            "max_seen_at": max_seen_at.isoformat(),
            "last_local_at": local_now.isoformat(),
            "last_effective_at": result.now.isoformat(),
            "last_ntp_at": ntp_now.isoformat() if ntp_now else None,
            "ntp_server": LICENSE_NTP_SERVER,
            "ntp_error": result.ntp_error,
            "clock_rollback_detected_at": (
                local_now.isoformat() if result.reason == "license_clock_rollback" else None
            ),
        }
        row = (
            db.query(SystemConfig)
            .filter(SystemConfig.key == LICENSE_TIME_ANCHOR_CONFIG_KEY)
            .first()
        )
        if row:
            row.value = value
            row.description = "许可证时间校验锚点"
        else:
            db.add(
                SystemConfig(
                    key=LICENSE_TIME_ANCHOR_CONFIG_KEY,
                    value=value,
                    description="许可证时间校验锚点",
                )
            )
        db.commit()

    @classmethod
    def _get_cached_ntp_time(cls) -> tuple[datetime | None, str | None]:
        cache_ttl = cls._ntp_cache_ttl()
        now_monotonic = time.monotonic()
        if cls._ntp_cache is not None:
            cached_time, cached_monotonic = cls._ntp_cache
            elapsed = now_monotonic - cached_monotonic
            if 0 <= elapsed <= cache_ttl:
                return cached_time + timedelta(seconds=elapsed), None
        if cls._ntp_failure_cache is not None:
            cached_error, cached_monotonic = cls._ntp_failure_cache
            elapsed = now_monotonic - cached_monotonic
            if 0 <= elapsed <= cache_ttl:
                return None, cached_error

        try:
            ntp_time = cls._query_ntp_time()
        except Exception as exc:
            logger.debug("NTP 时间校验失败: {}", exc)
            cls._ntp_failure_cache = ("ntp_unavailable", now_monotonic)
            return None, cls._ntp_failure_cache[0]

        cls._ntp_cache = (ntp_time, now_monotonic)
        cls._ntp_failure_cache = None
        return ntp_time, None

    @staticmethod
    def _query_ntp_time() -> datetime:
        packet = b"\x1b" + 47 * b"\0"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(float(os.getenv("AETHER_LICENSE_NTP_TIMEOUT", "1.5")))
            client.sendto(packet, (LICENSE_NTP_SERVER, 123))
            data, _ = client.recvfrom(48)
        if len(data) < 48:
            raise ValueError("invalid_ntp_response")
        seconds = struct.unpack("!12I", data)[10]
        unix_seconds = seconds - 2_208_988_800
        return datetime.fromtimestamp(unix_seconds, timezone.utc)

    @staticmethod
    def _ntp_cache_ttl() -> float:
        try:
            return max(float(os.getenv("AETHER_LICENSE_NTP_CACHE_SECONDS", "300")), 0.0)
        except ValueError:
            return 300.0

    @staticmethod
    def _clock_rollback_tolerance() -> timedelta:
        try:
            seconds = max(
                int(os.getenv("AETHER_LICENSE_CLOCK_ROLLBACK_TOLERANCE_SECONDS", "300")),
                0,
            )
        except ValueError:
            seconds = 300
        return timedelta(seconds=seconds)

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
    def _payload_machine_fingerprint(cls, payload: dict[str, Any]) -> str | None:
        direct = cls._optional_str(payload.get("machine_fingerprint"))
        if direct:
            return direct
        machine = payload.get("machine")
        if isinstance(machine, dict):
            return cls._optional_str(machine.get("fingerprint"))
        return None

    @classmethod
    def _status_from_payload(
        cls,
        payload: dict[str, Any],
        *,
        licensed: bool,
        demo_mode: bool,
        mode: str,
        reason: str | None = None,
        license_source: str | None = None,
        time_check: TimeCheckResult | None = None,
    ) -> LicenseStatus:
        expected_machine_fingerprint = cls._payload_machine_fingerprint(payload)
        current_machine_fingerprint = cls.get_machine_fingerprint()
        return LicenseStatus(
            licensed=licensed,
            demo_mode=demo_mode,
            mode=mode,
            reason=reason,
            license_id=cls._optional_str(payload.get("license_id")),
            customer=cls._optional_str(payload.get("customer")),
            edition="full" if licensed else None,
            expires_at=cls._optional_str(payload.get("expires_at")),
            issued_at=cls._optional_str(payload.get("issued_at")),
            features=["all"] if licensed else [],
            limits={},
            instance_id=cls._optional_str(payload.get("instance_id")),
            license_source=license_source,
            machine_fingerprint=current_machine_fingerprint,
            machine_bound=bool(
                expected_machine_fingerprint
                and expected_machine_fingerprint == current_machine_fingerprint
            ),
            can_deactivate=license_source == "database",
            time_anchor_at=(
                time_check.anchor_at.isoformat()
                if time_check is not None and time_check.anchor_at is not None
                else None
            ),
            time_source=time_check.source if time_check is not None else None,
            ntp_checked_at=(
                time_check.ntp_at.isoformat()
                if time_check is not None and time_check.ntp_at is not None
                else None
            ),
            ntp_error=time_check.ntp_error if time_check is not None else None,
        )
