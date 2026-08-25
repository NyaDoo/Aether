#!/usr/bin/env python3
"""Run the local Ark material-assets API end-to-end and write an HTTP transcript.

The harness uses only Python's standard library. It logs in with the local
administrator, creates disposable Aether API-key and AK/SK credentials, runs
native Action and console REST checks, and removes every deletable resource it
created in a finally block.

Required environment variables:
  AETHER_ADMIN_EMAIL
  AETHER_ADMIN_PASSWORD

Typical local invocation:
  set -a; source .env; set +a
  AETHER_ADMIN_EMAIL="$ADMIN_EMAIL" AETHER_ADMIN_PASSWORD="$ADMIN_PASSWORD" \
    python3 tools/test_material_assets_e2e.py

Reports redact credentials, signatures, cookies, verification tokens, and URL
query secrets. Report body lengths and hashes are computed after redaction;
resource IDs, non-secret request fields, status codes, response envelopes, and
non-secret headers are retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable


JSON = dict[str, Any] | list[Any] | str | int | float | bool | None
Validator = Callable[[int, JSON | None], str | None]

DEFAULT_IMAGE_URL = (
    "https://ark-project.tos-cn-beijing.volces.com/doc_image/r2v_tea_pic1.jpg"
)
DEFAULT_VIDEO_URL = (
    "https://ark-project.tos-cn-beijing.volces.com/doc_video/r2v_tea_video1.mp4"
)
DEFAULT_AUDIO_URL = (
    "https://ark-project.tos-cn-beijing.volces.com/doc_audio/r2v_tea_audio1.mp3"
)
DEFAULT_CALLBACK_URL = "https://example.com/aether-material-assets-e2e/callback"
VERSION = "2024-01-01"
REGION = "cn-beijing"
SERVICE = "ark"

SECRET_JSON_KEYS = {
    "access_token",
    "refresh_token",
    "password",
    "key",
    "access_key_id",
    "secret_access_key",
    "bytedtoken",
    "byted_token",
}
SECRET_HEADER_NAMES = {
    "api-key",
    "x-api-key",
    "cookie",
    "set-cookie",
    "proxy-authorization",
}
URL_KEYS = {
    "url",
    "h5link",
    "h5_link",
    "verification_url",
    "preview_url",
    "callbackurl",
    "callback_url",
}

OFFICIAL_GROUP_ID = re.compile(r"^group-\d{14}-[a-z0-9]{5}$")
OFFICIAL_ASSET_ID = re.compile(r"^asset-\d{14}-[a-z0-9]{5}$")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def mask_identifier(value: str) -> str:
    value = value.strip()
    if len(value) <= 8:
        return "<redacted>"
    return f"{value[:4]}…{value[-4:]}"


def sensitive_query_name(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return any(
        marker in normalized
        for marker in ("token", "signature", "credential", "secret", "api_key")
    ) or normalized in {"key", "authorization"}


def redact_url(value: str, *, redact_all_query: bool = False) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc or not parsed.query:
        return value
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = urllib.parse.urlencode(
        [
            (
                name,
                "<redacted>"
                if redact_all_query or sensitive_query_name(name)
                else item,
            )
            for name, item in pairs
        ],
        doseq=True,
    )
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, redacted_query, parsed.fragment)
    )


def redact_authorization(value: str) -> str:
    stripped = value.strip()
    if stripped.lower().startswith("bearer "):
        return "Bearer <redacted>"
    if not stripped.startswith("HMAC-SHA256 "):
        return "<redacted>"
    fields: list[str] = []
    for raw in stripped.removeprefix("HMAC-SHA256 ").split(","):
        name, separator, field_value = raw.strip().partition("=")
        if not separator:
            continue
        if name == "Credential":
            access_key, slash, scope = field_value.partition("/")
            field_value = f"{mask_identifier(access_key)}{slash}{scope}"
        elif name == "Signature":
            field_value = "<redacted>"
        fields.append(f"{name}={field_value}")
    return "HMAC-SHA256 " + ", ".join(fields)


def redact_headers(headers: dict[str, str] | Iterable[tuple[str, str]]) -> dict[str, str]:
    items = headers.items() if isinstance(headers, dict) else headers
    output: dict[str, str] = {}
    for raw_name, raw_value in items:
        name = str(raw_name)
        value = str(raw_value)
        lower = name.lower()
        if lower == "authorization":
            output[name] = redact_authorization(value)
        elif lower in SECRET_HEADER_NAMES:
            output[name] = "<redacted>"
        else:
            output[name] = value
    return output


def redact_json(value: Any, key: str | None = None) -> Any:
    lower_key = (key or "").lower()
    if lower_key in SECRET_JSON_KEYS:
        if lower_key in {"bytedtoken", "byted_token"} and isinstance(value, str):
            return mask_identifier(value)
        return "<redacted>"
    if isinstance(value, dict):
        return {str(name): redact_json(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [redact_json(item, key) for item in value]
    if isinstance(value, str) and lower_key in URL_KEYS:
        return redact_url(
            value,
            redact_all_query=lower_key
            in {"h5link", "h5_link", "verification_url", "preview_url", "callbackurl", "callback_url"},
        )
    return value


def official_group_id(value: str) -> bool:
    return OFFICIAL_GROUP_ID.fullmatch(value) is not None


def official_asset_id(value: str) -> bool:
    return OFFICIAL_ASSET_ID.fullmatch(value) is not None


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hmac_sha256(key: bytes, value: bytes) -> bytes:
    return hmac.new(key, value, hashlib.sha256).digest()


def canonical_query(query: str) -> str:
    pairs: list[tuple[str, str]] = []
    for name, value in urllib.parse.parse_qsl(query, keep_blank_values=True):
        encoded_name = urllib.parse.quote(name, safe="-_.~")
        encoded_value = urllib.parse.quote(value, safe="-_.~")
        pairs.append((encoded_name, encoded_value))
    pairs.sort()
    return "&".join(f"{name}={value}" for name, value in pairs)


def normalize_header_value(value: str) -> str:
    return " ".join(value.split())


def native_success_validator(action: str) -> Validator:
    def validate(_status: int, payload: JSON | None) -> str | None:
        if not isinstance(payload, dict):
            return "response is not a JSON object"
        metadata = payload.get("ResponseMetadata")
        if not isinstance(metadata, dict):
            return "ResponseMetadata is missing"
        error = metadata.get("Error")
        if error not in (None, {}):
            return f"ResponseMetadata.Error is present: {error}"
        if metadata.get("Action") != action:
            return f"ResponseMetadata.Action is not {action}"
        if metadata.get("Version") != VERSION:
            return f"ResponseMetadata.Version is not {VERSION}"
        if not isinstance(payload.get("Result"), dict):
            return "Result is missing or is not an object"
        return None

    return validate


def object_validator(required_keys: Iterable[str] = ()) -> Validator:
    required = tuple(required_keys)

    def validate(_status: int, payload: JSON | None) -> str | None:
        if not isinstance(payload, dict):
            return "response is not a JSON object"
        missing = [name for name in required if name not in payload]
        if missing:
            return f"missing response fields: {', '.join(missing)}"
        return None

    return validate


class Harness:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("--base-url must be an absolute HTTP(S) URL")
        self.host = parsed.netloc
        self.run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.name_prefix = f"aether-e2e-{self.run_id.lower()}"
        self.device_id = f"material-assets-e2e-{self.run_id}"
        default_report_dir = Path("/tmp") / f"aether-material-assets-e2e-{self.run_id}"
        self.report_dir = Path(args.report_dir) if args.report_dir else default_report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.json_report = self.report_dir / "requests-responses.json"
        self.markdown_report = self.report_dir / "requests-responses.md"
        self.records: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.skipped: list[dict[str, str]] = []
        self.jwt: str | None = None
        self.user_id: str | None = None
        self.api_key_id: str | None = None
        self.api_key: str | None = None
        self.aksk_id: str | None = None
        self.access_key_id: str | None = None
        self.secret_access_key: str | None = None
        self.created_group_ids: set[str] = set()
        self.created_asset_ids: set[str] = set()
        self.asset_statuses: dict[str, str] = {}
        self.verification_session_ids: list[str] = []
        self.native_verification_sessions_created = 0
        self.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        self._write_reports()

    def _absolute_url(self, path: str) -> str:
        if not path.startswith("/"):
            raise ValueError(f"request path must start with '/': {path}")
        return self.base_url + path

    def _write_reports(self) -> None:
        passed = sum(1 for item in self.records if item["passed"])
        failed = sum(1 for item in self.records if not item["passed"] and item["required"])
        optional_failed = sum(
            1 for item in self.records if not item["passed"] and not item["required"]
        )
        check_failed = sum(
            1 for item in self.checks if not item["passed"] and item["required"]
        )
        payload = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "base_url": self.base_url,
            "summary": {
                "requests": len(self.records),
                "passed": passed,
                "failed": failed,
                "optional_failed": optional_failed,
                "checks": len(self.checks),
                "check_failures": check_failed,
                "skipped": len(self.skipped),
            },
            "created_resource_state": {
                "remaining_group_ids": sorted(self.created_group_ids),
                "remaining_asset_ids": sorted(self.created_asset_ids),
                "verification_session_ids": self.verification_session_ids,
                "native_verification_sessions_created": self.native_verification_sessions_created,
            },
            "requests": self.records,
            "checks": self.checks,
            "skipped": self.skipped,
        }
        self.json_report.write_text(pretty_json(payload) + "\n", encoding="utf-8")

        lines = [
            "# Aether 素材库本地 E2E 请求与响应",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Base URL: `{self.base_url}`",
            f"- 请求数: {len(self.records)}",
            f"- 请求通过: {passed}",
            f"- 必测请求失败: {failed}",
            f"- 必测断言失败: {check_failed}",
            f"- 跳过: {len(self.skipped)}",
            "- 安全处理：凭据、签名、Cookie、真人验证 Token 与 URL 查询密钥已脱敏。",
            "- 正文长度与 SHA-256 均基于脱敏后的报告正文，不是原始密钥的可验证摘要。",
            "",
        ]
        for record in self.records:
            state = "PASS" if record["passed"] else ("FAIL" if record["required"] else "INFO")
            lines.extend(
                [
                    f"## {record['index']:03d}. [{state}] {record['label']}",
                    "",
                    f"- 时间: `{record['started_at']}`",
                    f"- 耗时: `{record['duration_ms']} ms`",
                    f"- 期望状态: `{record['expected_status']}`",
                    f"- 实际状态: `{record['response']['status']}`",
                ]
            )
            if record.get("failure"):
                lines.append(f"- 失败原因: `{record['failure']}`")
            lines.extend(
                [
                    "",
                    "### Request",
                    "",
                    "```text",
                    f"{record['request']['method']} {record['request']['url']}",
                    pretty_json(record["request"]["headers"]),
                    "",
                    record["request"].get("body", ""),
                    "```",
                    "",
                    "### Response",
                    "",
                    "```text",
                    f"HTTP {record['response']['status']}",
                    pretty_json(record["response"]["headers"]),
                    "",
                    record["response"].get("body", ""),
                    "```",
                    "",
                ]
            )
        if self.checks:
            lines.extend(["# 业务断言", ""])
            for check in self.checks:
                state = "PASS" if check["passed"] else "FAIL"
                lines.extend(
                    [
                        f"- [{state}] {check['label']}: {check['detail']}",
                    ]
                )
            lines.append("")
        if self.skipped:
            lines.extend(["# 跳过项", ""])
            for item in self.skipped:
                lines.append(f"- {item['label']}: {item['reason']}")
            lines.append("")
        self.markdown_report.write_text("\n".join(lines), encoding="utf-8")

    def add_check(self, label: str, passed: bool, detail: str, required: bool = True) -> None:
        self.checks.append(
            {"label": label, "passed": passed, "detail": detail, "required": required}
        )
        state = "PASS" if passed else "FAIL"
        print(f"[CHECK {state}] {label}: {detail}", flush=True)
        self._write_reports()

    def skip(self, label: str, reason: str) -> None:
        self.skipped.append({"label": label, "reason": reason})
        print(f"[SKIP] {label}: {reason}", flush=True)
        self._write_reports()

    def request(
        self,
        label: str,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        has_json_body: bool = False,
        expected_status: Iterable[int] = (200,),
        validator: Validator | None = None,
        required: bool = True,
        binary_response: bool = False,
    ) -> tuple[int, JSON | None, bytes, bool]:
        method = method.upper()
        url = self._absolute_url(path)
        request_headers = dict(headers or {})
        request_body = b""
        if has_json_body:
            request_body = compact_json(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        data = request_body if has_json_body else None
        expected = tuple(expected_status)
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        started = time.perf_counter()
        response_status = 0
        response_headers: dict[str, str] = {}
        response_bytes = b""
        transport_error: str | None = None
        request = urllib.request.Request(
            url=url,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                response_status = response.status
                response_headers = dict(response.headers.items())
                response_bytes = response.read()
        except urllib.error.HTTPError as error:
            response_status = error.code
            response_headers = dict(error.headers.items()) if error.headers else {}
            response_bytes = error.read()
        except Exception as error:  # noqa: BLE001 - transcript transport failures verbatim
            transport_error = f"{type(error).__name__}: {error}"
        duration_ms = round((time.perf_counter() - started) * 1000, 2)

        response_json: JSON | None = None
        response_text = ""
        response_report_bytes = b""
        if response_bytes:
            try:
                decoded = response_bytes.decode("utf-8")
                response_json = json.loads(decoded)
                redacted_response = redact_json(response_json)
                response_text = pretty_json(redacted_response)
                response_report_bytes = compact_json(redacted_response).encode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError):
                if binary_response:
                    binary_summary = {
                        "binary": True,
                        "length": len(response_bytes),
                        "sha256": sha256_hex(response_bytes),
                        "content_type": response_headers.get("Content-Type"),
                    }
                    response_text = pretty_json(binary_summary)
                    response_report_bytes = compact_json(binary_summary).encode("utf-8")
                else:
                    response_text = response_bytes.decode("utf-8", errors="replace")
                    response_report_bytes = response_text.encode("utf-8")
        failure = transport_error
        if failure is None and response_status not in expected:
            failure = f"expected HTTP {expected}, got {response_status}"
        if failure is None and validator is not None:
            failure = validator(response_status, response_json)
        passed = failure is None

        request_body_for_report = ""
        request_report_bytes = b""
        if has_json_body:
            redacted_request = redact_json(json_body)
            request_body_for_report = pretty_json(redacted_request)
            request_report_bytes = compact_json(redacted_request).encode("utf-8")
        record = {
            "index": len(self.records) + 1,
            "label": label,
            "required": required,
            "passed": passed,
            "failure": failure,
            "started_at": started_at,
            "duration_ms": duration_ms,
            "expected_status": list(expected),
            "request": {
                "method": method,
                "url": redact_url(url),
                "headers": redact_headers(request_headers),
                "body": request_body_for_report,
                "body_sha256": sha256_hex(request_report_bytes),
                "body_length": len(request_report_bytes),
            },
            "response": {
                "status": response_status,
                "headers": redact_headers(response_headers),
                "body": response_text,
                "body_sha256": sha256_hex(response_report_bytes),
                "body_length": len(response_report_bytes),
            },
        }
        self.records.append(record)
        state = "PASS" if passed else ("FAIL" if required else "INFO")
        print(
            f"[{record['index']:03d}] {state} {label} -> HTTP {response_status} ({duration_ms} ms)",
            flush=True,
        )
        if failure:
            print(f"      {failure}", flush=True)
        self._write_reports()
        return response_status, response_json, response_bytes, passed

    def jwt_headers(self) -> dict[str, str]:
        if not self.jwt:
            raise RuntimeError("JWT is unavailable")
        return {
            "Authorization": f"Bearer {self.jwt}",
            "X-Client-Device-Id": self.device_id,
        }

    def api_key_headers(self, carrier: str = "bearer") -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("temporary Aether API key is unavailable")
        if carrier == "bearer":
            return {"Authorization": f"Bearer {self.api_key}"}
        if carrier == "x-api-key":
            return {"X-Api-Key": self.api_key}
        if carrier == "api-key":
            return {"Api-Key": self.api_key}
        raise ValueError(f"unsupported API-key carrier: {carrier}")

    def signed_headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        if not self.access_key_id or not self.secret_access_key:
            raise RuntimeError("temporary Aether AK/SK is unavailable")
        parsed = urllib.parse.urlsplit(self._absolute_url(path))
        now = dt.datetime.now(dt.timezone.utc)
        x_date = now.strftime("%Y%m%dT%H%M%SZ")
        short_date = now.strftime("%Y%m%d")
        payload_hash = sha256_hex(body)
        signing_headers = {
            "content-type": "application/json",
            "host": self.host,
            "x-content-sha256": payload_hash,
            "x-date": x_date,
        }
        signed_names = ";".join(sorted(signing_headers))
        canonical_headers = "".join(
            f"{name}:{normalize_header_value(signing_headers[name])}\n"
            for name in sorted(signing_headers)
        )
        canonical_uri = urllib.parse.quote(parsed.path or "/", safe="/-_.~")
        canonical_request = "\n".join(
            [
                method.upper(),
                canonical_uri,
                canonical_query(parsed.query),
                canonical_headers,
                signed_names,
                payload_hash,
            ]
        )
        credential_scope = f"{short_date}/{REGION}/{SERVICE}/request"
        string_to_sign = "\n".join(
            [
                "HMAC-SHA256",
                x_date,
                credential_scope,
                sha256_hex(canonical_request.encode("utf-8")),
            ]
        )
        k_date = hmac_sha256(self.secret_access_key.encode("utf-8"), short_date.encode())
        k_region = hmac_sha256(k_date, REGION.encode())
        k_service = hmac_sha256(k_region, SERVICE.encode())
        k_signing = hmac_sha256(k_service, b"request")
        signature = hmac_sha256(k_signing, string_to_sign.encode()).hex()
        authorization = (
            f"HMAC-SHA256 Credential={self.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_names}, Signature={signature}"
        )
        return {
            "Authorization": authorization,
            "Content-Type": "application/json",
            "Host": self.host,
            "X-Date": x_date,
            "X-Content-Sha256": payload_hash,
        }

    def action(
        self,
        label: str,
        action: str,
        body: dict[str, Any],
        *,
        auth: str = "bearer",
        version: str = VERSION,
        expected_status: Iterable[int] = (200,),
        expect_success: bool = True,
        required: bool = True,
    ) -> tuple[int, JSON | None, bytes, bool]:
        path = "/?" + urllib.parse.urlencode({"Action": action, "Version": version})
        raw = compact_json(body).encode("utf-8")
        if auth == "aksk":
            headers = self.signed_headers("POST", path, raw)
        else:
            headers = self.api_key_headers(auth)
        return self.request(
            label,
            "POST",
            path,
            headers=headers,
            json_body=body,
            has_json_body=True,
            expected_status=expected_status,
            validator=native_success_validator(action) if expect_success else None,
            required=required,
        )

    @staticmethod
    def result(payload: JSON | None) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        result = payload.get("Result")
        return result if isinstance(result, dict) else None

    def setup_credentials(self) -> bool:
        admin_email = os.environ.get("AETHER_ADMIN_EMAIL", "").strip()
        admin_password = os.environ.get("AETHER_ADMIN_PASSWORD", "")
        if not admin_email or not admin_password:
            self.add_check(
                "管理员环境变量",
                False,
                "AETHER_ADMIN_EMAIL or AETHER_ADMIN_PASSWORD is missing",
            )
            return False
        login_body = {
            "email": admin_email,
            "password": admin_password,
            "auth_type": "local",
        }
        _, payload, _, passed = self.request(
            "管理员登录",
            "POST",
            "/api/auth/login",
            headers={"X-Client-Device-Id": self.device_id},
            json_body=login_body,
            has_json_body=True,
            expected_status=(200,),
            validator=object_validator(("access_token", "user_id")),
        )
        if not passed or not isinstance(payload, dict):
            return False
        self.jwt = str(payload.get("access_token") or "")
        self.user_id = str(payload.get("user_id") or "")

        _, payload, _, passed = self.request(
            "创建一次性 Aether API Key",
            "POST",
            "/api/users/me/api-keys",
            headers=self.jwt_headers(),
            json_body={
                "name": f"{self.name_prefix}-api-key",
                "credential_type": "api_key",
            },
            has_json_body=True,
            expected_status=(200, 201),
            validator=object_validator(("id", "key")),
        )
        if passed and isinstance(payload, dict):
            self.api_key_id = str(payload.get("id") or "")
            self.api_key = str(payload.get("key") or "")

        _, payload, _, passed = self.request(
            "创建一次性 Aether AK/SK",
            "POST",
            "/api/users/me/api-keys",
            headers=self.jwt_headers(),
            json_body={
                "name": f"{self.name_prefix}-aksk",
                "credential_type": "volc_aksk",
            },
            has_json_body=True,
            expected_status=(200, 201),
            validator=object_validator(("id", "access_key_id", "secret_access_key")),
        )
        if passed and isinstance(payload, dict):
            self.aksk_id = str(payload.get("id") or "")
            self.access_key_id = str(payload.get("access_key_id") or "")
            self.secret_access_key = str(payload.get("secret_access_key") or "")

        ready = bool(
            self.jwt
            and self.user_id
            and self.api_key
            and self.access_key_id
            and self.secret_access_key
        )
        self.add_check(
            "一次性测试凭据就绪",
            ready,
            "JWT, API Key and AK/SK created" if ready else "one or more credentials missing",
        )
        return ready

    def provider_probe(self) -> bool:
        def validate(_status: int, payload: JSON | None) -> str | None:
            if not isinstance(payload, dict) or payload.get("success") is not True:
                return "provider probe did not succeed"
            if payload.get("key_id") != self.args.provider_key_id:
                return "provider probe returned an unexpected key_id"
            if payload.get("endpoint_id") != self.args.endpoint_id:
                return "provider probe returned an unexpected endpoint_id"
            return None

        _, _, _, passed = self.request(
            "Provider 素材库基础连通性",
            "POST",
            f"/api/admin/endpoints/keys/{urllib.parse.quote(self.args.provider_key_id, safe='')}/asset-library/test",
            headers=self.jwt_headers(),
            json_body={"endpoint_id": self.args.endpoint_id},
            has_json_body=True,
            expected_status=(200,),
            validator=validate,
        )
        return passed

    def auth_and_protocol_smoke(self) -> None:
        list_body = {
            "Filter": {"GroupType": "AIGC"},
            "PageNumber": 1,
            "PageSize": 10,
            "SortBy": "CreateTime",
            "SortOrder": "Desc",
            "ProjectName": "default",
        }
        for carrier, title in (
            ("bearer", "Authorization Bearer"),
            ("x-api-key", "X-Api-Key"),
            ("api-key", "Api-Key"),
            ("aksk", "Aether AK/SK"),
        ):
            self.action(
                f"鉴权载体：{title}",
                "ListAssetGroups",
                list_body,
                auth=carrier,
            )

        self.request(
            "兼容入口：Action 位于路径",
            "POST",
            "/v3/asset-library/ListAssetGroups",
            headers=self.api_key_headers("bearer"),
            json_body=list_body,
            has_json_body=True,
            expected_status=(200,),
            validator=native_success_validator("ListAssetGroups"),
        )
        self.request(
            "兼容入口：Action 位于 JSON body",
            "POST",
            "/v3/asset-library",
            headers=self.api_key_headers("bearer"),
            json_body={"Action": "ListAssetGroups", "Version": VERSION, **list_body},
            has_json_body=True,
            expected_status=(200,),
            validator=native_success_validator("ListAssetGroups"),
        )

        self.action(
            "错误协议：Version 不匹配",
            "ListAssetGroups",
            list_body,
            version="2023-01-01",
            expected_status=(400,),
            expect_success=False,
        )
        self.action(
            "错误协议：未知 Action",
            "UnknownAssetAction",
            {},
            expected_status=(400,),
            expect_success=False,
        )
        conflicting = self.api_key_headers("bearer")
        conflicting["X-Api-Key"] = "sk-conflicting-test-credential"
        self.request(
            "错误鉴权：冲突凭据",
            "POST",
            "/?Action=ListAssetGroups&Version=2024-01-01",
            headers=conflicting,
            json_body=list_body,
            has_json_body=True,
            expected_status=(400,),
        )
        self.request(
            "错误方法：官方 Action 不接受 GET",
            "GET",
            "/?Action=ListAssetGroups&Version=2024-01-01",
            headers=self.api_key_headers("bearer"),
            expected_status=(404, 405),
        )

    def create_optional_group(self) -> None:
        _, payload, _, passed = self.action(
            "CreateAssetGroup：GroupType 省略",
            "CreateAssetGroup",
            {
                "Name": f"{self.name_prefix}-optional-type",
                "Description": "official optional GroupType E2E",
                "ProjectName": "default",
            },
            auth="x-api-key",
        )
        result = self.result(payload)
        group_id = str(result.get("Id") or "") if result else ""
        valid = passed and official_group_id(group_id)
        self.add_check(
            "GroupType 非必填",
            valid,
            f"created {group_id}" if valid else "upstream rejected or returned no group-* ID",
        )
        if valid:
            self.created_group_ids.add(group_id)
            _, _, _, deleted = self.action(
                "DeleteAssetGroup：清理省略 GroupType 测试组",
                "DeleteAssetGroup",
                {"Id": group_id, "ProjectName": "default"},
                auth="x-api-key",
            )
            if deleted:
                self.created_group_ids.discard(group_id)

    def create_main_group(self) -> str | None:
        _, payload, _, passed = self.action(
            "CreateAssetGroup：创建主测试组",
            "CreateAssetGroup",
            {
                "Name": f"{self.name_prefix}-group",
                "Description": "Aether local material-assets full E2E",
                "GroupType": "AIGC",
                "ProjectName": "default",
            },
        )
        result = self.result(payload)
        group_id = str(result.get("Id") or "") if result else ""
        valid = passed and official_group_id(group_id)
        self.add_check(
            "主测试组返回官方 ID",
            valid,
            group_id or "no group ID",
        )
        if not valid:
            return None
        self.created_group_ids.add(group_id)
        return group_id

    def group_lifecycle(self, group_id: str) -> None:
        self.action(
            "GetAssetGroup：查询主测试组",
            "GetAssetGroup",
            {"Id": group_id, "ProjectName": "default"},
        )
        self.action(
            "ListAssetGroups：按 ID/名称过滤",
            "ListAssetGroups",
            {
                "Filter": {
                    "GroupIds": [group_id],
                    "GroupType": "AIGC",
                    "Name": self.name_prefix,
                },
                "PageNumber": 1,
                "PageSize": 10,
                "SortBy": "CreateTime",
                "SortOrder": "Desc",
                "ProjectName": "default",
            },
        )
        self.action(
            "UpdateAssetGroup：更新名称和描述",
            "UpdateAssetGroup",
            {
                "Id": group_id,
                "Name": f"{self.name_prefix}-group-native-updated",
                "Description": "updated through native Action",
                "ProjectName": "default",
            },
        )
        self.action(
            "GetAssetGroup：验证原生更新",
            "GetAssetGroup",
            {"Id": group_id, "ProjectName": "default"},
        )
        encoded = urllib.parse.quote(group_id, safe="")
        self.request(
            "REST GET group",
            "GET",
            f"/api/material-assets/groups/{encoded}",
            headers=self.api_key_headers(),
            expected_status=(200,),
            validator=object_validator(("id", "name", "group_type", "asset_count")),
        )
        self.request(
            "REST PATCH group",
            "PATCH",
            f"/api/material-assets/groups/{encoded}",
            headers=self.api_key_headers(),
            json_body={"name": f"{self.name_prefix}-group-rest-updated"},
            has_json_body=True,
            expected_status=(200,),
            validator=object_validator(("id", "name")),
        )
        self.request(
            "Admin REST GET group with owner",
            "GET",
            f"/api/admin/material-assets/groups/{encoded}?"
            + urllib.parse.urlencode({"user_id": self.user_id or ""}),
            headers=self.jwt_headers(),
            expected_status=(200,),
            validator=object_validator(("id", "name", "group_type", "asset_count")),
        )
        self.action(
            "GetAssetGroup：验证 REST 更新",
            "GetAssetGroup",
            {"Id": group_id, "ProjectName": "default"},
            auth="aksk",
        )

    def rest_group_routes(self) -> None:
        self.request(
            "REST GET groups list",
            "GET",
            "/api/material-assets/groups?"
            + urllib.parse.urlencode({"group_type": "AIGC", "search": self.name_prefix}),
            headers=self.api_key_headers(),
            expected_status=(200,),
            validator=object_validator(("items", "total")),
        )

        _, payload, _, passed = self.request(
            "REST POST group",
            "POST",
            "/api/material-assets/groups",
            headers=self.api_key_headers(),
            json_body={
                "name": f"{self.name_prefix}-rest-group",
                "description": "user REST group coverage",
                "project_name": "default",
            },
            has_json_body=True,
            expected_status=(201,),
            validator=object_validator(("id", "name", "group_type")),
        )
        group_id = str(payload.get("id") or "") if isinstance(payload, dict) else ""
        valid = passed and official_group_id(group_id)
        self.add_check("REST 用户创建返回官方 group-* ID", valid, group_id or "no group ID")
        if valid:
            self.created_group_ids.add(group_id)
            _, _, _, deleted = self.request(
                "REST DELETE group",
                "DELETE",
                "/api/material-assets/groups/" + urllib.parse.quote(group_id, safe=""),
                headers=self.api_key_headers(),
                expected_status=(204,),
            )
            if deleted:
                self.created_group_ids.discard(group_id)

        user_query = urllib.parse.urlencode({"user_id": self.user_id or ""})
        self.request(
            "Admin REST GET groups list",
            "GET",
            f"/api/admin/material-assets/groups?{user_query}",
            headers=self.jwt_headers(),
            expected_status=(200,),
            validator=object_validator(("items", "total")),
        )
        _, payload, _, passed = self.request(
            "Admin REST POST group",
            "POST",
            "/api/admin/material-assets/groups",
            headers=self.jwt_headers(),
            json_body={
                "user_id": self.user_id,
                "name": f"{self.name_prefix}-admin-rest-group",
                "description": "admin REST group coverage",
                "group_type": "AIGC",
                "project_name": "default",
            },
            has_json_body=True,
            expected_status=(201,),
            validator=object_validator(("id", "name", "group_type", "status")),
        )
        admin_group_id = str(payload.get("id") or "") if isinstance(payload, dict) else ""
        valid = passed and official_group_id(admin_group_id)
        self.add_check(
            "REST 管理员创建返回官方 group-* ID",
            valid,
            admin_group_id or "no group ID",
        )
        if not valid:
            return
        self.created_group_ids.add(admin_group_id)
        encoded = urllib.parse.quote(admin_group_id, safe="")
        self.request(
            "Admin REST PATCH group",
            "PATCH",
            f"/api/admin/material-assets/groups/{encoded}",
            headers=self.jwt_headers(),
            json_body={
                "user_id": self.user_id,
                "name": f"{self.name_prefix}-admin-rest-updated",
            },
            has_json_body=True,
            expected_status=(200,),
            validator=object_validator(("id", "name", "group_type", "status")),
        )
        _, _, _, deleted = self.request(
            "Admin REST DELETE group",
            "DELETE",
            f"/api/admin/material-assets/groups/{encoded}?{user_query}",
            headers=self.jwt_headers(),
            expected_status=(204,),
        )
        if deleted:
            self.created_group_ids.discard(admin_group_id)

    def negative_asset_validation(self, group_id: str) -> None:
        self.action(
            "CreateAsset 错误：不支持的 AssetType",
            "CreateAsset",
            {
                "GroupId": group_id,
                "URL": self.args.image_url,
                "Name": f"{self.name_prefix}-invalid-type",
                "AssetType": "Document",
                "ProjectName": "default",
            },
            expected_status=(400,),
            expect_success=False,
        )
        self.action(
            "CreateAsset 错误：私网 URL",
            "CreateAsset",
            {
                "GroupId": group_id,
                "URL": "https://127.0.0.1/private.png",
                "Name": f"{self.name_prefix}-private-url",
                "AssetType": "Image",
                "ProjectName": "default",
            },
            expected_status=(400,),
            expect_success=False,
        )

    def create_native_assets(self, group_id: str) -> dict[str, str]:
        created: dict[str, str] = {}
        cases = (
            ("Image", self.args.image_url),
        )
        for case_index, (asset_type, source_url) in enumerate(cases):
            if case_index and self.args.create_interval:
                time.sleep(self.args.create_interval)
            payload: JSON | None = None
            passed = False
            for attempt in range(1, self.args.create_attempts + 1):
                status, payload, _, passed = self.action(
                    f"CreateAsset：{asset_type} #{attempt}",
                    "CreateAsset",
                    {
                        "GroupId": group_id,
                        "URL": source_url,
                        "Name": f"{self.name_prefix}-{asset_type.lower()}",
                        "AssetType": asset_type,
                        "ProjectName": "default",
                    },
                    required=False,
                )
                if passed or status != 429:
                    break
                if attempt < self.args.create_attempts:
                    time.sleep(self.args.create_retry_delay)
            result = self.result(payload)
            asset_id = str(result.get("Id") or "") if result else ""
            valid = passed and official_asset_id(asset_id)
            self.add_check(
                f"{asset_type} 返回官方 asset-* ID",
                valid,
                asset_id or "no asset ID",
            )
            if valid:
                created[asset_type] = asset_id
                self.created_asset_ids.add(asset_id)
        return created

    def sequential_video_audio_lifecycle(self, group_id: str) -> None:
        for asset_type, source_url in (
            ("Video", self.args.video_url),
            ("Audio", self.args.audio_url),
        ):
            if self.args.create_interval:
                time.sleep(self.args.create_interval)
            payload: JSON | None = None
            passed = False
            for attempt in range(1, self.args.create_attempts + 1):
                status, payload, _, passed = self.action(
                    f"CreateAsset：{asset_type} #{attempt}",
                    "CreateAsset",
                    {
                        "GroupId": group_id,
                        "URL": source_url,
                        "Name": f"{self.name_prefix}-{asset_type.lower()}",
                        "AssetType": asset_type,
                        "ProjectName": "default",
                    },
                    required=False,
                )
                if passed or status != 429:
                    break
                if attempt < self.args.create_attempts:
                    time.sleep(self.args.create_retry_delay)
            result = self.result(payload)
            asset_id = str(result.get("Id") or "") if result else ""
            valid = passed and official_asset_id(asset_id)
            self.add_check(
                f"{asset_type} 返回官方 asset-* ID",
                valid,
                asset_id or "no asset ID",
            )
            if not valid:
                continue
            self.created_asset_ids.add(asset_id)
            self.poll_native_assets({asset_type: asset_id})
            self.action(
                f"UpdateAsset：{asset_type}",
                "UpdateAsset",
                {
                    "Id": asset_id,
                    "Name": f"{self.name_prefix}-{asset_type.lower()}-updated",
                    "ProjectName": "default",
                },
            )
            _, _, _, deleted = self.action(
                f"DeleteAsset：{asset_type}",
                "DeleteAsset",
                {"Id": asset_id, "ProjectName": "default"},
            )
            if deleted:
                self.created_asset_ids.discard(asset_id)

    def poll_native_assets(self, created: dict[str, str]) -> None:
        pending = dict(created)
        terminal: dict[str, str] = {}
        for attempt in range(1, self.args.poll_attempts + 1):
            for asset_type, asset_id in list(pending.items()):
                _, payload, _, passed = self.action(
                    f"GetAsset 轮询 {asset_type} #{attempt}",
                    "GetAsset",
                    {"Id": asset_id, "ProjectName": "default"},
                    required=True,
                )
                result = self.result(payload)
                status = str(result.get("Status") or "") if result else ""
                if not passed:
                    terminal[asset_type] = "QueryError"
                    self.asset_statuses[asset_id] = "QueryError"
                    pending.pop(asset_type, None)
                    continue
                if passed and status in {"Active", "Failed"}:
                    terminal[asset_type] = status
                    self.asset_statuses[asset_id] = status
                    pending.pop(asset_type, None)
            if not pending:
                break
            if attempt < self.args.poll_attempts:
                time.sleep(self.args.poll_interval)
        for asset_type, asset_id in created.items():
            status = terminal.get(asset_type) or self.asset_statuses.get(asset_id) or "Processing"
            self.add_check(
                f"{asset_type} 异步入库为 Active",
                status == "Active",
                f"asset_id={asset_id}, final_status={status}",
            )

    def native_asset_reads_and_updates(self, group_id: str, created: dict[str, str]) -> None:
        self.action(
            "ListAssets：按组/状态/名称过滤",
            "ListAssets",
            {
                "Filter": {
                    "GroupIds": [group_id],
                    "GroupType": "AIGC",
                    "Statuses": ["Active", "Processing", "Failed"],
                    "Name": self.name_prefix,
                },
                "PageNumber": 1,
                "PageSize": 100,
                "SortBy": "CreateTime",
                "SortOrder": "Desc",
                "ProjectName": "default",
            },
        )
        for asset_type, asset_id in created.items():
            self.action(
                f"UpdateAsset：{asset_type}",
                "UpdateAsset",
                {
                    "Id": asset_id,
                    "Name": f"{self.name_prefix}-{asset_type.lower()}-updated",
                    "ProjectName": "default",
                },
            )
            self.action(
                f"GetAsset：验证 {asset_type} 更新",
                "GetAsset",
                {"Id": asset_id, "ProjectName": "default"},
            )
        self.request(
            "REST GET assets list",
            "GET",
            "/api/material-assets/assets?"
            + urllib.parse.urlencode(
                {"group_id": group_id, "search": self.name_prefix, "page": 1, "page_size": 100}
            ),
            headers=self.api_key_headers(),
            expected_status=(200,),
            validator=object_validator(("items", "total", "page", "page_size")),
        )

    def rest_asset_lifecycle(self, group_id: str) -> None:
        payload: JSON | None = None
        passed = False
        for attempt in range(1, self.args.create_attempts + 1):
            status, payload, _, passed = self.request(
                f"REST POST asset from URL #{attempt}",
                "POST",
                "/api/material-assets/assets/url",
                headers=self.api_key_headers(),
                json_body={
                    "group_id": group_id,
                    "url": self.args.image_url,
                    "name": f"{self.name_prefix}-rest-image",
                    "asset_type": "Image",
                },
                has_json_body=True,
                expected_status=(201,),
                validator=object_validator(("id", "group_id", "asset_type", "status")),
                required=False,
            )
            if passed or status != 429:
                break
            if attempt < self.args.create_attempts:
                time.sleep(self.args.create_retry_delay)
        asset_id = str(payload.get("id") or "") if isinstance(payload, dict) else ""
        valid = passed and official_asset_id(asset_id)
        self.add_check(
            "REST 创建返回官方 asset-* ID",
            valid,
            asset_id or "no asset ID",
        )
        if not valid:
            return
        self.created_asset_ids.add(asset_id)
        encoded = urllib.parse.quote(asset_id, safe="")
        for attempt in range(1, min(self.args.poll_attempts, 4) + 1):
            _, item, _, item_passed = self.request(
                f"REST GET asset 轮询 #{attempt}",
                "GET",
                f"/api/material-assets/assets/{encoded}",
                headers=self.api_key_headers(),
                expected_status=(200,),
                validator=object_validator(("id", "status", "asset_type", "group_id")),
            )
            status = str(item.get("status") or "") if isinstance(item, dict) else ""
            if not item_passed:
                break
            if item_passed and status == "Active":
                break
            if status == "Failed":
                break
            if attempt < min(self.args.poll_attempts, 4):
                time.sleep(self.args.poll_interval)
        self.request(
            "REST PATCH asset",
            "PATCH",
            f"/api/material-assets/assets/{encoded}",
            headers=self.api_key_headers(),
            json_body={"name": f"{self.name_prefix}-rest-image-updated"},
            has_json_body=True,
            expected_status=(200,),
            validator=object_validator(("id", "name")),
        )
        self.request(
            "REST GET asset preview",
            "GET",
            f"/api/material-assets/assets/{encoded}/preview",
            headers=self.api_key_headers(),
            expected_status=(200,),
            binary_response=True,
        )
        _, _, _, deleted = self.request(
            "REST DELETE asset",
            "DELETE",
            f"/api/material-assets/assets/{encoded}",
            headers=self.api_key_headers(),
            expected_status=(204,),
        )
        if deleted:
            self.created_asset_ids.discard(asset_id)

    def rest_upload_rejection(self, group_id: str) -> None:
        self.request(
            "REST POST upload：官方仅支持 URL",
            "POST",
            "/api/material-assets/assets/upload",
            headers=self.api_key_headers(),
            json_body={"group_id": group_id},
            has_json_body=True,
            expected_status=(501,),
        )

    def admin_rest_asset_routes(self, group_id: str) -> None:
        query = urllib.parse.urlencode(
            {"user_id": self.user_id or "", "group_id": group_id, "page": 1, "page_size": 100}
        )
        self.request(
            "Admin REST GET assets list",
            "GET",
            f"/api/admin/material-assets/assets?{query}",
            headers=self.jwt_headers(),
            expected_status=(200,),
            validator=object_validator(("items", "total", "page", "page_size")),
        )

        payload: JSON | None = None
        passed = False
        for attempt in range(1, self.args.create_attempts + 1):
            status, payload, _, passed = self.request(
                f"Admin REST POST asset from URL #{attempt}",
                "POST",
                "/api/admin/material-assets/assets/url",
                headers=self.jwt_headers(),
                json_body={
                    "user_id": self.user_id,
                    "group_id": group_id,
                    "url": self.args.image_url,
                    "name": f"{self.name_prefix}-admin-rest-image",
                    "asset_type": "Image",
                },
                has_json_body=True,
                expected_status=(201,),
                validator=object_validator(
                    ("id", "group_id", "asset_type", "status", "provider_id")
                ),
                required=False,
            )
            if passed or status != 429:
                break
            if attempt < self.args.create_attempts:
                time.sleep(self.args.create_retry_delay)
        asset_id = str(payload.get("id") or "") if isinstance(payload, dict) else ""
        valid = passed and official_asset_id(asset_id)
        self.add_check(
            "REST 管理员创建返回官方 asset-* ID",
            valid,
            asset_id or "no asset ID",
        )
        if valid:
            self.created_asset_ids.add(asset_id)
            encoded = urllib.parse.quote(asset_id, safe="")
            user_query = urllib.parse.urlencode({"user_id": self.user_id or ""})
            self.request(
                "Admin REST GET asset",
                "GET",
                f"/api/admin/material-assets/assets/{encoded}?{user_query}",
                headers=self.jwt_headers(),
                expected_status=(200,),
                validator=object_validator(
                    ("id", "group_id", "asset_type", "status", "provider_id")
                ),
            )
            self.request(
                "Admin REST PATCH asset",
                "PATCH",
                f"/api/admin/material-assets/assets/{encoded}",
                headers=self.jwt_headers(),
                json_body={
                    "user_id": self.user_id,
                    "name": f"{self.name_prefix}-admin-rest-image-updated",
                },
                has_json_body=True,
                expected_status=(200,),
                validator=object_validator(("id", "name", "provider_id")),
            )
            self.request(
                "Admin REST GET asset preview",
                "GET",
                f"/api/admin/material-assets/assets/{encoded}/preview?{user_query}",
                headers=self.jwt_headers(),
                expected_status=(200,),
                binary_response=True,
            )
            _, _, _, deleted = self.request(
                "Admin REST DELETE asset",
                "DELETE",
                f"/api/admin/material-assets/assets/{encoded}?{user_query}",
                headers=self.jwt_headers(),
                expected_status=(204,),
            )
            if deleted:
                self.created_asset_ids.discard(asset_id)

        self.request(
            "Admin REST POST upload：官方仅支持 URL",
            "POST",
            "/api/admin/material-assets/assets/upload",
            headers=self.jwt_headers(),
            json_body={"user_id": self.user_id, "group_id": group_id},
            has_json_body=True,
            expected_status=(501,),
        )

    def verification_actions(self) -> None:
        _, payload, _, passed = self.action(
            "CreateVisualValidateSession",
            "CreateVisualValidateSession",
            {"CallbackURL": self.args.callback_url, "ProjectName": "default"},
        )
        result = self.result(payload)
        token = str(result.get("BytedToken") or "") if result else ""
        self.add_check(
            "真人验证返回 BytedToken",
            passed and bool(token),
            mask_identifier(token) if token else "no BytedToken",
        )
        if token:
            self.native_verification_sessions_created += 1
            self._write_reports()
            self.action(
                "GetVisualValidateResult：未人工完成路径",
                "GetVisualValidateResult",
                {"BytedToken": token, "ProjectName": "default"},
            )
        else:
            self.skip("GetVisualValidateResult", "CreateVisualValidateSession returned no token")

        _, rest_payload, _, rest_passed = self.request(
            "REST POST verification session",
            "POST",
            "/api/material-assets/verification-sessions",
            headers=self.api_key_headers(),
            json_body={"callback_url": self.args.callback_url},
            has_json_body=True,
            expected_status=(201,),
            validator=object_validator(("id", "status")),
        )
        session_id = (
            str(rest_payload.get("id") or "") if isinstance(rest_payload, dict) else ""
        )
        if rest_passed and session_id:
            self.verification_session_ids.append(session_id)
            self.request(
                "REST GET verification session",
                "GET",
                "/api/material-assets/verification-sessions/"
                + urllib.parse.quote(session_id, safe=""),
                headers=self.api_key_headers(),
                expected_status=(200,),
                validator=object_validator(("id", "status")),
            )

        _, admin_payload, _, admin_passed = self.request(
            "Admin REST POST verification session",
            "POST",
            "/api/admin/material-assets/verification-sessions",
            headers=self.jwt_headers(),
            json_body={
                "user_id": self.user_id,
                "callback_url": self.args.callback_url,
                "project_name": "default",
            },
            has_json_body=True,
            expected_status=(201,),
            validator=object_validator(("id", "status")),
        )
        admin_session_id = (
            str(admin_payload.get("id") or "") if isinstance(admin_payload, dict) else ""
        )
        if admin_passed and admin_session_id:
            self.verification_session_ids.append(admin_session_id)
            self.request(
                "Admin REST GET verification session",
                "GET",
                "/api/admin/material-assets/verification-sessions/"
                + urllib.parse.quote(admin_session_id, safe="")
                + "?"
                + urllib.parse.urlencode({"user_id": self.user_id or ""}),
                headers=self.jwt_headers(),
                expected_status=(200,),
                validator=object_validator(("id", "status")),
            )

    def delete_native_assets(self) -> None:
        for asset_id in sorted(list(self.created_asset_ids)):
            _, _, _, deleted = self.action(
                f"DeleteAsset：{asset_id}",
                "DeleteAsset",
                {"Id": asset_id, "ProjectName": "default"},
            )
            if deleted:
                self.created_asset_ids.discard(asset_id)

    def close_main_group(self, group_id: str) -> None:
        self.action(
            "ListAssets：删除后复验",
            "ListAssets",
            {
                "Filter": {"GroupIds": [group_id], "GroupType": "AIGC"},
                "PageNumber": 1,
                "PageSize": 100,
                "ProjectName": "default",
            },
        )
        _, _, _, deleted = self.action(
            "DeleteAssetGroup：删除主测试组",
            "DeleteAssetGroup",
            {"Id": group_id, "ProjectName": "default"},
        )
        if deleted:
            self.created_group_ids.discard(group_id)
        self.action(
            "ListAssetGroups：删除组后复验",
            "ListAssetGroups",
            {
                "Filter": {"GroupIds": [group_id], "GroupType": "AIGC"},
                "PageNumber": 1,
                "PageSize": 10,
                "ProjectName": "default",
            },
        )

    def cleanup(self) -> None:
        print("[CLEANUP] cleaning resources created by this run", flush=True)
        if self.api_key:
            for asset_id in sorted(list(self.created_asset_ids)):
                try:
                    _, _, _, deleted = self.action(
                        f"Cleanup DeleteAsset：{asset_id}",
                        "DeleteAsset",
                        {"Id": asset_id, "ProjectName": "default"},
                        required=False,
                    )
                    if deleted:
                        self.created_asset_ids.discard(asset_id)
                except Exception as error:  # noqa: BLE001
                    self.skip(f"Cleanup asset {asset_id}", str(error))
            for group_id in sorted(list(self.created_group_ids)):
                try:
                    _, _, _, deleted = self.action(
                        f"Cleanup DeleteAssetGroup：{group_id}",
                        "DeleteAssetGroup",
                        {"Id": group_id, "ProjectName": "default"},
                        required=False,
                    )
                    if deleted:
                        self.created_group_ids.discard(group_id)
                except Exception as error:  # noqa: BLE001
                    self.skip(f"Cleanup group {group_id}", str(error))
        if self.jwt:
            for credential_id, label in (
                (self.aksk_id, "删除一次性 Aether AK/SK"),
                (self.api_key_id, "删除一次性 Aether API Key"),
            ):
                if not credential_id:
                    continue
                self.request(
                    label,
                    "DELETE",
                    "/api/users/me/api-keys/"
                    + urllib.parse.quote(credential_id, safe=""),
                    headers=self.jwt_headers(),
                    expected_status=(200, 204),
                    required=False,
                )
        self._write_reports()

    def run(self) -> int:
        try:
            self.request(
                "Gateway health",
                "GET",
                "/_gateway/health",
                expected_status=(200,),
                validator=lambda _status, payload: (
                    None
                    if isinstance(payload, dict) and payload.get("status") == "ok"
                    else "gateway health payload is not ok"
                ),
            )
            if not self.setup_credentials():
                return 1
            if not self.provider_probe():
                self.skip(
                    "素材库业务用例",
                    "指定 provider key/endpoint 连通性测试失败，停止以避免误测其他上游",
                )
                return 1
            self.auth_and_protocol_smoke()
            self.create_optional_group()
            main_group_id = self.create_main_group()
            if main_group_id:
                self.group_lifecycle(main_group_id)
                self.rest_group_routes()
                self.negative_asset_validation(main_group_id)
                created = self.create_native_assets(main_group_id)
                self.poll_native_assets(created)
                self.native_asset_reads_and_updates(main_group_id, created)
                self.rest_asset_lifecycle(main_group_id)
                self.rest_upload_rejection(main_group_id)
                self.delete_native_assets()
                self.sequential_video_audio_lifecycle(main_group_id)
                self.admin_rest_asset_routes(main_group_id)
                self.verification_actions()
                self.close_main_group(main_group_id)
            else:
                for label in (
                    "group lifecycle",
                    "Image/Video/Audio lifecycle",
                    "REST asset lifecycle",
                ):
                    self.skip(label, "main CreateAssetGroup failed")
                self.verification_actions()
        finally:
            self.cleanup()
            print(f"JSON report: {self.json_report}", flush=True)
            print(f"Markdown report: {self.markdown_report}", flush=True)
        required_request_failures = sum(
            1 for record in self.records if record["required"] and not record["passed"]
        )
        required_check_failures = sum(
            1 for check in self.checks if check["required"] and not check["passed"]
        )
        return 1 if required_request_failures or required_check_failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8084")
    parser.add_argument(
        "--provider-key-id",
        default=os.environ.get(
            "AETHER_MATERIAL_ASSET_PROVIDER_KEY_ID",
            "9f9a391e-432c-477d-a912-9bfe20e71293",
        ),
    )
    parser.add_argument(
        "--endpoint-id",
        default=os.environ.get(
            "AETHER_MATERIAL_ASSET_ENDPOINT_ID",
            "63ad8f82-961e-4961-8cd2-434434af840a",
        ),
    )
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--audio-url", default=DEFAULT_AUDIO_URL)
    parser.add_argument("--callback-url", default=DEFAULT_CALLBACK_URL)
    parser.add_argument("--poll-attempts", type=int, default=8)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--create-attempts", type=int, default=3)
    parser.add_argument("--create-interval", type=float, default=5.0)
    parser.add_argument("--create-retry-delay", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--report-dir")
    args = parser.parse_args()
    if args.poll_attempts < 1:
        parser.error("--poll-attempts must be >= 1")
    if args.poll_interval < 0:
        parser.error("--poll-interval must be >= 0")
    if args.create_attempts < 1:
        parser.error("--create-attempts must be >= 1")
    if args.create_interval < 0:
        parser.error("--create-interval must be >= 0")
    if args.create_retry_delay < 0:
        parser.error("--create-retry-delay must be >= 0")
    return args


def main() -> int:
    try:
        harness = Harness(parse_args())
        return harness.run()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001
        print(f"fatal: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
