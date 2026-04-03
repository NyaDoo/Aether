from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.requests import ClientDisconnect

from src.config.settings import config
from src.core.enums import UserRole
from src.core.exceptions import BalanceInsufficientException
from src.core.logger import logger
from src.database.database import create_session
from src.models.database import ApiKey, AuditEventType, User
from src.services.auth.service import AuthService
from src.services.auth.session_service import SessionService
from src.services.rate_limit.user_rpm_limiter import SYSTEM_RPM_CONFIG_KEY, get_user_rpm_limiter
from src.services.system.audit import AuditService
from src.services.system.config import SystemConfigService
from src.services.usage.service import UsageService
from src.services.wallet import WalletService
from src.utils.perf import PerfRecorder
from src.utils.request_utils import get_request_identity_metadata, update_request_state

if TYPE_CHECKING:
    from src.models.database import ManagementToken

from .adapter import ApiAdapter, ApiMode
from .context import ApiRequestContext

# 高频轮询端点，抑制其 debug 日志以减少噪音
QUIET_POLLING_PATHS: set[str] = {
    "/api/admin/usage/active",
    "/api/admin/usage/records",
    "/api/admin/usage/stats",
    "/api/admin/usage/aggregation/stats",
    "/api/admin/health/status",
    "/api/wallet/today-cost",
}

TRUSTED_GATEWAY_HEADER = "x-aether-gateway"
TRUSTED_GATEWAY_EXECUTION_PATH_HEADER = "x-aether-execution-path"
TRUSTED_AUTH_USER_ID_HEADER = "x-aether-auth-user-id"
TRUSTED_AUTH_API_KEY_ID_HEADER = "x-aether-auth-api-key-id"
TRUSTED_AUTH_BALANCE_HEADER = "x-aether-auth-balance-remaining"
TRUSTED_AUTH_ACCESS_ALLOWED_HEADER = "x-aether-auth-access-allowed"
TRUSTED_RATE_LIMIT_PREFLIGHT_HEADER = "x-aether-rate-limit-preflight"


@dataclass(slots=True)
class _PipelinePerfState:
    request: Request
    labels: dict[str, str]
    sampled: bool

    @classmethod
    def create(
        cls,
        request: Request,
        *,
        adapter: ApiAdapter,
        mode: ApiMode,
    ) -> _PipelinePerfState:
        labels = {
            "mode": getattr(mode, "value", str(mode)),
            "adapter": adapter.name,
        }
        sampled = PerfRecorder.should_store_sample()
        if sampled:
            setattr(request.state, "perf_sampled", True)
            setattr(
                request.state,
                "perf_metrics",
                {"pipeline": {}, "sample_rate": getattr(config, "perf_store_sample_rate", 1.0)},
            )
        return cls(request=request, labels=labels, sampled=sampled)

    def record_ms(self, key: str, duration: float | None) -> None:
        if duration is None:
            return
        perf_metrics = getattr(self.request.state, "perf_metrics", None)
        if not isinstance(perf_metrics, dict):
            return
        bucket = perf_metrics.setdefault("pipeline", {})
        bucket[key] = int(duration * 1000)


@dataclass(slots=True)
class _PipelineAuthState:
    user: User | None
    api_key: ApiKey | None
    management_token: ManagementToken | None


class ApiRequestPipeline:
    """负责统一执行认证、余额校验、上下文构建等通用逻辑的管道。"""

    def __init__(
        self,
        auth_service: AuthService = AuthService,
        usage_service: UsageService = UsageService,
        audit_service: AuditService = AuditService,
    ):
        self.auth_service = auth_service
        self.usage_service = usage_service
        self.audit_service = audit_service

    def _commit_session_touch(self, db: Session, *, scope: str) -> None:
        """Persist session last_seen updates immediately to avoid holding row locks.

        Admin usage views can execute heavy read queries after authentication.
        If the request later stalls, leaving the session touch inside the request
        transaction can block all subsequent requests that update the same
        `user_sessions` row. Commit the touch in its own short transaction so
        later long-running reads cannot keep the session row locked.
        """
        original_expire_on_commit = getattr(db, "expire_on_commit", None)
        try:
            if original_expire_on_commit is not None:
                db.expire_on_commit = False
            db.commit()
        except Exception as exc:
            try:
                db.rollback()
            except Exception as rollback_exc:
                logger.debug("[Pipeline] {} session touch rollback failed: {}", scope, rollback_exc)
            logger.warning("[Pipeline] failed to persist {} session touch: {}", scope, exc)
        finally:
            if original_expire_on_commit is not None:
                db.expire_on_commit = original_expire_on_commit

    def _touch_authenticated_session(
        self,
        request: Request,
        db: Session,
        *,
        user_id: str,
        session_id: str,
        scope: str,
    ) -> Any:
        """Resolve and touch the authenticated user session via shared request identity."""

        client_device_id = SessionService.extract_client_device_id(request)
        session = SessionService.get_active_session(db, session_id, user_id)
        if not session:
            raise HTTPException(status_code=401, detail="登录会话已失效，请重新登录")

        SessionService.assert_session_device_matches(session, client_device_id)
        identity = get_request_identity_metadata(request)
        session_touched = SessionService.touch_session(
            session,
            client_ip=identity.client_ip,
            user_agent=identity.user_agent,
        )
        if session_touched:
            self._commit_session_touch(db, scope=scope)
        return session

    async def run(
        self,
        adapter: ApiAdapter,
        http_request: Request,
        db: Session,
        *,
        mode: ApiMode = ApiMode.STANDARD,
        api_format_hint: str | None = None,
        path_params: dict[str, Any] | None = None,
    ) -> Any:
        perf_state = _PipelinePerfState.create(http_request, adapter=adapter, mode=mode)
        is_quiet = http_request.url.path in QUIET_POLLING_PATHS
        gateway_execution_path = self._capture_trusted_gateway_execution_path(http_request)
        self._log_pipeline_entry(
            http_request,
            adapter,
            mode,
            gateway_execution_path=gateway_execution_path,
            quiet=is_quiet,
        )

        auth_state = await self._authenticate_request_legacy(
            http_request,
            db,
            adapter,
            mode=mode,
            quiet=is_quiet,
            perf_state=perf_state,
        )
        await self._apply_legacy_request_guards(http_request, db, mode=mode, auth_state=auth_state)

        context_or_response = await self._build_pipeline_context_legacy(
            adapter,
            http_request,
            db,
            mode=mode,
            api_format_hint=api_format_hint,
            path_params=path_params,
            auth_state=auth_state,
            quiet=is_quiet,
            perf_state=perf_state,
        )
        if isinstance(context_or_response, JSONResponse):
            return context_or_response

        return await self._dispatch_adapter_request_legacy(
            adapter,
            context_or_response,
            perf_state=perf_state,
        )

    def _log_pipeline_entry(
        self,
        request: Request,
        adapter: ApiAdapter,
        mode: ApiMode,
        *,
        gateway_execution_path: str | None,
        quiet: bool,
    ) -> None:
        if quiet:
            return
        logger.debug(
            "[Pipeline] {} {} | adapter={}, mode={}, gateway_path={}",
            request.method,
            request.url.path,
            adapter.__class__.__name__,
            mode,
            gateway_execution_path or "-",
        )

    async def _authenticate_request_legacy(
        self,
        request: Request,
        db: Session,
        adapter: ApiAdapter,
        *,
        mode: ApiMode,
        quiet: bool,
        perf_state: _PipelinePerfState,
    ) -> _PipelineAuthState:
        """Legacy Python auth/session entrypoint.

        Rust bridge can currently short-circuit only the client API-key path via
        trusted gateway headers. JWT/session and management-token flows still
        terminate in Python.
        """

        auth_start = PerfRecorder.start(force=perf_state.sampled)
        try:
            if mode == ApiMode.ADMIN:
                user, management_token = await self._authenticate_admin(request, db)
                return _PipelineAuthState(user=user, api_key=None, management_token=management_token)
            if mode == ApiMode.USER:
                user, management_token = await self._authenticate_user(request, db)
                return _PipelineAuthState(user=user, api_key=None, management_token=management_token)
            if mode == ApiMode.PUBLIC:
                return _PipelineAuthState(user=None, api_key=None, management_token=None)
            if mode == ApiMode.MANAGEMENT:
                user, management_token = await self._authenticate_management(request, db)
                return _PipelineAuthState(user=user, api_key=None, management_token=management_token)

            user, api_key = await self._authenticate_client(
                request,
                db,
                adapter,
                quiet=quiet,
            )
            return _PipelineAuthState(user=user, api_key=api_key, management_token=None)
        finally:
            auth_duration = PerfRecorder.stop(
                auth_start,
                "pipeline_auth",
                labels=perf_state.labels,
            )
            perf_state.record_ms("auth_ms", auth_duration)

    async def _apply_legacy_request_guards(
        self,
        request: Request,
        db: Session,
        *,
        mode: ApiMode,
        auth_state: _PipelineAuthState,
    ) -> None:
        """Python-side request guards kept for shared limiter/config coupling."""

        if mode not in {ApiMode.STANDARD, ApiMode.PROXY}:
            return
        if not auth_state.user or not auth_state.api_key:
            return
        if self._trusted_gateway_completed_rate_limit_preflight(request):
            return
        await self._check_user_rate_limit(request, db, auth_state.user, auth_state.api_key)

    async def _build_pipeline_context_legacy(
        self,
        adapter: ApiAdapter,
        request: Request,
        db: Session,
        *,
        mode: ApiMode,
        api_format_hint: str | None,
        path_params: dict[str, Any] | None,
        auth_state: _PipelineAuthState,
        quiet: bool,
        perf_state: _PipelinePerfState,
    ) -> ApiRequestContext | JSONResponse:
        """Legacy Python context boundary.

        Request body reads, FastAPI request.state hydration, and context assembly
        still rely on the Python request object and remain in-process for now.
        """

        raw_body = await self._read_request_body_for_context_legacy(
            request,
            adapter,
            perf_state=perf_state,
        )
        if isinstance(raw_body, JSONResponse):
            return raw_body

        resolved_api_format_hint = self._resolve_api_format_hint(adapter, api_format_hint)
        context_start = PerfRecorder.start(force=perf_state.sampled)
        context = ApiRequestContext.build(
            request=request,
            db=db,
            user=auth_state.user,
            api_key=auth_state.api_key,
            raw_body=raw_body,
            mode=mode.value,
            api_format_hint=resolved_api_format_hint,
            path_params=path_params,
        )
        context_duration = PerfRecorder.stop(
            context_start,
            "pipeline_context_build",
            labels=perf_state.labels,
        )
        perf_state.record_ms("context_build_ms", context_duration)
        await self._apply_context_runtime_state_legacy(
            context,
            mode=mode,
            auth_state=auth_state,
            quiet=quiet,
        )
        return context

    async def _read_request_body_for_context_legacy(
        self,
        request: Request,
        adapter: ApiAdapter,
        *,
        perf_state: _PipelinePerfState,
    ) -> bytes | None | JSONResponse:
        should_eager_read_body = request.method in {"POST", "PUT", "PATCH"} and getattr(
            adapter, "eager_request_body", True
        )
        if not should_eager_read_body:
            return None

        try:
            import asyncio

            body_start = PerfRecorder.start(force=perf_state.sampled)
            body_size = 0
            try:
                raw_body = await asyncio.wait_for(request.body(), timeout=config.request_body_timeout)
                body_size = len(raw_body) if raw_body is not None else 0
                return raw_body
            finally:
                body_duration = PerfRecorder.stop(
                    body_start,
                    "pipeline_body_read",
                    labels=perf_state.labels,
                    log_hint=f"size={body_size}",
                )
                perf_state.record_ms("body_read_ms", body_duration)
                if perf_state.sampled:
                    perf_metrics = getattr(request.state, "perf_metrics", None)
                    if isinstance(perf_metrics, dict):
                        perf_metrics.setdefault("pipeline", {})["body_bytes"] = int(body_size)
        except TimeoutError:
            timeout_sec = int(config.request_body_timeout)
            logger.error("读取请求体超时({}s),可能客户端未发送完整请求体", timeout_sec)
            raise HTTPException(
                status_code=408,
                detail=f"Request timeout: body not received within {timeout_sec} seconds",
            )
        except ClientDisconnect:
            logger.warning(
                "[Pipeline] 客户端在读取请求体期间断开连接: {} {}",
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=499,
                content={"error": "client_disconnected", "message": "Client closed request"},
            )

    async def _apply_context_runtime_state_legacy(
        self,
        context: ApiRequestContext,
        *,
        mode: ApiMode,
        auth_state: _PipelineAuthState,
        quiet: bool,
    ) -> None:
        if auth_state.management_token:
            context.management_token = auth_state.management_token
        context.quiet_logging = quiet

        if mode not in {ApiMode.STANDARD, ApiMode.PROXY, ApiMode.USER} or not auth_state.user:
            return

        if context.prefetched_balance_remaining is not None:
            remaining = context.prefetched_balance_remaining
        else:
            remaining = await self._calculate_balance_remaining_async(
                auth_state.user,
                api_key=auth_state.api_key,
            )
        context.balance_remaining = remaining

    async def _dispatch_adapter_request_legacy(
        self,
        adapter: ApiAdapter,
        context: ApiRequestContext,
        *,
        perf_state: _PipelinePerfState,
    ) -> Any:
        """Legacy Python adapter dispatch/audit boundary."""

        authorize_start = PerfRecorder.start(force=perf_state.sampled)
        try:
            authorize_result = adapter.authorize(context)
            if hasattr(authorize_result, "__await__"):
                await authorize_result
        finally:
            authorize_duration = PerfRecorder.stop(
                authorize_start,
                "pipeline_authorize",
                labels=perf_state.labels,
            )
            perf_state.record_ms("authorize_ms", authorize_duration)

        handle_start = PerfRecorder.start(force=perf_state.sampled)
        try:
            response = await adapter.handle(context)
            handle_duration = PerfRecorder.stop(
                handle_start,
                "pipeline_handle",
                labels=perf_state.labels,
            )
            perf_state.record_ms("handle_ms", handle_duration)
            status_code = getattr(response, "status_code", None)
            self._record_audit_event(context, adapter, success=True, status_code=status_code)
            return response
        except HTTPException as exc:
            handle_duration = PerfRecorder.stop(
                handle_start,
                "pipeline_handle",
                labels=perf_state.labels,
            )
            perf_state.record_ms("handle_ms", handle_duration)
            err_detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            self._record_audit_event(
                context,
                adapter,
                success=False,
                status_code=exc.status_code,
                error=err_detail,
            )
            raise
        except ClientDisconnect:
            handle_duration = PerfRecorder.stop(
                handle_start,
                "pipeline_handle",
                labels=perf_state.labels,
            )
            perf_state.record_ms("handle_ms", handle_duration)
            logger.warning(
                "[Pipeline] 客户端在处理期间断开连接: {} {}",
                context.request_method,
                context.request_path,
            )
            self._record_audit_event(
                context,
                adapter,
                success=False,
                status_code=499,
                error="client_disconnected",
            )
            return JSONResponse(
                status_code=499,
                content={"error": "client_disconnected", "message": "Client closed request"},
            )
        except Exception as exc:
            handle_duration = PerfRecorder.stop(
                handle_start,
                "pipeline_handle",
                labels=perf_state.labels,
            )
            perf_state.record_ms("handle_ms", handle_duration)
            if isinstance(exc, SQLAlchemyError):
                # SQL 执行失败后事务会进入 aborted 状态；先回滚，避免审计写入二次报错。
                try:
                    context.db.rollback()
                except Exception as rollback_exc:
                    logger.debug("[Pipeline] 回滚失败（可忽略）: {}", rollback_exc)
            self._record_audit_event(
                context,
                adapter,
                success=False,
                status_code=500,
                error=str(exc),
            )
            raise

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _resolve_api_format_hint(adapter: ApiAdapter, explicit_hint: str | None) -> str | None:
        normalized_hint = (explicit_hint or "").strip()
        if normalized_hint:
            return normalized_hint

        allowed_api_formats = getattr(adapter, "allowed_api_formats", None)
        if isinstance(allowed_api_formats, (list, tuple)):
            for candidate in allowed_api_formats:
                if isinstance(candidate, str):
                    normalized_candidate = candidate.strip()
                    if normalized_candidate:
                        return normalized_candidate

        adapter_api_format = getattr(adapter, "api_format", None)
        if isinstance(adapter_api_format, str):
            normalized_adapter_api_format = adapter_api_format.strip()
            if normalized_adapter_api_format:
                return normalized_adapter_api_format

        return None

    async def _check_user_rate_limit(
        self,
        request: Request,
        db: Session,
        user: User,
        api_key: ApiKey,
    ) -> None:
        limiter = await get_user_rpm_limiter()
        system_default_raw = SystemConfigService.get_config(db, SYSTEM_RPM_CONFIG_KEY, default=0)
        system_default = max(int(system_default_raw or 0), 0)

        if api_key.is_standalone:
            effective_user_limit = (
                max(int(api_key.rate_limit or 0), 0)
                if api_key.rate_limit is not None
                else system_default
            )
            user_rpm_key = limiter.get_standalone_rpm_key(api_key.id)
            key_rpm_limit = 0
        else:
            effective_user_limit = (
                max(int(user.rate_limit or 0), 0) if user.rate_limit is not None else system_default
            )
            user_rpm_key = limiter.get_user_rpm_key(user.id)
            key_rpm_limit = max(int(api_key.rate_limit or 0), 0)

        result = await limiter.check_and_consume(
            user_rpm_key=user_rpm_key,
            user_rpm_limit=effective_user_limit,
            key_rpm_key=limiter.get_key_rpm_key(api_key.id),
            key_rpm_limit=key_rpm_limit,
        )

        if result.allowed:
            return

        scope = result.scope or "user"
        limit = result.limit or (effective_user_limit if scope == "user" else key_rpm_limit)
        retry_after = result.retry_after or limiter.get_retry_after()

        headers = {
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Scope": scope,
        }
        update_request_state(request, rate_limit_scope=scope)
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试", headers=headers)

    async def _authenticate_client(
        self, request: Request, db: Session, adapter: ApiAdapter, **_kw: object
    ) -> tuple[User, ApiKey]:
        """Bridge entry for API-key auth.

        Trusted gateway headers are the preferred Rust handoff. Once Rust has
        locally completed API-key/balance preflight, the gateway strips the raw
        provider credential before proxying to Python and this method becomes a
        thin ORM reattach shell. Direct client API keys still use the legacy
        Python authenticator as the compatibility fallback.
        """

        trusted_auth = self._authenticate_client_from_gateway_bridge(request, db)
        if trusted_auth is not None:
            return trusted_auth

        return await self._authenticate_client_legacy_fallback(request, db, adapter)

    def _authenticate_client_from_gateway_bridge(
        self,
        request: Request,
        db: Session,
    ) -> tuple[User, ApiKey] | None:
        """Rust bridge shell for trusted gateway auth headers."""

        return self._try_trusted_gateway_auth(request, db)

    async def _authenticate_client_legacy_fallback(
        self,
        request: Request,
        db: Session,
        adapter: ApiAdapter,
    ) -> tuple[User, ApiKey]:
        """Legacy Python fallback for requests without trusted Rust preflight."""

        client_api_key = adapter.extract_api_key(request)
        if not client_api_key:
            raise HTTPException(status_code=401, detail="请提供API密钥")

        auth_result = await self.auth_service.authenticate_api_key_threadsafe(client_api_key)
        if not auth_result:
            raise HTTPException(status_code=401, detail="无效的API密钥")

        user = auth_result.user
        api_key = auth_result.api_key
        if not user or not api_key:
            raise HTTPException(status_code=401, detail="无效的API密钥")

        return self._bind_authenticated_client(
            request,
            db,
            user_id=getattr(user, "id", None),
            api_key_id=getattr(api_key, "id", None),
            balance_remaining=auth_result.balance_remaining,
            access_allowed=auth_result.access_allowed,
            strict=True,
        )

    def _bind_authenticated_client(
        self,
        request: Request,
        db: Session,
        *,
        user_id: str | None,
        api_key_id: str | None,
        balance_remaining: float | None,
        access_allowed: bool,
        strict: bool,
    ) -> tuple[User, ApiKey] | None:
        """Normalize bridge/fallback API-key auth into request-scoped ORM objects.

        Trusted gateway auth uses `strict=False` so stale bridge headers can
        fall back to the legacy Python path. Direct API-key auth uses
        `strict=True` and treats the same invariant checks as terminal.
        """

        def _fail_or_skip() -> None:
            if strict:
                raise HTTPException(status_code=401, detail="无效的API密钥")

        if not user_id or not api_key_id:
            _fail_or_skip()
            return None

        db_user = db.query(User).filter(User.id == user_id).first()
        db_api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
        if not db_user or not db_api_key:
            _fail_or_skip()
            return None

        if not db_user.is_active or db_user.is_deleted:
            _fail_or_skip()
            return None
        if not db_api_key.is_active:
            _fail_or_skip()
            return None
        if db_api_key.is_locked and not db_api_key.is_standalone:
            raise HTTPException(status_code=403, detail="该密钥已被管理员锁定，请联系管理员")
        if db_api_key.user_id != db_user.id:
            _fail_or_skip()
            return None

        if db_api_key.expires_at:
            expires_at = db_api_key.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at < datetime.now(timezone.utc):
                _fail_or_skip()
                return None

        update_request_state(
            request,
            user_id=db_user.id,
            api_key_id=db_api_key.id,
            prefetched_balance_remaining=balance_remaining,
        )

        if not access_allowed:
            raise BalanceInsufficientException(balance_type="USD", remaining=balance_remaining)

        return db_user, db_api_key

    def _try_trusted_gateway_auth(
        self,
        request: Request,
        db: Session,
    ) -> tuple[User, ApiKey] | None:
        """Rust bridge handoff for API-key auth/balance preflight."""

        if not self._is_trusted_gateway_request(request):
            return None
        self._capture_trusted_gateway_execution_path(request)

        user_id = str(request.headers.get(TRUSTED_AUTH_USER_ID_HEADER) or "").strip()
        api_key_id = str(request.headers.get(TRUSTED_AUTH_API_KEY_ID_HEADER) or "").strip()
        if not user_id or not api_key_id:
            return None

        balance_remaining = self._parse_trusted_balance_remaining(
            request.headers.get(TRUSTED_AUTH_BALANCE_HEADER)
        )
        access_allowed = self._parse_trusted_bool_header(
            request.headers.get(TRUSTED_AUTH_ACCESS_ALLOWED_HEADER),
            default=True,
        )

        return self._bind_authenticated_client(
            request,
            db,
            user_id=user_id,
            api_key_id=api_key_id,
            balance_remaining=balance_remaining,
            access_allowed=access_allowed,
            strict=False,
        )

    @staticmethod
    def _is_trusted_gateway_request(request: Request) -> bool:
        gateway_marker = str(request.headers.get(TRUSTED_GATEWAY_HEADER) or "").strip().lower()
        if not gateway_marker.startswith("rust-phase3"):
            return False

        host = request.client.host if request.client else ""
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _parse_trusted_balance_remaining(value: str | None) -> float | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_trusted_bool_header(value: str | None, *, default: bool) -> bool:
        raw = str(value or "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return default

    def _capture_trusted_gateway_execution_path(self, request: Request) -> str | None:
        if not self._is_trusted_gateway_request(request):
            return None

        execution_path = str(
            request.headers.get(TRUSTED_GATEWAY_EXECUTION_PATH_HEADER) or ""
        ).strip()
        if not execution_path:
            return None

        update_request_state(request, gateway_execution_path=execution_path)
        return execution_path

    def _trusted_gateway_completed_rate_limit_preflight(self, request: Request) -> bool:
        if not self._is_trusted_gateway_request(request):
            return False
        return self._parse_trusted_bool_header(
            request.headers.get(TRUSTED_RATE_LIMIT_PREFLIGHT_HEADER),
            default=False,
        )

    async def _try_token_prefix_auth(
        self, token: str, request: Request, db: Session
    ) -> tuple[User, Any] | None:
        """尝试通过模块注册的 token 前缀认证器认证

        Returns:
            (User, token_record) 元组，或 None（无前缀匹配）

        Raises:
            HTTPException: 前缀匹配但认证失败时抛出 401
        """
        from src.core.modules.hooks import AUTH_TOKEN_PREFIX_AUTHENTICATORS, get_hook_dispatcher

        authenticators = await get_hook_dispatcher().dispatch(AUTH_TOKEN_PREFIX_AUTHENTICATORS)
        for auth_info in authenticators or []:
            prefix = auth_info.get("prefix", "")
            authenticate_fn = auth_info.get("authenticate")
            if prefix and token.startswith(prefix):
                if not authenticate_fn:
                    logger.warning("Token prefix '{}' has no authenticate callback", prefix)
                    raise HTTPException(status_code=401, detail="认证服务不可用")
                identity = get_request_identity_metadata(request)
                auth_db = create_session()
                try:
                    result = await authenticate_fn(auth_db, token, identity.client_ip)
                    if result:
                        for instance in result:
                            if instance is None:
                                continue
                            try:
                                auth_db.expunge(instance)
                            except Exception:
                                pass
                        return result
                finally:
                    auth_db.close()
                # 前缀匹配但认证失败
                module_name = auth_info.get("module", "unknown")
                raise HTTPException(status_code=401, detail=f"无效或过期的 Token ({module_name})")
        return None  # 无前缀匹配

    def _reattach_token_auth_result(
        self,
        db: Session,
        token_auth_result: tuple[User, Any],
    ) -> tuple[User, Any]:
        """将前缀认证返回对象绑定到当前请求会话，避免后续写入失效。"""
        user, management_token = token_auth_result

        db_user = db.query(User).filter(User.id == user.id).first()
        if not db_user:
            raise HTTPException(status_code=401, detail="无效或过期的 Token")

        if management_token is None:
            return db_user, None

        token_id = getattr(management_token, "id", None)
        token_model: Any = type(management_token)
        if token_id is None or not hasattr(token_model, "id"):
            return db_user, management_token

        db_management_token = db.query(token_model).filter(token_model.id == token_id).first()
        if not db_management_token:
            raise HTTPException(status_code=401, detail="无效或过期的 Token")
        return db_user, db_management_token

    async def _authenticate_admin(
        self, request: Request, db: Session
    ) -> tuple[User, ManagementToken | None]:
        """Legacy Python admin auth; supports JWT and Management Token."""
        authorization = request.headers.get("authorization")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="缺少管理员凭证")

        token = authorization[7:].strip()

        token_auth_result = await self._try_token_prefix_auth(token, request, db)
        if token_auth_result is not None:
            user, management_token = self._reattach_token_auth_result(db, token_auth_result)

            if user.role != UserRole.ADMIN:
                logger.warning("非管理员尝试通过 Management Token 访问管理端点: {}", user.email)
                raise HTTPException(status_code=403, detail="需要管理员权限")

            update_request_state(
                request,
                user_id=user.id,
                management_token_id=management_token.id if management_token else None,
                user_session_id=None,
            )
            return user, management_token

        try:
            payload = await self.auth_service.verify_token(token, token_type="access")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Admin token 验证失败: {}", exc)
            raise HTTPException(status_code=401, detail="无效的管理员令牌")

        user_id = payload.get("user_id")
        session_id = payload.get("session_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="无效的管理员令牌")
        if not session_id:
            raise HTTPException(status_code=401, detail="登录会话已失效，请重新登录")

        db_user = db.query(User).filter(User.id == user_id).first()
        if not db_user or not db_user.is_active or db_user.is_deleted:
            raise HTTPException(status_code=403, detail="用户不存在或已禁用")

        if not self.auth_service.token_identity_matches_user(payload, db_user):
            raise HTTPException(status_code=403, detail="无效的管理员令牌")

        if db_user.role != UserRole.ADMIN:
            logger.warning("非管理员尝试通过 JWT 访问管理端点: {}", db_user.email)
            raise HTTPException(status_code=403, detail="需要管理员权限")

        session = self._touch_authenticated_session(
            request,
            db,
            user_id=str(user_id),
            session_id=str(session_id),
            scope="admin",
        )
        update_request_state(
            request,
            user_session_id=session.id,
            user_id=db_user.id,
        )
        return db_user, None

    async def _authenticate_user(
        self, request: Request, db: Session
    ) -> tuple[User, ManagementToken | None]:
        """Legacy Python user auth; supports JWT and Management Token."""
        authorization = request.headers.get("authorization")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="缺少用户凭证")

        token = authorization[7:].strip()

        token_auth_result = await self._try_token_prefix_auth(token, request, db)
        if token_auth_result is not None:
            user, management_token = self._reattach_token_auth_result(db, token_auth_result)
            update_request_state(
                request,
                user_id=user.id,
                management_token_id=management_token.id if management_token else None,
                user_session_id=None,
            )
            return user, management_token

        try:
            payload = await self.auth_service.verify_token(token, token_type="access")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("User token 验证失败: {}", exc)
            raise HTTPException(status_code=401, detail="无效的用户令牌")

        user_id = payload.get("user_id")
        session_id = payload.get("session_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="无效的用户令牌")
        if not session_id:
            raise HTTPException(status_code=401, detail="登录会话已失效，请重新登录")

        db_user = db.query(User).filter(User.id == user_id).first()
        if not db_user or not db_user.is_active or db_user.is_deleted:
            raise HTTPException(status_code=403, detail="用户不存在或已禁用")

        if not self.auth_service.token_identity_matches_user(payload, db_user):
            raise HTTPException(status_code=403, detail="无效的用户令牌")

        session = self._touch_authenticated_session(
            request,
            db,
            user_id=str(user_id),
            session_id=str(session_id),
            scope="user",
        )
        update_request_state(
            request,
            user_session_id=session.id,
            user_id=db_user.id,
        )
        return db_user, None

    async def _authenticate_management(
        self, request: Request, db: Session
    ) -> tuple[User, ManagementToken]:
        """Legacy Python Management Token 认证。"""
        authorization = request.headers.get("authorization")
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="缺少 Management Token")

        token = authorization[7:].strip()

        # 通过钩子检查是否匹配模块注册的 token 前缀
        # _try_token_prefix_auth 会在前缀匹配但认证失败时直接抛 HTTPException
        token_auth_result = await self._try_token_prefix_auth(token, request, db)
        if token_auth_result is not None:
            user, management_token = self._reattach_token_auth_result(db, token_auth_result)

            update_request_state(
                request,
                user_id=user.id,
                management_token_id=management_token.id if management_token else None,
                user_session_id=None,
            )

            return user, management_token

        raise HTTPException(
            status_code=401,
            detail="无效的 Token 格式，需要 Management Token",
        )

    async def _calculate_balance_remaining_async(
        self, user: User | None, api_key: ApiKey | None = None
    ) -> float | None:
        if not user:
            return None

        user_id = getattr(user, "id", None)
        api_key_id = getattr(api_key, "id", None)

        # API Key 链路通常已在认证阶段预取余额；这里只保留为无预取路径的兜底查询。
        def _load_balance() -> float | None:
            thread_db = create_session()
            try:
                db_user = (
                    thread_db.query(User).filter(User.id == user_id).first() if user_id else None
                )
                db_api_key = (
                    thread_db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
                    if api_key_id
                    else None
                )
                balance = WalletService.get_balance_snapshot(
                    thread_db,
                    user=db_user,
                    api_key=db_api_key,
                )
                return float(balance) if balance is not None else None
            finally:
                thread_db.close()

        return await run_in_threadpool(_load_balance)

    def _record_audit_event(
        self,
        context: ApiRequestContext,
        adapter: ApiAdapter,
        *,
        success: bool,
        status_code: int | None = None,
        error: str | None = None,
    ) -> None:
        """Legacy Python request-scoped audit writer.

        事务策略：
        - 默认复用请求级 Session，由中间件在请求结束时统一提交。
        - 若路由已显式提交主事务（tx_committed_by_route=True），则审计日志会落在新的事务中，
          这里需要立即提交，否则中间件会跳过二次提交，导致审计记录丢失。
        """
        if not getattr(adapter, "audit_log_enabled", True):
            return

        if context.db is None:
            return

        event_type = adapter.audit_success_event if success else adapter.audit_failure_event
        if not event_type:
            if not success and status_code == 401:
                event_type = AuditEventType.UNAUTHORIZED_ACCESS
            else:
                event_type = (
                    AuditEventType.REQUEST_SUCCESS if success else AuditEventType.REQUEST_FAILED
                )

        metadata = self._build_audit_metadata(
            context=context,
            adapter=adapter,
            success=success,
            status_code=status_code,
            error=error,
        )

        context.sync_runtime_state_from_request()

        try:
            # 复用请求级 Session，不创建新的连接
            # 审计记录随主事务一起提交，由中间件统一管理
            self.audit_service.log_event(
                db=context.db,
                event_type=event_type,
                description=f"{context.request_method} {context.request_path} via {adapter.name}",
                user_id=context.user.id if context.user else None,
                api_key_id=context.api_key.id if context.api_key else None,
                ip_address=context.client_ip,
                user_agent=context.user_agent,
                request_id=context.request_id,
                status_code=status_code,
                error_message=error,
                metadata=metadata,
            )
            if context.tx_committed_by_route:
                try:
                    context.db.commit()
                except Exception:
                    context.db.rollback()
                    raise
        except Exception as exc:
            # 审计失败不应影响主请求，仅记录警告
            logger.warning("[Audit] Failed to record event for adapter={}: {}", adapter.name, exc)

    def _build_audit_metadata(
        self,
        context: ApiRequestContext,
        adapter: ApiAdapter,
        *,
        success: bool,
        status_code: int | None,
        error: str | None,
    ) -> dict:
        duration_ms = max((time.time() - context.start_time) * 1000, 0.0)
        metadata: dict[str, Any] = {
            "path": context.request_path,
            "path_params": dict(context.path_params or {}),
            "method": context.request_method,
            "adapter": adapter.name,
            "adapter_class": adapter.__class__.__name__,
            "adapter_mode": getattr(adapter.mode, "value", str(adapter.mode)),
            "mode": context.mode,
            "api_format_hint": context.api_format_hint,
            "query": context.query_params,
            "duration_ms": round(duration_ms, 2),
            "request_body_bytes": len(context.raw_body or b""),
            "has_body": bool(context.raw_body),
            "request_content_type": context.request_content_type,
            "balance_remaining": context.balance_remaining,
            "success": success,
            # 传递 quiet_logging 标志给审计服务，用于抑制高频轮询日志
            "quiet_logging": getattr(context, "quiet_logging", False),
        }
        if status_code is not None:
            metadata["status_code"] = status_code
        if context.gateway_execution_path:
            metadata["gateway_execution_path"] = context.gateway_execution_path
        if context.rate_limit_scope:
            metadata["rate_limit_scope"] = context.rate_limit_scope

        if context.user and getattr(context.user, "role", None):
            role = context.user.role
            metadata["user_role"] = getattr(role, "value", role)

        if context.api_key:
            if getattr(context.api_key, "name", None):
                metadata["api_key_name"] = context.api_key.name
            # 使用脱敏后的密钥显示
            if hasattr(context.api_key, "get_display_key"):
                metadata["api_key_display"] = context.api_key.get_display_key()

        extra_details: dict[str, Any] = {}
        if context.audit_metadata:
            extra_details.update(context.audit_metadata)

        try:
            adapter_details = adapter.get_audit_metadata(
                context,
                success=success,
                status_code=status_code,
                error=error,
            )
            if adapter_details:
                extra_details.update(adapter_details)
        except Exception as exc:
            logger.warning(
                "[Audit] Adapter metadata failed: {}: {}", adapter.__class__.__name__, exc
            )

        if extra_details:
            metadata["details"] = extra_details

        if error:
            metadata["error"] = error

        return self._sanitize_metadata(metadata)

    def _sanitize_metadata(self, value: Any, depth: int = 0) -> Any:
        if value is None:
            return None
        if depth > 5:
            return str(value)
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            sanitized = {}
            for key, val in value.items():
                cleaned = self._sanitize_metadata(val, depth + 1)
                if cleaned is not None:
                    sanitized[str(key)] = cleaned
            return sanitized
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_metadata(item, depth + 1) for item in value]
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
        return str(value)


_shared_pipeline = ApiRequestPipeline()


def get_pipeline() -> ApiRequestPipeline:
    """返回全局共享的无状态请求管道实例。"""
    return _shared_pipeline
