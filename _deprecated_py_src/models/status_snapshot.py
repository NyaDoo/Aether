from __future__ import annotations

from pydantic import BaseModel, Field


class OAuthStatusSnapshotResponse(BaseModel):
    code: str = Field(default="none", description="OAuth 状态代码")
    label: str | None = Field(default=None, description="OAuth 状态标签")
    reason: str | None = Field(default=None, description="OAuth 状态原因")
    expires_at: int | None = Field(default=None, description="OAuth 过期时间（Unix 时间戳）")
    invalid_at: int | None = Field(default=None, description="OAuth 失效时间（Unix 时间戳）")
    source: str | None = Field(default=None, description="OAuth 状态来源")
    requires_reauth: bool = Field(default=False, description="是否需要重新授权")
    expiring_soon: bool = Field(default=False, description="是否即将过期")


class AccountStatusSnapshotResponse(BaseModel):
    code: str = Field(default="ok", description="账号状态代码")
    label: str | None = Field(default=None, description="账号状态标签")
    reason: str | None = Field(default=None, description="账号状态原因")
    blocked: bool = Field(default=False, description="是否为账号级阻塞")
    source: str | None = Field(default=None, description="账号状态来源")
    recoverable: bool = Field(default=False, description="是否为可恢复状态")


class QuotaStatusSnapshotResponse(BaseModel):
    code: str = Field(default="unknown", description="额度状态代码")
    label: str | None = Field(default=None, description="额度状态标签")
    reason: str | None = Field(default=None, description="额度状态原因")
    exhausted: bool = Field(default=False, description="额度是否耗尽")
    usage_ratio: float | None = Field(default=None, description="额度使用比例 [0, 1]")
    updated_at: int | None = Field(default=None, description="额度刷新时间（Unix 时间戳）")
    reset_seconds: float | None = Field(default=None, description="距离重置剩余秒数")
    plan_type: str | None = Field(default=None, description="额度读取到的套餐类型")


class ProviderKeyStatusSnapshotResponse(BaseModel):
    oauth: OAuthStatusSnapshotResponse = Field(default_factory=OAuthStatusSnapshotResponse)
    account: AccountStatusSnapshotResponse = Field(default_factory=AccountStatusSnapshotResponse)
    quota: QuotaStatusSnapshotResponse = Field(default_factory=QuotaStatusSnapshotResponse)
