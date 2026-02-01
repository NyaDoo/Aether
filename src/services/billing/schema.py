"""
Billing schema (stable contracts)

These dataclasses are meant to be stored in `Usage.request_metadata` / `Task.request_metadata`
for auditability. They are internal-only and MUST NOT be exposed to end users without sanitizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BILLING_SNAPSHOT_SCHEMA_VERSION = "1.0"

BillingSnapshotStatus = Literal["complete", "incomplete", "no_rule", "legacy"]


@dataclass(frozen=True)
class BillingSnapshot:
    """Stable billing snapshot for audit."""

    schema_version: str = BILLING_SNAPSHOT_SCHEMA_VERSION

    # Rule info (optional for legacy/no_rule)
    rule_id: str | None = None
    rule_name: str | None = None
    scope: str | None = None

    # Rule expression (internal, do not expose to clients)
    expression: str | None = None

    # Dimensions
    dimensions_used: dict[str, Any] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)

    # Result
    cost: float = 0.0
    status: BillingSnapshotStatus = "no_rule"

    # Audit
    calculated_at: str = ""  # ISO 8601

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "scope": self.scope,
            "expression": self.expression,
            "dimensions_used": self.dimensions_used,
            "missing_required": self.missing_required,
            "cost": self.cost,
            "status": self.status,
            "calculated_at": self.calculated_at,
        }


@dataclass(frozen=True)
class CostResult:
    """Billing calculation output."""

    cost: float
    status: BillingSnapshotStatus
    snapshot: BillingSnapshot
