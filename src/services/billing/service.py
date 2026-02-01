from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.config.settings import config
from src.core.logger import logger
from src.services.billing.dimension_collector_service import DimensionCollectorService
from src.services.billing.formula_engine import BillingIncompleteError, FormulaEngine
from src.services.billing.rule_service import BillingRuleService
from src.services.model.cost import ModelCostService

from .schema import BILLING_SNAPSHOT_SCHEMA_VERSION, BillingSnapshot, CostResult


class BillingService:
    """
    BillingService (pure-ish application helper for billing domain).

    Notes:
    - This service **does not** write Usage rows.
    - It may read billing rules & collectors from DB.
    """

    def __init__(self, db: Session):
        self.db = db
        self._formula_engine = FormulaEngine()
        self._dimension_collector = DimensionCollectorService(db)

    def collect_dimensions(
        self,
        *,
        api_format: str | None,
        task_type: str | None,
        request: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        base_dimensions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._dimension_collector.collect_dimensions(
            api_format=api_format,
            task_type=task_type,
            request=request,
            response=response,
            metadata=metadata,
            base_dimensions=base_dimensions,
        )

    def calculate(
        self,
        *,
        task_type: str,
        model: str,
        provider_id: str,
        dimensions: dict[str, Any],
        strict_mode: bool | None = None,
    ) -> CostResult:
        """
        Calculate cost for a task.

        Returns:
            CostResult (includes BillingSnapshot)

        Raises:
            BillingIncompleteError: when strict_mode=True and required dims missing.
        """
        strict = config.billing_strict_mode if strict_mode is None else bool(strict_mode)

        lookup = BillingRuleService.find_rule(
            self.db,
            provider_id=provider_id,
            model_name=model,
            task_type=task_type,
        )

        if lookup and lookup.rule and lookup.rule.expression:
            rule = lookup.rule
            result = self._formula_engine.evaluate(
                expression=rule.expression,
                variables=rule.variables or {},
                dimensions=dimensions,
                dimension_mappings=rule.dimension_mappings or {},
                strict_mode=strict,
            )
            cost = float(result.cost) if result.status == "complete" else 0.0
            snapshot = BillingSnapshot(
                schema_version=BILLING_SNAPSHOT_SCHEMA_VERSION,
                rule_id=str(rule.id),
                rule_name=str(rule.name),
                scope=str(getattr(lookup, "scope", None) or ""),
                expression=str(rule.expression),
                dimensions_used=dimensions,
                missing_required=result.missing_required,
                cost=cost,
                status=result.status,
                calculated_at=datetime.now(timezone.utc).isoformat(),
            )
            return CostResult(cost=cost, status=result.status, snapshot=snapshot)

        # No rule fallback
        if task_type in ("chat", "cli"):
            input_tokens = int(dimensions.get("input_tokens") or 0)
            output_tokens = int(dimensions.get("output_tokens") or 0)
            cost = float(
                ModelCostService.calculate_cost(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            snapshot = BillingSnapshot(
                schema_version=BILLING_SNAPSHOT_SCHEMA_VERSION,
                rule_id=None,
                rule_name=None,
                scope=None,
                expression=None,
                dimensions_used=dimensions,
                missing_required=[],
                cost=cost,
                status="legacy",
                calculated_at=datetime.now(timezone.utc).isoformat(),
            )
            return CostResult(cost=cost, status="legacy", snapshot=snapshot)

        logger.warning(
            "No billing rule for task (task_type=%s, model=%s, provider_id=%s)",
            task_type,
            model,
            provider_id,
        )
        snapshot = BillingSnapshot(
            schema_version=BILLING_SNAPSHOT_SCHEMA_VERSION,
            rule_id=None,
            rule_name=None,
            scope=None,
            expression=None,
            dimensions_used=dimensions,
            missing_required=[],
            cost=0.0,
            status="no_rule",
            calculated_at=datetime.now(timezone.utc).isoformat(),
        )
        return CostResult(cost=0.0, status="no_rule", snapshot=snapshot)
