"""Admin comparison stats routes."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from src.api.base.admin_adapter import AdminApiAdapter
from src.api.base.context import ApiRequestContext
from src.config.constants import CacheTTL
from src.database import get_db
from src.services.system.stats_aggregator import AggregatedStats, StatsFilter, query_stats_hybrid
from src.services.system.time_range import TimeRangeParams
from src.utils.cache_decorator import cache_result

from .common import pipeline

router = APIRouter()


class AdminComparisonAdapter(AdminApiAdapter):
    def __init__(
        self,
        current_start: date,
        current_end: date,
        comparison_type: Literal["period", "year"],
        timezone_name: str | None,
        tz_offset_minutes: int | None,
    ) -> None:
        self.current_start = current_start
        self.current_end = current_end
        self.comparison_type = comparison_type
        self.timezone_name = timezone_name
        self.tz_offset_minutes = tz_offset_minutes

    @cache_result(
        key_prefix="admin:stats:comparison",
        ttl=CacheTTL.ADMIN_USAGE_AGGREGATION,
        user_specific=False,
        vary_by=[
            "current_start",
            "current_end",
            "comparison_type",
            "timezone_name",
            "tz_offset_minutes",
        ],
    )
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        if self.current_start > self.current_end:
            raise HTTPException(status_code=400, detail="current_start must be <= current_end")

        days = (self.current_end - self.current_start).days + 1

        def _safe_year_shift(value: date) -> date:
            try:
                return value.replace(year=value.year - 1)
            except ValueError:
                return value.replace(year=value.year - 1, day=28)

        if self.comparison_type == "period":
            comparison_end = self.current_start - timedelta(days=1)
            comparison_start = comparison_end - timedelta(days=days - 1)
        else:
            comparison_start = _safe_year_shift(self.current_start)
            comparison_end = _safe_year_shift(self.current_end)

        current_range = TimeRangeParams(
            start_date=self.current_start,
            end_date=self.current_end,
            timezone=self.timezone_name,
            tz_offset_minutes=self.tz_offset_minutes or 0,
        ).validate_and_resolve()
        comparison_range = TimeRangeParams(
            start_date=comparison_start,
            end_date=comparison_end,
            timezone=self.timezone_name,
            tz_offset_minutes=self.tz_offset_minutes or 0,
        ).validate_and_resolve()

        current_stats = query_stats_hybrid(context.db, current_range, filters=StatsFilter())
        comparison_stats = query_stats_hybrid(context.db, comparison_range, filters=StatsFilter())

        def _stats_payload(stats: AggregatedStats) -> dict[str, Any]:
            total_tokens = (
                stats.input_tokens
                + stats.output_tokens
                + stats.cache_creation_tokens
                + stats.cache_read_tokens
            )
            return {
                "total_requests": stats.total_requests,
                "total_tokens": total_tokens,
                "total_cost": float(stats.total_cost),
                "actual_total_cost": float(stats.actual_total_cost),
                "avg_response_time_ms": float(stats.avg_response_time_ms),
                "error_requests": stats.error_requests,
            }

        def _pct_change(current: float, previous: float) -> float | None:
            if previous == 0:
                return None if current != 0 else 0.0
            return round((current - previous) / previous * 100, 2)

        current_payload = _stats_payload(current_stats)
        comparison_payload = _stats_payload(comparison_stats)
        changes = {
            key: _pct_change(float(current_payload[key]), float(comparison_payload[key]))
            for key in current_payload.keys()
        }

        return {
            "current": current_payload,
            "comparison": comparison_payload,
            "change_percent": changes,
            "current_start": self.current_start.isoformat(),
            "current_end": self.current_end.isoformat(),
            "comparison_start": comparison_start.isoformat(),
            "comparison_end": comparison_end.isoformat(),
        }


@router.get("/comparison")
async def get_comparison(
    request: Request,
    db: Session = Depends(get_db),
    current_start: date = Query(...),
    current_end: date = Query(...),
    comparison_type: Literal["period", "year"] = Query("period"),
    timezone_name: str | None = Query(None, alias="timezone"),
    tz_offset_minutes: int | None = Query(0),
) -> Any:
    adapter = AdminComparisonAdapter(
        current_start=current_start,
        current_end=current_end,
        comparison_type=comparison_type,
        timezone_name=timezone_name,
        tz_offset_minutes=tz_offset_minutes,
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)
