"""Admin error stats routes."""

from __future__ import annotations

from datetime import date, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.base.admin_adapter import AdminApiAdapter
from src.api.base.context import ApiRequestContext
from src.config.constants import CacheTTL
from src.database import get_db
from src.models.database import StatsDailyError, Usage
from src.services.system.time_range import TimeRangeParams
from src.utils.cache_decorator import cache_result

from .common import _apply_admin_default_range, _build_time_range_params, pipeline

router = APIRouter()


class AdminErrorDistributionAdapter(AdminApiAdapter):
    def __init__(self, time_range: TimeRangeParams | None) -> None:
        self.time_range = _apply_admin_default_range(time_range)

    @cache_result(
        key_prefix="admin:stats:errors:distribution",
        ttl=CacheTTL.ADMIN_USAGE_AGGREGATION,
        user_specific=False,
        vary_by=[
            "time_range.start_date",
            "time_range.end_date",
            "time_range.preset",
            "time_range.timezone",
            "time_range.tz_offset_minutes",
        ],
    )
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        if not self.time_range:
            return {"distribution": [], "trend": []}

        time_range = self.time_range
        is_utc = (time_range.timezone in {None, "UTC"}) and time_range.tz_offset_minutes == 0
        distribution: dict[str, int] = {}
        trend: dict[str, dict[str, int]] = {}

        if is_utc:
            start_utc, end_utc = time_range.to_utc_datetime_range()
            rows = (
                context.db.query(StatsDailyError)
                .filter(StatsDailyError.date >= start_utc, StatsDailyError.date < end_utc)
                .all()
            )
            for row in rows:
                date_str = (
                    row.date.astimezone(timezone.utc).date().isoformat()
                    if row.date.tzinfo
                    else row.date.date().isoformat()
                )
                distribution[row.error_category] = distribution.get(row.error_category, 0) + int(
                    row.count or 0
                )
                trend.setdefault(date_str, {})
                trend[date_str][row.error_category] = trend[date_str].get(
                    row.error_category, 0
                ) + int(row.count or 0)
        else:
            for local_date, day_start_utc, day_end_utc in time_range.get_local_day_hours():
                rows = (
                    context.db.query(
                        Usage.error_category,
                        func.count(Usage.id).label("count"),
                    )
                    .filter(
                        Usage.created_at >= day_start_utc,
                        Usage.created_at < day_end_utc,
                        Usage.error_category.isnot(None),
                    )
                    .group_by(Usage.error_category)
                    .all()
                )
                date_str = local_date.isoformat()
                for row in rows:
                    distribution[row.error_category] = distribution.get(
                        row.error_category, 0
                    ) + int(row.count or 0)
                    trend.setdefault(date_str, {})
                    trend[date_str][row.error_category] = trend[date_str].get(
                        row.error_category, 0
                    ) + int(row.count or 0)

        trend_items = []
        for day in sorted(trend.keys()):
            counts = trend[day]
            total = sum(counts.values())
            trend_items.append({"date": day, "total": total, "categories": counts})

        distribution_items = [
            {"category": category, "count": count}
            for category, count in sorted(
                distribution.items(), key=lambda item: item[1], reverse=True
            )
        ]

        return {"distribution": distribution_items, "trend": trend_items}


@router.get("/errors/distribution")
async def get_error_distribution(
    request: Request,
    db: Session = Depends(get_db),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    preset: str | None = Query(None),
    timezone_name: str | None = Query(None, alias="timezone"),
    tz_offset_minutes: int | None = Query(0),
) -> Any:
    time_range = _build_time_range_params(
        start_date, end_date, preset, timezone_name, tz_offset_minutes
    )
    adapter = AdminErrorDistributionAdapter(time_range=time_range)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)
