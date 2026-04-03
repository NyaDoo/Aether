"""Admin time series stats routes."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from src.api.base.admin_adapter import AdminApiAdapter
from src.api.base.context import ApiRequestContext
from src.config.constants import CacheTTL
from src.database import get_db
from src.services.system.stats_aggregator import TimeSeriesFilter, query_time_series
from src.services.system.time_range import TimeRangeParams
from src.utils.cache_decorator import cache_result

from .common import _apply_admin_default_range, _build_time_range_params, pipeline

router = APIRouter()


class AdminTimeSeriesAdapter(AdminApiAdapter):
    def __init__(
        self,
        time_range: TimeRangeParams | None,
        user_id: str | None,
        model: str | None,
        provider_name: str | None,
    ) -> None:
        self.time_range = _apply_admin_default_range(time_range)
        self.user_id = user_id
        self.model = model
        self.provider_name = provider_name

    @cache_result(
        key_prefix="admin:stats:time_series",
        ttl=CacheTTL.ADMIN_USAGE_AGGREGATION,
        user_specific=False,
        vary_by=[
            "time_range.start_date",
            "time_range.end_date",
            "time_range.preset",
            "time_range.timezone",
            "time_range.tz_offset_minutes",
            "time_range.granularity",
            "user_id",
            "model",
            "provider_name",
        ],
    )
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        if not self.time_range:
            return []
        try:
            return query_time_series(
                context.db,
                self.time_range,
                filters=TimeSeriesFilter(
                    user_id=self.user_id, model=self.model, provider_name=self.provider_name
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/time-series")
async def get_time_series(
    request: Request,
    db: Session = Depends(get_db),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    preset: str | None = Query(None),
    granularity: Literal["hour", "day", "week", "month"] = Query("day"),
    timezone_name: str | None = Query(None, alias="timezone"),
    tz_offset_minutes: int | None = Query(0),
    user_id: str | None = Query(None),
    model: str | None = Query(None),
    provider_name: str | None = Query(None),
) -> Any:
    time_range = _build_time_range_params(
        start_date, end_date, preset, timezone_name, tz_offset_minutes
    )
    if time_range:
        time_range.granularity = granularity
    adapter = AdminTimeSeriesAdapter(
        time_range=time_range,
        user_id=user_id,
        model=model,
        provider_name=provider_name,
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)
