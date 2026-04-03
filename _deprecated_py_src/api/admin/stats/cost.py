"""Admin cost stats routes."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.base.admin_adapter import AdminApiAdapter
from src.api.base.context import ApiRequestContext
from src.config.constants import CacheTTL
from src.database import get_db
from src.models.database import Usage
from src.services.system.stats_aggregator import query_time_series
from src.services.system.time_range import TimeRangeParams
from src.utils.cache_decorator import cache_result

from .common import (
    _apply_admin_default_range,
    _build_time_range_from_days,
    _build_time_range_params,
    _linear_regression,
    pipeline,
)

router = APIRouter()


class AdminCostForecastAdapter(AdminApiAdapter):
    def __init__(
        self,
        time_range: TimeRangeParams | None,
        days: int,
        forecast_days: int,
        timezone_name: str | None,
        tz_offset_minutes: int | None,
    ) -> None:
        self.time_range = time_range
        self.days = days
        self.forecast_days = forecast_days
        self.timezone_name = timezone_name
        self.fallback_tz_offset_minutes = tz_offset_minutes

    @cache_result(
        key_prefix="admin:stats:cost:forecast",
        ttl=CacheTTL.ADMIN_USAGE_AGGREGATION,
        user_specific=False,
        vary_by=[
            "time_range.start_date",
            "time_range.end_date",
            "time_range.preset",
            "time_range.timezone",
            "time_range.tz_offset_minutes",
            "days",
            "forecast_days",
            "timezone_name",
            "fallback_tz_offset_minutes",
        ],
    )
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        time_range = self.time_range or _build_time_range_from_days(
            self.days, self.timezone_name, self.fallback_tz_offset_minutes
        )
        time_range.granularity = "day"
        try:
            series = query_time_series(context.db, time_range)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        history = [
            {"date": item["date"], "total_cost": float(item.get("total_cost", 0.0))}
            for item in series
        ]

        values = [item["total_cost"] for item in history]
        slope, intercept = _linear_regression(values)

        forecast = []
        if history:
            last_date = date.fromisoformat(history[-1]["date"])
        else:
            last_date = time_range.end_date
        for i in range(self.forecast_days):
            idx = len(values) + i
            predicted = max(0.0, slope * idx + intercept)
            forecast.append(
                {
                    "date": (last_date + timedelta(days=i + 1)).isoformat(),
                    "total_cost": round(predicted, 4),
                }
            )

        return {
            "history": history,
            "forecast": forecast,
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "start_date": time_range.start_date.isoformat(),
            "end_date": time_range.end_date.isoformat(),
        }


@router.get("/cost/forecast")
async def get_cost_forecast(
    request: Request,
    db: Session = Depends(get_db),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    preset: str | None = Query(None),
    timezone_name: str | None = Query(None, alias="timezone"),
    tz_offset_minutes: int | None = Query(0),
    days: int = Query(30, ge=7, le=365),
    forecast_days: int = Query(7, ge=1, le=90),
) -> Any:
    time_range = _build_time_range_params(
        start_date, end_date, preset, timezone_name, tz_offset_minutes
    )
    adapter = AdminCostForecastAdapter(
        time_range=time_range,
        days=days,
        forecast_days=forecast_days,
        timezone_name=timezone_name,
        tz_offset_minutes=tz_offset_minutes,
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


class AdminCostSavingsAdapter(AdminApiAdapter):
    def __init__(
        self,
        time_range: TimeRangeParams | None,
        provider_name: str | None,
        model: str | None,
    ) -> None:
        self.time_range = _apply_admin_default_range(time_range)
        self.provider_name = provider_name
        self.model = model

    @cache_result(
        key_prefix="admin:stats:cost:savings",
        ttl=CacheTTL.ADMIN_USAGE_AGGREGATION,
        user_specific=False,
        vary_by=[
            "time_range.start_date",
            "time_range.end_date",
            "time_range.preset",
            "time_range.timezone",
            "time_range.tz_offset_minutes",
            "provider_name",
            "model",
        ],
    )
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        if not self.time_range:
            return {
                "cache_read_tokens": 0,
                "cache_read_cost": 0.0,
                "cache_creation_cost": 0.0,
                "estimated_full_cost": 0.0,
                "cache_savings": 0.0,
            }

        start_utc, end_utc = self.time_range.to_utc_datetime_range()
        query = context.db.query(
            func.sum(Usage.cache_read_input_tokens).label("cache_read_tokens"),
            func.sum(Usage.cache_read_cost_usd).label("cache_read_cost"),
            func.sum(Usage.cache_creation_cost_usd).label("cache_creation_cost"),
            func.sum(
                func.coalesce(Usage.output_price_per_1m, 0) * Usage.cache_read_input_tokens
            ).label("estimated_full_cost_raw"),
        ).filter(Usage.created_at >= start_utc, Usage.created_at < end_utc)
        if self.provider_name:
            query = query.filter(Usage.provider_name == self.provider_name)
        if self.model:
            query = query.filter(Usage.model == self.model)

        row = query.first()
        cache_read_tokens = int(getattr(row, "cache_read_tokens", 0) or 0)
        cache_read_cost = float(getattr(row, "cache_read_cost", 0) or 0.0)
        cache_creation_cost = float(getattr(row, "cache_creation_cost", 0) or 0.0)
        estimated_full_cost = float(getattr(row, "estimated_full_cost_raw", 0) or 0.0) / 1_000_000
        if estimated_full_cost <= 0 and cache_read_cost > 0:
            estimated_full_cost = cache_read_cost * 10
        cache_savings = estimated_full_cost - cache_read_cost

        return {
            "cache_read_tokens": cache_read_tokens,
            "cache_read_cost": round(cache_read_cost, 6),
            "cache_creation_cost": round(cache_creation_cost, 6),
            "estimated_full_cost": round(estimated_full_cost, 6),
            "cache_savings": round(cache_savings, 6),
        }


@router.get("/cost/savings")
async def get_cost_savings(
    request: Request,
    db: Session = Depends(get_db),
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    preset: str | None = Query(None),
    timezone_name: str | None = Query(None, alias="timezone"),
    tz_offset_minutes: int | None = Query(0),
    provider_name: str | None = Query(None),
    model: str | None = Query(None),
) -> Any:
    time_range = _build_time_range_params(
        start_date, end_date, preset, timezone_name, tz_offset_minutes
    )
    adapter = AdminCostSavingsAdapter(
        time_range=time_range, provider_name=provider_name, model=model
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)
