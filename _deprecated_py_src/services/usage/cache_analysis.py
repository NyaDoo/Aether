from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models.database import Usage, User


class UsageCacheAnalysisMixin:
    """缓存分析方法"""

    @staticmethod
    def analyze_cache_affinity_ttl(
        db: Session,
        user_id: str | None = None,
        api_key_id: str | None = None,
        hours: int = 168,
    ) -> dict[str, Any]:
        """
        分析用户请求间隔分布，推荐合适的缓存亲和性 TTL

        通过分析同一用户连续请求之间的时间间隔，判断用户的使用模式：
        - 高频用户（间隔短）：5 分钟 TTL 足够
        - 中频用户：15-30 分钟 TTL
        - 低频用户（间隔长）：需要 60 分钟 TTL

        Args:
            db: 数据库会话
            user_id: 指定用户 ID（可选，为空则分析所有用户）
            api_key_id: 指定 API Key ID（可选）
            hours: 分析最近多少小时的数据

        Returns:
            包含分析结果的字典
        """
        from sqlalchemy import text

        # 计算时间范围
        start_date = datetime.now(timezone.utc) - timedelta(hours=hours)

        # 构建 SQL 查询 - 使用窗口函数计算请求间隔
        # 按 user_id 或 api_key_id 分组，计算同一组内连续请求的时间差
        group_by_field = "api_key_id" if api_key_id else "user_id"

        # 构建过滤条件
        filter_clause = ""
        if user_id or api_key_id:
            filter_clause = f"AND {group_by_field} = :filter_id"

        sql = text(f"""
            WITH user_requests AS (
                SELECT
                    {group_by_field} as group_id,
                    created_at,
                    LAG(created_at) OVER (
                        PARTITION BY {group_by_field}
                        ORDER BY created_at
                    ) as prev_request_at
                FROM usage
                WHERE status = 'completed'
                  AND created_at > :start_date
                  AND {group_by_field} IS NOT NULL
                  {filter_clause}
            ),
            intervals AS (
                SELECT
                    group_id,
                    EXTRACT(EPOCH FROM (created_at - prev_request_at)) / 60.0 as interval_minutes
                FROM user_requests
                WHERE prev_request_at IS NOT NULL
            ),
            user_stats AS (
                SELECT
                    group_id,
                    COUNT(*) as request_count,
                    COUNT(*) FILTER (WHERE interval_minutes <= 5) as within_5min,
                    COUNT(*) FILTER (WHERE interval_minutes > 5 AND interval_minutes <= 15) as within_15min,
                    COUNT(*) FILTER (WHERE interval_minutes > 15 AND interval_minutes <= 30) as within_30min,
                    COUNT(*) FILTER (WHERE interval_minutes > 30 AND interval_minutes <= 60) as within_60min,
                    COUNT(*) FILTER (WHERE interval_minutes > 60) as over_60min,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY interval_minutes) as median_interval,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY interval_minutes) as p75_interval,
                    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY interval_minutes) as p90_interval,
                    AVG(interval_minutes) as avg_interval,
                    MIN(interval_minutes) as min_interval,
                    MAX(interval_minutes) as max_interval
                FROM intervals
                GROUP BY group_id
                HAVING COUNT(*) >= 2
            )
            SELECT * FROM user_stats
            ORDER BY request_count DESC
        """)

        params: dict[str, Any] = {
            "start_date": start_date,
        }
        if user_id:
            params["filter_id"] = user_id
        elif api_key_id:
            params["filter_id"] = api_key_id

        result = db.execute(sql, params)
        rows = result.fetchall()

        # 收集所有 user_id 以便批量查询用户信息
        group_ids = [row[0] for row in rows]

        # 如果是按 user_id 分组，查询用户信息
        user_info_map: dict[str, dict[str, str]] = {}
        if group_by_field == "user_id" and group_ids:
            users = db.query(User).filter(User.id.in_(group_ids)).all()
            for user in users:
                user_info_map[str(user.id)] = {
                    "username": str(user.username),
                    "email": str(user.email) if user.email else "",
                }

        # 处理结果
        users_analysis = []
        for row in rows:
            # row 是一个 tuple，按查询顺序访问
            (
                group_id,
                request_count,
                within_5min,
                within_15min,
                within_30min,
                within_60min,
                over_60min,
                median_interval,
                p75_interval,
                p90_interval,
                avg_interval,
                min_interval,
                max_interval,
            ) = row

            # 计算推荐 TTL
            recommended_ttl = UsageCacheAnalysisMixin._calculate_recommended_ttl(
                p75_interval, p90_interval
            )

            # 获取用户信息
            user_info = user_info_map.get(str(group_id), {})

            # 计算各区间占比
            total_intervals = request_count
            users_analysis.append(
                {
                    "group_id": group_id,
                    "username": user_info.get("username"),
                    "email": user_info.get("email"),
                    "request_count": request_count,
                    "interval_distribution": {
                        "within_5min": within_5min,
                        "within_15min": within_15min,
                        "within_30min": within_30min,
                        "within_60min": within_60min,
                        "over_60min": over_60min,
                    },
                    "interval_percentages": {
                        "within_5min": round(within_5min / total_intervals * 100, 1),
                        "within_15min": round(within_15min / total_intervals * 100, 1),
                        "within_30min": round(within_30min / total_intervals * 100, 1),
                        "within_60min": round(within_60min / total_intervals * 100, 1),
                        "over_60min": round(over_60min / total_intervals * 100, 1),
                    },
                    "percentiles": {
                        "p50": round(float(median_interval), 2) if median_interval else None,
                        "p75": round(float(p75_interval), 2) if p75_interval else None,
                        "p90": round(float(p90_interval), 2) if p90_interval else None,
                    },
                    "avg_interval_minutes": (
                        round(float(avg_interval), 2) if avg_interval else None
                    ),
                    "min_interval_minutes": (
                        round(float(min_interval), 2) if min_interval else None
                    ),
                    "max_interval_minutes": (
                        round(float(max_interval), 2) if max_interval else None
                    ),
                    "recommended_ttl_minutes": recommended_ttl,
                    "recommendation_reason": UsageCacheAnalysisMixin._get_ttl_recommendation_reason(
                        recommended_ttl, p75_interval, p90_interval
                    ),
                }
            )

        # 汇总统计
        ttl_distribution = {"5min": 0, "15min": 0, "30min": 0, "60min": 0}
        for analysis in users_analysis:
            ttl = analysis["recommended_ttl_minutes"]
            if ttl <= 5:
                ttl_distribution["5min"] += 1
            elif ttl <= 15:
                ttl_distribution["15min"] += 1
            elif ttl <= 30:
                ttl_distribution["30min"] += 1
            else:
                ttl_distribution["60min"] += 1

        return {
            "analysis_period_hours": hours,
            "total_users_analyzed": len(users_analysis),
            "ttl_distribution": ttl_distribution,
            "users": users_analysis,
        }

    @staticmethod
    def _calculate_recommended_ttl(
        p75_interval: float | None,
        p90_interval: float | None,
    ) -> int:
        """
        根据请求间隔分布计算推荐的缓存 TTL

        策略：
        - 如果 90% 的请求间隔都在 5 分钟内 -> 5 分钟 TTL
        - 如果 75% 的请求间隔在 15 分钟内 -> 15 分钟 TTL
        - 如果 75% 的请求间隔在 30 分钟内 -> 30 分钟 TTL
        - 否则 -> 60 分钟 TTL
        """
        if p90_interval is None or p75_interval is None:
            return 5  # 默认值

        # 如果 90% 的间隔都在 5 分钟内
        if p90_interval <= 5:
            return 5

        # 如果 75% 的间隔在 15 分钟内
        if p75_interval <= 15:
            return 15

        # 如果 75% 的间隔在 30 分钟内
        if p75_interval <= 30:
            return 30

        # 低频用户，需要更长的 TTL
        return 60

    @staticmethod
    def _get_ttl_recommendation_reason(
        ttl: int,
        p75_interval: float | None,
        p90_interval: float | None,
    ) -> str:
        """生成 TTL 推荐理由"""
        if p75_interval is None or p90_interval is None:
            return "数据不足，使用默认值"

        if ttl == 5:
            return f"高频用户：90% 的请求间隔在 {p90_interval:.1f} 分钟内"
        elif ttl == 15:
            return f"中高频用户：75% 的请求间隔在 {p75_interval:.1f} 分钟内"
        elif ttl == 30:
            return f"中频用户：75% 的请求间隔在 {p75_interval:.1f} 分钟内"
        else:
            return f"低频用户：75% 的请求间隔为 {p75_interval:.1f} 分钟，建议使用长 TTL"

    @staticmethod
    def get_cache_hit_analysis(
        db: Session,
        user_id: str | None = None,
        api_key_id: str | None = None,
        hours: int = 168,
    ) -> dict[str, Any]:
        """
        分析缓存命中情况

        Args:
            db: 数据库会话
            user_id: 指定用户 ID（可选）
            api_key_id: 指定 API Key ID（可选）
            hours: 分析最近多少小时的数据

        Returns:
            缓存命中分析结果
        """
        start_date = datetime.now(timezone.utc) - timedelta(hours=hours)

        # 基础查询
        query = db.query(
            func.count(Usage.id).label("total_requests"),
            func.sum(Usage.input_tokens).label("total_input_tokens"),
            func.sum(Usage.cache_read_input_tokens).label("total_cache_read_tokens"),
            func.sum(Usage.cache_creation_input_tokens).label("total_cache_creation_tokens"),
            func.sum(Usage.cache_read_cost_usd).label("total_cache_read_cost"),
            func.sum(Usage.cache_creation_cost_usd).label("total_cache_creation_cost"),
        ).filter(
            Usage.status == "completed",
            Usage.created_at >= start_date,
        )

        if user_id:
            query = query.filter(Usage.user_id == user_id)
        if api_key_id:
            query = query.filter(Usage.api_key_id == api_key_id)

        result = query.first()

        if result is None:
            total_requests = 0
            total_input_tokens = 0
            total_cache_read_tokens = 0
            total_cache_creation_tokens = 0
            total_cache_read_cost = 0.0
            total_cache_creation_cost = 0.0
        else:
            total_requests = result.total_requests or 0
            total_input_tokens = result.total_input_tokens or 0
            total_cache_read_tokens = result.total_cache_read_tokens or 0
            total_cache_creation_tokens = result.total_cache_creation_tokens or 0
            total_cache_read_cost = float(result.total_cache_read_cost or 0)
            total_cache_creation_cost = float(result.total_cache_creation_cost or 0)

        # 计算缓存命中率（按 token 数）
        # 总输入上下文 = input_tokens + cache_read_tokens（因为 input_tokens 不含 cache_read）
        # 或者如果 input_tokens 已经包含 cache_read，则直接用 input_tokens
        # 这里假设 cache_read_tokens 是额外的，命中率 = cache_read / (input + cache_read)
        total_context_tokens = total_input_tokens + total_cache_read_tokens
        cache_hit_rate = 0.0
        if total_context_tokens > 0:
            cache_hit_rate = total_cache_read_tokens / total_context_tokens * 100

        # 计算节省的费用
        # 缓存读取价格是正常输入价格的 10%，所以节省了 90%
        # 节省 = cache_read_tokens * (正常价格 - 缓存价格) = cache_read_cost * 9
        # 因为 cache_read_cost 是按 10% 价格算的，如果按 100% 算就是 10 倍
        estimated_savings = total_cache_read_cost * 9  # 节省了 90%

        # 统计有缓存命中的请求数
        requests_with_cache_hit = db.query(func.count(Usage.id)).filter(
            Usage.status == "completed",
            Usage.created_at >= start_date,
            Usage.cache_read_input_tokens > 0,
        )
        if user_id:
            requests_with_cache_hit = requests_with_cache_hit.filter(Usage.user_id == user_id)
        if api_key_id:
            requests_with_cache_hit = requests_with_cache_hit.filter(Usage.api_key_id == api_key_id)
        requests_with_cache_hit_count = int(requests_with_cache_hit.scalar() or 0)

        return {
            "analysis_period_hours": hours,
            "total_requests": total_requests,
            "requests_with_cache_hit": requests_with_cache_hit_count,
            "request_cache_hit_rate": (
                round(requests_with_cache_hit_count / total_requests * 100, 2)
                if total_requests > 0
                else 0
            ),
            "total_input_tokens": total_input_tokens,
            "total_cache_read_tokens": total_cache_read_tokens,
            "total_cache_creation_tokens": total_cache_creation_tokens,
            "token_cache_hit_rate": round(cache_hit_rate, 2),
            "total_cache_read_cost_usd": round(total_cache_read_cost, 4),
            "total_cache_creation_cost_usd": round(total_cache_creation_cost, 4),
            "estimated_savings_usd": round(estimated_savings, 4),
        }

    @staticmethod
    def get_interval_timeline(
        db: Session,
        hours: int = 24,
        limit: int = 10000,
        user_id: str | None = None,
        include_user_info: bool = False,
    ) -> dict[str, Any]:
        """
        获取请求间隔时间线数据，用于散点图展示

        Args:
            db: 数据库会话
            hours: 分析最近多少小时的数据（默认24小时）
            limit: 最大返回数据点数量（默认10000）
            user_id: 指定用户 ID（可选，为空则返回所有用户）
            include_user_info: 是否包含用户信息（用于管理员多用户视图）

        Returns:
            包含时间线数据点的字典，每个数据点包含 model 字段用于按模型区分颜色
        """
        from sqlalchemy import text

        start_date = datetime.now(timezone.utc) - timedelta(hours=hours)

        # 构建用户过滤条件
        user_filter = "AND u.user_id = :user_id" if user_id else ""

        # 根据是否需要用户信息选择不同的查询
        if include_user_info and not user_id:
            # 管理员视图：返回带用户信息的数据点
            # 使用按比例采样，保持每个用户的数据量比例不变
            sql = text(f"""
                WITH request_intervals AS (
                    SELECT
                        u.created_at,
                        u.user_id,
                        u.model,
                        usr.username,
                        LAG(u.created_at) OVER (
                            PARTITION BY u.user_id
                            ORDER BY u.created_at
                        ) as prev_request_at
                    FROM usage u
                    LEFT JOIN users usr ON u.user_id = usr.id
                    WHERE u.status = 'completed'
                      AND u.created_at > :start_date
                      AND u.user_id IS NOT NULL
                      {user_filter}
                ),
                filtered_intervals AS (
                    SELECT
                        created_at,
                        user_id,
                        model,
                        username,
                        EXTRACT(EPOCH FROM (created_at - prev_request_at)) / 60.0 as interval_minutes,
                        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at) as rn
                    FROM request_intervals
                    WHERE prev_request_at IS NOT NULL
                      AND EXTRACT(EPOCH FROM (created_at - prev_request_at)) / 60.0 <= 120
                ),
                total_count AS (
                    SELECT COUNT(*) as cnt FROM filtered_intervals
                ),
                user_totals AS (
                    SELECT user_id, COUNT(*) as user_cnt FROM filtered_intervals GROUP BY user_id
                ),
                user_limits AS (
                    SELECT
                        ut.user_id,
                        CASE WHEN tc.cnt <= :limit THEN ut.user_cnt
                             ELSE GREATEST(CEIL(ut.user_cnt::float * :limit / tc.cnt), 1)::int
                        END as user_limit
                    FROM user_totals ut, total_count tc
                )
                SELECT
                    fi.created_at,
                    fi.user_id,
                    fi.model,
                    fi.username,
                    fi.interval_minutes
                FROM filtered_intervals fi
                JOIN user_limits ul ON fi.user_id = ul.user_id
                WHERE fi.rn <= ul.user_limit
                ORDER BY fi.created_at
            """)
        else:
            # 普通视图：返回时间、间隔和模型信息
            sql = text(f"""
                WITH request_intervals AS (
                    SELECT
                        u.created_at,
                        u.user_id,
                        u.model,
                        LAG(u.created_at) OVER (
                            PARTITION BY u.user_id
                            ORDER BY u.created_at
                        ) as prev_request_at
                    FROM usage u
                    WHERE u.status = 'completed'
                      AND u.created_at > :start_date
                      AND u.user_id IS NOT NULL
                      {user_filter}
                )
                SELECT
                    created_at,
                    model,
                    EXTRACT(EPOCH FROM (created_at - prev_request_at)) / 60.0 as interval_minutes
                FROM request_intervals
                WHERE prev_request_at IS NOT NULL
                  AND EXTRACT(EPOCH FROM (created_at - prev_request_at)) / 60.0 <= 120
                ORDER BY created_at
                LIMIT :limit
            """)

        params: dict[str, Any] = {"start_date": start_date, "limit": limit}
        if user_id:
            params["user_id"] = user_id

        result = db.execute(sql, params)
        rows = result.fetchall()

        # 转换为时间线数据点
        points = []
        users_map: dict[str, str] = {}  # user_id -> username
        models_set: set = set()  # 收集所有出现的模型

        if include_user_info and not user_id:
            for row in rows:
                created_at, row_user_id, model, username, interval_minutes = row
                point_data: dict[str, Any] = {
                    "x": created_at.isoformat(),
                    "y": round(float(interval_minutes), 2),
                    "user_id": str(row_user_id),
                }
                if model:
                    point_data["model"] = model
                    models_set.add(model)
                points.append(point_data)
                if row_user_id and username:
                    users_map[str(row_user_id)] = username
        else:
            for row in rows:
                created_at, model, interval_minutes = row
                point_data = {"x": created_at.isoformat(), "y": round(float(interval_minutes), 2)}
                if model:
                    point_data["model"] = model
                    models_set.add(model)
                points.append(point_data)

        response: dict[str, Any] = {
            "analysis_period_hours": hours,
            "total_points": len(points),
            "points": points,
        }

        if include_user_info and not user_id:
            response["users"] = users_map

        # 如果有模型信息，返回模型列表
        if models_set:
            response["models"] = sorted(models_set)

        return response
