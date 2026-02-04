"""
系统维护定时任务调度器

包含以下任务：
- 统计聚合：每天凌晨聚合前一天的统计数据
- Provider 签到：每天凌晨执行所有已配置 Provider 的签到
- 使用记录清理：分级清理策略（压缩、清空、删除）
- 审计日志清理：定期清理过期的审计日志
- 连接池监控：定期检查数据库连接池状态
- Pending 状态清理：清理异常的 Pending 状态记录
- Gemini 文件映射清理：清理过期的 Gemini 文件→Key 映射
- OAuth Token 刷新：主动刷新即将过期的 OAuth token

使用 APScheduler 进行任务调度，支持时区配置。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.core.logger import logger
from src.database import create_session
from src.models.database import AuditLog, Provider, Usage
from src.services.provider_ops.service import ProviderOpsService
from src.services.system.config import SystemConfigService
from src.services.system.scheduler import get_scheduler
from src.services.system.stats_aggregator import StatsAggregatorService
from src.services.user.apikey import ApiKeyService
from src.utils.compression import compress_json


class MaintenanceScheduler:
    """系统维护任务调度器"""

    # 签到任务的 job_id
    CHECKIN_JOB_ID = "provider_checkin"
    # OAuth 刷新任务的 job_id
    OAUTH_REFRESH_JOB_ID = "oauth_token_refresh"
    # 用户配额重置任务的 job_id
    USER_QUOTA_RESET_JOB_ID = "user_quota_reset"

    def __init__(self) -> None:
        self.running = False
        self._interval_tasks = []
        self._stats_aggregation_lock = asyncio.Lock()

    def trigger_oauth_refresh_check(self) -> None:
        """
        触发 OAuth Token 刷新检查

        当新增 OAuth Key 时调用此方法，重新调度刷新任务。
        会取消当前的调度，并立即重新计算下次执行时间。
        """
        if not self.running:
            return

        asyncio.create_task(self._schedule_next_oauth_refresh())

    def _get_checkin_time(self) -> tuple[int, int]:
        """获取签到任务的执行时间

        Returns:
            (hour, minute) 元组
        """
        db = create_session()
        try:
            time_str = SystemConfigService.get_config(db, "provider_checkin_time", "01:05")
            return self._parse_time_string(time_str)
        finally:
            db.close()

    def _get_user_quota_reset_time(self) -> tuple[int, int]:
        """获取用户配额重置任务的执行时间

        Returns:
            (hour, minute) 元组
        """
        db = create_session()
        try:
            time_str = SystemConfigService.get_config(db, "user_quota_reset_time", "05:00")
            return self._parse_user_quota_reset_time_string(time_str)
        finally:
            db.close()

    @staticmethod
    def _parse_time_string(time_str: str) -> tuple[int, int]:
        """解析时间字符串为 (hour, minute) 元组

        Args:
            time_str: HH:MM 格式的时间字符串

        Returns:
            (hour, minute) 元组，解析失败返回默认值 (1, 5)
        """
        try:
            if not time_str or ":" not in time_str:
                return (1, 5)
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            # 验证范围
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return (hour, minute)
            return (1, 5)
        except (ValueError, IndexError):
            return (1, 5)

    @staticmethod
    def _parse_user_quota_reset_time_string(time_str: str) -> tuple[int, int]:
        """解析用户配额重置时间字符串为 (hour, minute) 元组

        Returns:
            (hour, minute) 元组，解析失败返回默认值 (5, 0)
        """
        try:
            if not time_str or ":" not in time_str:
                return (5, 0)
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            # 验证范围
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return (hour, minute)
            return (5, 0)
        except (ValueError, IndexError):
            return (5, 0)

    def update_checkin_time(self, time_str: str) -> bool:
        """更新签到任务的执行时间

        Args:
            time_str: HH:MM 格式的时间字符串

        Returns:
            是否成功更新
        """
        hour, minute = self._parse_time_string(time_str)

        scheduler = get_scheduler()
        success = scheduler.reschedule_cron_job(
            self.CHECKIN_JOB_ID,
            hour=hour,
            minute=minute,
        )

        if success:
            logger.info(f"Provider 签到任务时间已更新为: {hour:02d}:{minute:02d}")

        return success

    def update_user_quota_reset_time(self, time_str: str) -> bool:
        """更新用户配额重置任务的执行时间

        Args:
            time_str: HH:MM 格式的时间字符串

        Returns:
            是否成功更新
        """
        hour, minute = self._parse_user_quota_reset_time_string(time_str)

        scheduler = get_scheduler()
        success = scheduler.reschedule_cron_job(
            self.USER_QUOTA_RESET_JOB_ID,
            hour=hour,
            minute=minute,
        )

        if success:
            logger.info(f"用户配额重置任务时间已更新为: {hour:02d}:{minute:02d}")

        return success

    def get_checkin_job_info(self) -> dict | None:
        """获取签到任务的信息

        Returns:
            任务信息字典
        """
        scheduler = get_scheduler()
        return scheduler.get_job_info(self.CHECKIN_JOB_ID)

    async def start(self) -> Any:
        """启动调度器"""
        if self.running:
            logger.warning("Maintenance scheduler already running")
            return

        self.running = True
        logger.info("系统维护调度器已启动")

        scheduler = get_scheduler()

        # 注册定时任务
        # 统计聚合任务 - UTC 00:05 执行
        scheduler.add_cron_job(
            self._scheduled_stats_aggregation,
            hour=0,
            minute=5,
            job_id="stats_aggregation",
            name="统计数据聚合",
            timezone="UTC",
        )
        # 小时统计聚合任务 - 每小时 05 分执行（UTC）
        scheduler.add_cron_job(
            self._scheduled_hourly_stats_aggregation,
            hour="*",
            minute=5,
            job_id="stats_hourly_aggregation",
            name="统计小时数据聚合",
            timezone="UTC",
        )
        # 统计聚合补偿任务 - 每 30 分钟检查缺失并回填
        scheduler.add_interval_job(
            self._scheduled_stats_aggregation,
            minutes=30,
            job_id="stats_aggregation_backfill",
            name="统计数据聚合补偿",
            backfill=True,
        )

        # 清理任务 - 凌晨 3 点执行
        scheduler.add_cron_job(
            self._scheduled_cleanup,
            hour=3,
            minute=0,
            job_id="usage_cleanup",
            name="使用记录清理",
        )

        # 连接池监控 - 每 5 分钟
        scheduler.add_interval_job(
            self._scheduled_monitor,
            minutes=5,
            job_id="pool_monitor",
            name="连接池监控",
        )

        # Pending 状态清理 - 每 5 分钟
        scheduler.add_interval_job(
            self._scheduled_pending_cleanup,
            minutes=5,
            job_id="pending_cleanup",
            name="Pending状态清理",
        )

        # 审计日志清理 - 凌晨 4 点执行
        scheduler.add_cron_job(
            self._scheduled_audit_cleanup,
            hour=4,
            minute=0,
            job_id="audit_cleanup",
            name="审计日志清理",
        )

        # Gemini 文件映射清理 - 每小时执行
        scheduler.add_interval_job(
            self._scheduled_gemini_file_mapping_cleanup,
            hours=1,
            job_id="gemini_file_mapping_cleanup",
            name="Gemini文件映射清理",
        )

        # Provider 签到任务 - 根据配置时间执行
        checkin_hour, checkin_minute = self._get_checkin_time()
        scheduler.add_cron_job(
            self._scheduled_provider_checkin,
            hour=checkin_hour,
            minute=checkin_minute,
            job_id=self.CHECKIN_JOB_ID,
            name="Provider签到",
        )

        # OAuth Token 刷新任务 - 动态调度
        # 根据最近即将过期的 token 时间来调度，避免固定间隔频繁查询
        # 启动时先执行一次，计算下次执行时间
        asyncio.create_task(self._schedule_next_oauth_refresh())

        # 用户配额重置任务 - 根据配置时间执行（按周期配置决定是否执行）
        quota_reset_hour, quota_reset_minute = self._get_user_quota_reset_time()
        scheduler.add_cron_job(
            self._scheduled_user_quota_reset,
            hour=quota_reset_hour,
            minute=quota_reset_minute,
            job_id=self.USER_QUOTA_RESET_JOB_ID,
            name="用户配额自动重置",
        )

        # 启动时执行一次初始化任务
        asyncio.create_task(self._run_startup_tasks())

    async def _run_startup_tasks(self) -> None:
        """启动时执行的初始化任务"""
        # 延迟执行，等待系统完全启动（Redis 连接、其他后台任务稳定）
        # 增加延迟时间避免与 UsageQueueConsumer 等后台任务竞争数据库连接
        await asyncio.sleep(10)

        try:
            logger.info("启动时执行首次清理任务...")
            await self._perform_cleanup()
        except Exception as e:
            logger.exception(f"启动时清理任务执行出错: {e}")

        try:
            logger.info("启动时检查统计数据...")
            await self._perform_stats_aggregation(backfill=True)
        except Exception as e:
            logger.exception(f"启动时统计聚合任务出错: {e}")

    async def stop(self) -> Any:
        """停止调度器"""
        if not self.running:
            return

        self.running = False
        scheduler = get_scheduler()
        scheduler.stop()

        logger.info("系统维护调度器已停止")

    # ========== 任务函数（APScheduler 直接调用异步函数） ==========

    async def _scheduled_stats_aggregation(self, backfill: bool = False) -> None:
        """统计聚合任务（定时调用）"""
        await self._perform_stats_aggregation(backfill=backfill)

    async def _scheduled_hourly_stats_aggregation(self) -> None:
        """小时统计聚合任务（定时调用）"""
        await self._perform_hourly_stats_aggregation()

    async def _scheduled_cleanup(self) -> None:
        """清理任务（定时调用）"""
        await self._perform_cleanup()

    async def _scheduled_monitor(self) -> None:
        """监控任务（定时调用）"""
        try:
            from src.database import log_pool_status

            log_pool_status()
        except Exception as e:
            logger.exception(f"连接池监控任务出错: {e}")

    async def _scheduled_pending_cleanup(self) -> None:
        """Pending 清理任务（定时调用）"""
        await self._perform_pending_cleanup()

    async def _scheduled_audit_cleanup(self) -> None:
        """审计日志清理任务（定时调用）"""
        await self._perform_audit_cleanup()

    async def _scheduled_gemini_file_mapping_cleanup(self) -> None:
        """Gemini 文件映射清理任务（定时调用）"""
        await self._perform_gemini_file_mapping_cleanup()

    async def _scheduled_provider_checkin(self) -> None:
        """Provider 签到任务（定时调用）"""
        await self._perform_provider_checkin()

    async def _scheduled_oauth_token_refresh(self) -> None:
        """OAuth Token 刷新任务（定时调用）"""
        await self._perform_oauth_token_refresh()
        # 执行完成后，调度下次执行
        await self._schedule_next_oauth_refresh()

    async def _schedule_next_oauth_refresh(self) -> None:
        """
        动态调度下次 OAuth Token 刷新任务

        策略：
        - 查询所有 OAuth Key 的 expires_at
        - 找到最近即将过期的 token（在 refresh_threshold 内）
        - 设置下次执行时间为：最近过期时间 - 提前量（如提前 1 小时刷新）
        - 如果没有即将过期的 token，设置默认间隔（如 6 小时后）
        """
        import json
        import time

        from src.core.crypto import crypto_service
        from src.models.database import ProviderAPIKey

        # 延迟启动，等待系统初始化
        await asyncio.sleep(5)

        scheduler = get_scheduler()
        job_id = "oauth_token_refresh"

        try:
            db = create_session()
            try:
                # 检查配置开关
                if not SystemConfigService.get_config(db, "enable_oauth_token_refresh", True):
                    logger.info("OAuth Token 自动刷新已禁用，不调度任务")
                    return
                # 查找所有活跃的 OAuth 类型 Key
                oauth_keys = (
                    db.query(ProviderAPIKey)
                    .filter(
                        ProviderAPIKey.auth_type == "oauth",
                        ProviderAPIKey.is_active == True,  # noqa: E712
                    )
                    .all()
                )

                if not oauth_keys:
                    # 没有 OAuth Key，6 小时后再检查
                    next_run = datetime.now(timezone.utc) + timedelta(hours=6)
                    scheduler.add_date_job(
                        self._scheduled_oauth_token_refresh,
                        run_date=next_run,
                        job_id=job_id,
                        name="OAuth Token刷新",
                    )
                    logger.info("没有 OAuth Key，下次检查时间: {}", next_run.isoformat())
                    return

                now = int(time.time())
                # 24 小时内过期的都需要刷新（含提前量）
                refresh_window = 24 * 3600
                # 提前 1 小时执行刷新
                refresh_advance = 1 * 3600
                refresh_threshold_seconds = refresh_window + refresh_advance
                refresh_threshold = now + refresh_threshold_seconds

                earliest_expires_at: int | None = None

                for key in oauth_keys:
                    if not key.auth_config:
                        continue

                    try:
                        decrypted_config = crypto_service.decrypt(key.auth_config)
                        token_meta = json.loads(decrypted_config)
                        expires_at = token_meta.get("expires_at")

                        if expires_at is None:
                            continue

                        expires_at_int = int(expires_at)

                        # 已经过期或在阈值内，需要立即刷新
                        if expires_at_int <= refresh_threshold:
                            # 立即执行
                            next_run = datetime.now(timezone.utc) + timedelta(seconds=10)
                            scheduler.add_date_job(
                                self._scheduled_oauth_token_refresh,
                                run_date=next_run,
                                job_id=job_id,
                                name="OAuth Token刷新",
                            )
                            logger.info(
                                "发现即将过期的 OAuth Token，立即执行刷新: {}",
                                next_run.isoformat(),
                            )
                            return

                        # 记录最近的过期时间
                        if earliest_expires_at is None or expires_at_int < earliest_expires_at:
                            earliest_expires_at = expires_at_int

                    except Exception:
                        continue

                # 计算下次执行时间
                if earliest_expires_at is not None:
                    # 在最近过期时间前 24 小时 + 提前量执行
                    next_run_ts = earliest_expires_at - refresh_threshold_seconds
                    # 确保不会是过去的时间
                    if next_run_ts <= now:
                        next_run_ts = now + 60  # 1 分钟后
                    next_run = datetime.fromtimestamp(next_run_ts, tz=timezone.utc)
                else:
                    # 没有有效的过期时间，6 小时后再检查
                    next_run = datetime.now(timezone.utc) + timedelta(hours=6)

                # 限制最大间隔为 24 小时
                max_next_run = datetime.now(timezone.utc) + timedelta(hours=24)
                if next_run > max_next_run:
                    next_run = max_next_run

                scheduler.add_date_job(
                    self._scheduled_oauth_token_refresh,
                    run_date=next_run,
                    job_id=job_id,
                    name="OAuth Token刷新",
                )
                logger.info("OAuth Token 刷新任务已调度，下次执行时间: {}", next_run.isoformat())

            finally:
                db.close()

        except Exception as e:
            logger.exception("调度 OAuth Token 刷新任务失败: {}", e)
            # 出错时 1 小时后重试
            next_run = datetime.now(timezone.utc) + timedelta(hours=1)
            try:
                scheduler.add_date_job(
                    self._scheduled_oauth_token_refresh,
                    run_date=next_run,
                    job_id=job_id,
                    name="OAuth Token刷新",
                )
            except Exception:
                pass

    async def _scheduled_user_quota_reset(self) -> None:
        """用户配额重置任务（定时调用）"""
        await self._perform_user_quota_reset()

    # ========== 实际任务实现 ==========

    async def _perform_stats_aggregation(self, backfill: bool = False) -> None:
        """执行统计聚合任务

        Args:
            backfill: 是否回填历史数据（启动时检查缺失的日期）
        """
        if self._stats_aggregation_lock.locked():
            logger.info("统计聚合任务正在运行，跳过本次触发")
            return

        async with self._stats_aggregation_lock:
            db = create_session()
            try:
                # 检查是否启用统计聚合
                if not SystemConfigService.get_config(db, "enable_stats_aggregation", True):
                    logger.info("统计聚合已禁用，跳过聚合任务")
                    return

                logger.info("开始执行统计数据聚合...")

                from src.models.database import StatsDaily
                from src.models.database import User as DBUser

                # 使用 UTC 日期，定时任务在 UTC 00:05 触发，聚合 UTC 昨天
                now_utc = datetime.now(timezone.utc)
                today_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

                if backfill:
                    # 启动时检查并回填缺失的日期
                    from src.models.database import StatsSummary

                    summary = db.query(StatsSummary).first()
                    if not summary:
                        # 首次运行，回填所有历史数据
                        logger.info("检测到首次运行，开始回填历史统计数据...")
                        days_to_backfill = SystemConfigService.get_config(
                            db, "stats_backfill_days", 365
                        )
                        count = StatsAggregatorService.backfill_historical_data(
                            db, days=days_to_backfill
                        )
                        logger.info(f"历史数据回填完成，共 {count} 天")
                        return

                    # 非首次运行，检查最近是否有缺失的日期需要回填
                    from src.models.database import StatsDailyModel, StatsDailyProvider

                    yesterday_utc_date = today_utc.date() - timedelta(days=1)
                    max_backfill_days: int = (
                        SystemConfigService.get_config(db, "max_stats_backfill_days", 30) or 30
                    )

                    # 计算回填检查的起始日期
                    check_start_date = yesterday_utc_date - timedelta(days=max_backfill_days - 1)
                    check_start_dt = datetime.combine(
                        check_start_date, datetime.min.time(), tzinfo=timezone.utc
                    )

                    # 获取 StatsDaily 和 StatsDailyModel 中已有数据的日期集合
                    existing_daily_dates = set()
                    existing_model_dates = set()
                    existing_provider_dates = set()

                    daily_stats = (
                        db.query(StatsDaily.date).filter(StatsDaily.date >= check_start_dt).all()
                    )
                    for (stat_date,) in daily_stats:
                        if stat_date.tzinfo is None:
                            stat_date = stat_date.replace(tzinfo=timezone.utc)
                        existing_daily_dates.add(stat_date.date())

                    model_stats = (
                        db.query(StatsDailyModel.date)
                        .filter(StatsDailyModel.date >= check_start_dt)
                        .distinct()
                        .all()
                    )
                    for (stat_date,) in model_stats:
                        if stat_date.tzinfo is None:
                            stat_date = stat_date.replace(tzinfo=timezone.utc)
                        existing_model_dates.add(stat_date.date())

                    provider_stats = (
                        db.query(StatsDailyProvider.date)
                        .filter(StatsDailyProvider.date >= check_start_dt)
                        .distinct()
                        .all()
                    )
                    for (stat_date,) in provider_stats:
                        if stat_date.tzinfo is None:
                            stat_date = stat_date.replace(tzinfo=timezone.utc)
                        existing_provider_dates.add(stat_date.date())

                    # 找出需要回填的日期
                    all_dates = set()
                    current = check_start_date
                    while current <= yesterday_utc_date:
                        all_dates.add(current)
                        current += timedelta(days=1)

                    # 需要回填 StatsDaily 的日期
                    missing_daily_dates = all_dates - existing_daily_dates
                    # 需要回填 StatsDailyModel 的日期
                    missing_model_dates = all_dates - existing_model_dates
                    # 需要回填 StatsDailyProvider 的日期
                    missing_provider_dates = all_dates - existing_provider_dates
                    # 合并所有需要处理的日期
                    dates_to_process = (
                        missing_daily_dates | missing_model_dates | missing_provider_dates
                    )

                    if dates_to_process:
                        sorted_dates = sorted(dates_to_process)
                        logger.info(
                            f"检测到 {len(dates_to_process)} 天的统计数据需要回填 "
                            f"(StatsDaily 缺失 {len(missing_daily_dates)} 天, "
                            f"StatsDailyModel 缺失 {len(missing_model_dates)} 天, "
                            f"StatsDailyProvider 缺失 {len(missing_provider_dates)} 天)"
                        )

                        users = db.query(DBUser.id).filter(DBUser.is_active.is_(True)).all()
                        user_ids = [user_id for (user_id,) in users]

                        failed_dates = 0
                        for current_date in sorted_dates:
                            try:
                                current_date_utc = datetime.combine(
                                    current_date, datetime.min.time(), tzinfo=timezone.utc
                                )
                                StatsAggregatorService.aggregate_daily_stats_bundle(
                                    db, current_date_utc, user_ids=user_ids
                                )
                            except Exception as e:
                                failed_dates += 1
                                logger.warning(f"回填日期 {current_date} 失败: {e}")
                                try:
                                    db.rollback()
                                except Exception as rollback_err:
                                    logger.error(f"回滚失败: {rollback_err}")

                        StatsAggregatorService.update_summary(db)

                        if failed_dates > 0:
                            logger.warning(
                                f"回填完成，共处理 {len(dates_to_process)} 天，"
                                f"失败: {failed_dates} 天"
                            )
                        else:
                            logger.info(f"缺失数据回填完成，共处理 {len(dates_to_process)} 天")
                    else:
                        logger.info("统计数据已是最新，无需回填")
                    return

                # 定时任务：聚合昨天 (UTC) 的数据
                yesterday_utc = today_utc - timedelta(days=1)
                users = db.query(DBUser.id).filter(DBUser.is_active.is_(True)).all()
                user_ids = [user_id for (user_id,) in users]

                StatsAggregatorService.aggregate_daily_stats_bundle(
                    db, yesterday_utc, user_ids=user_ids
                )

                StatsAggregatorService.update_summary(db)

                logger.info("统计数据聚合完成")

            except Exception as e:
                logger.exception(f"统计聚合任务执行失败: {e}")
                try:
                    db.rollback()
                except Exception:
                    pass
            finally:
                db.close()

    async def _perform_hourly_stats_aggregation(self) -> None:
        """执行小时统计聚合任务"""
        db = create_session()
        try:
            if not SystemConfigService.get_config(db, "enable_stats_aggregation", True):
                logger.info("统计聚合已禁用，跳过小时聚合任务")
                return

            now_utc = datetime.now(timezone.utc)
            last_hour = now_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            StatsAggregatorService.aggregate_hourly_stats_bundle(db, last_hour)
            logger.info(f"小时统计聚合完成: {last_hour.isoformat()}")
        except Exception as e:
            logger.exception(f"小时统计聚合任务执行失败: {e}")
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    async def _perform_pending_cleanup(self) -> None:
        """执行 pending 状态清理"""
        db = create_session()
        try:
            from src.services.usage.service import UsageService

            # 获取配置的超时时间（默认 10 分钟）
            timeout_minutes = SystemConfigService.get_config(
                db, "pending_request_timeout_minutes", 10
            )

            # 执行清理
            cleaned_count = UsageService.cleanup_stale_pending_requests(
                db, timeout_minutes=timeout_minutes
            )

            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 条超时的 pending/streaming 请求")

        except Exception as e:
            logger.exception(f"清理 pending 请求失败: {e}")
            db.rollback()
        finally:
            db.close()

    async def _perform_audit_cleanup(self) -> None:
        """执行审计日志清理任务"""
        db = create_session()
        try:
            # 检查是否启用自动清理
            if not SystemConfigService.get_config(db, "enable_auto_cleanup", True):
                logger.info("自动清理已禁用，跳过审计日志清理")
                return

            # 获取审计日志保留天数（默认 30 天，最少 7 天）
            audit_retention_days = max(
                SystemConfigService.get_config(db, "audit_log_retention_days", 30),
                7,  # 最少保留 7 天，防止误配置删除所有审计日志
            )
            batch_size = SystemConfigService.get_config(db, "cleanup_batch_size", 1000)

            cutoff_time = datetime.now(timezone.utc) - timedelta(days=audit_retention_days)

            logger.info(f"开始清理 {audit_retention_days} 天前的审计日志...")

            total_deleted = 0
            while True:
                # 先查询要删除的记录 ID（分批）
                records_to_delete = (
                    db.query(AuditLog.id)
                    .filter(AuditLog.created_at < cutoff_time)
                    .limit(batch_size)
                    .all()
                )

                if not records_to_delete:
                    break

                record_ids = [r.id for r in records_to_delete]

                # 执行删除
                result = db.execute(
                    delete(AuditLog)
                    .where(AuditLog.id.in_(record_ids))
                    .execution_options(synchronize_session=False)
                )

                rows_deleted = result.rowcount
                db.commit()

                total_deleted += rows_deleted
                logger.debug(f"已删除 {rows_deleted} 条审计日志，累计 {total_deleted} 条")

                await asyncio.sleep(0.1)

            if total_deleted > 0:
                logger.info(f"审计日志清理完成，共删除 {total_deleted} 条记录")
            else:
                logger.info("无需清理的审计日志")

        except Exception as e:
            logger.exception(f"审计日志清理失败: {e}")
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    async def _perform_gemini_file_mapping_cleanup(self) -> None:
        """清理过期的 Gemini 文件映射记录"""
        db = create_session()
        try:
            from src.services.gemini_files_mapping import cleanup_expired_mappings

            deleted_count = cleanup_expired_mappings(db)

            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 条过期的 Gemini 文件映射")

        except Exception as e:
            logger.exception(f"Gemini 文件映射清理失败: {e}")
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    async def _perform_provider_checkin(self) -> None:
        """执行 Provider 签到任务

        遍历所有已配置 provider_ops 的 Provider，触发签到。
        签到会在余额查询时一起执行（先签到再查询余额）。
        """
        db = create_session()
        try:
            # 检查是否启用签到任务
            if not SystemConfigService.get_config(db, "enable_provider_checkin", True):
                logger.info("Provider 签到已禁用，跳过签到任务")
                return

            # 获取所有已配置 provider_ops 的活跃 Provider（只查询需要的字段）
            providers = (
                db.query(Provider.id, Provider.config).filter(Provider.is_active.is_(True)).all()
            )
            provider_ids = [p.id for p in providers if p.config and p.config.get("provider_ops")]

            if not provider_ids:
                logger.info("无已配置的 Provider，跳过签到任务")
                return

            logger.info(f"开始执行 Provider 签到，共 {len(provider_ids)} 个...")

            # 释放主 session 的连接，避免在整个签到期间占用连接池
            # （后续每个 provider 将使用独立短生命周期 session）
            try:
                if db.in_transaction():
                    db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
            try:
                db.close()
            except Exception:
                pass
            db = None

            # 使用信号量限制并发，避免同时发起过多请求
            concurrency = 3  # 签到任务并发数
            semaphore = asyncio.Semaphore(concurrency)

            async def _checkin_provider(provider_id: str) -> tuple[str, bool, str]:
                """执行单个 Provider 的签到"""
                async with semaphore:
                    task_db = create_session()
                    try:
                        service = ProviderOpsService(task_db)
                        # 触发余额查询（会先执行签到）
                        result = await service.query_balance(provider_id)
                        # 检查签到结果
                        checkin_success = None
                        checkin_message = ""
                        if result.data and hasattr(result.data, "extra") and result.data.extra:
                            checkin_success = result.data.extra.get("checkin_success")
                            checkin_message = result.data.extra.get("checkin_message", "")
                        if checkin_success is True:
                            return provider_id, True, checkin_message
                        elif checkin_success is False:
                            return provider_id, False, checkin_message
                        else:
                            # None 表示未执行签到（可能没配置 Cookie）
                            return provider_id, False, "未执行签到"
                    except Exception as e:
                        logger.warning(f"Provider {provider_id} 签到失败: {e}")
                        return provider_id, False, str(e)
                    finally:
                        try:
                            task_db.close()
                        except Exception:
                            pass

            # 并行执行签到
            tasks = [_checkin_provider(pid) for pid in provider_ids]
            results = await asyncio.gather(*tasks)

            # 统计结果
            success_count = sum(1 for _, success, _ in results if success)
            logger.info(f"Provider 签到完成: {success_count}/{len(provider_ids)} 成功")

            # 记录详细结果
            for provider_id, success, message in results:
                if success:
                    logger.debug(f"  - {provider_id}: 签到成功 - {message}")
                elif message != "未执行签到":
                    logger.debug(f"  - {provider_id}: 签到失败 - {message}")

        except Exception as e:
            logger.exception(f"Provider 签到任务执行失败: {e}")
        finally:
            if db is not None:
                db.close()

    async def _perform_user_quota_reset(self) -> None:
        """执行用户配额自动重置任务

        适用范围：
        - 未删除（is_deleted=false）
        - 仅对 quota_usd != NULL 的用户生效
        """
        db = create_session()
        try:
            # 检查是否启用用户配额重置
            if not SystemConfigService.get_config(db, "enable_user_quota_reset", False):
                logger.info("用户配额自动重置已禁用，跳过任务")
                return

            # 重置周期（天数），不限制上限
            interval_value = SystemConfigService.get_config(db, "user_quota_reset_interval_days", 1)
            try:
                interval_days = int(interval_value)
            except Exception:
                interval_days = 1
            if interval_days < 1:
                interval_days = 1

            # 滚动计算：根据上次执行日（APP_TIMEZONE）判断是否到期
            last_reset_at = SystemConfigService.get_config(db, "user_quota_last_reset_at")

            should_run = True
            if last_reset_at:
                last_dt: datetime | None = None
                try:
                    if isinstance(last_reset_at, str):
                        last_dt = datetime.fromisoformat(last_reset_at)
                except Exception:
                    last_dt = None

                if last_dt is None:
                    logger.warning("user_quota_last_reset_at 格式无效，视为需要执行一次")
                else:
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)

                    from zoneinfo import ZoneInfo

                    from src.services.system.scheduler import APP_TIMEZONE

                    tz = ZoneInfo(APP_TIMEZONE)
                    now_local = datetime.now(tz)
                    last_local_date = last_dt.astimezone(tz).date()
                    days_since_reset = (now_local.date() - last_local_date).days

                    if days_since_reset < 0:
                        logger.warning(
                            "user_quota_last_reset_at 在未来，跳过本次用户配额自动重置"
                        )
                        should_run = False
                    elif days_since_reset < interval_days:
                        logger.info(
                            f"用户配额自动重置未到周期，跳过任务（{days_since_reset}/{interval_days}天）"
                        )
                        should_run = False

            if not should_run:
                return

            from src.models.database import User as DBUser

            now_utc = datetime.now(timezone.utc)
            reset_count = (
                db.query(DBUser)
                .filter(
                    DBUser.is_deleted.is_(False),
                    DBUser.quota_usd.isnot(None),
                )
                .update(
                    {
                        DBUser.used_usd: 0.0,
                        DBUser.updated_at: now_utc,
                    },
                    synchronize_session=False,
                )
            )
            db.commit()

            # 记录 last_reset_at（成功执行后更新，滚动计算用）
            SystemConfigService.set_config(
                db,
                "user_quota_last_reset_at",
                now_utc.isoformat(),
                "用户配额自动重置的上次执行时间（UTC，内部使用）",
            )

            logger.info(f"用户配额自动重置完成: interval_days={interval_days}, 重置用户数={reset_count}")

        except Exception as e:
            logger.exception(f"用户配额自动重置任务执行失败: {e}")
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

    async def _perform_cleanup(self) -> None:
        """执行清理任务"""
        db = create_session()
        try:
            # 检查是否启用自动清理
            if not SystemConfigService.get_config(db, "enable_auto_cleanup", True):
                logger.info("自动清理已禁用，跳过清理任务")
                return

            logger.info("开始执行使用记录分级清理...")

            # 获取配置参数
            detail_retention = SystemConfigService.get_config(db, "detail_log_retention_days", 7)
            compressed_retention = SystemConfigService.get_config(
                db, "compressed_log_retention_days", 90
            )
            header_retention = SystemConfigService.get_config(db, "header_retention_days", 90)
            log_retention = SystemConfigService.get_config(db, "log_retention_days", 365)
            batch_size = SystemConfigService.get_config(db, "cleanup_batch_size", 1000)

            now = datetime.now(timezone.utc)

            # 1. 压缩详细日志 (body 字段 -> 压缩字段)
            detail_cutoff = now - timedelta(days=detail_retention)
            body_compressed = await self._cleanup_body_fields(db, detail_cutoff, batch_size)

            # 2. 清理压缩字段（90天后）
            compressed_cutoff = now - timedelta(days=compressed_retention)
            compressed_cleaned = await self._cleanup_compressed_fields(
                db, compressed_cutoff, batch_size
            )

            # 3. 清理请求头
            header_cutoff = now - timedelta(days=header_retention)
            header_cleaned = await self._cleanup_header_fields(db, header_cutoff, batch_size)

            # 4. 删除过期记录
            log_cutoff = now - timedelta(days=log_retention)
            records_deleted = await self._delete_old_records(db, log_cutoff, batch_size)

            # 5. 清理过期的API Keys
            auto_delete = SystemConfigService.get_config(db, "auto_delete_expired_keys", False)
            keys_cleaned = ApiKeyService.cleanup_expired_keys(db, auto_delete=auto_delete)

            logger.info(
                f"清理完成: 压缩 {body_compressed} 条, "
                f"清理压缩字段 {compressed_cleaned} 条, "
                f"清理header {header_cleaned} 条, "
                f"删除记录 {records_deleted} 条, "
                f"清理过期Keys {keys_cleaned} 条"
            )

        except Exception as e:
            logger.exception(f"清理任务执行失败: {e}")
            db.rollback()
        finally:
            db.close()

    async def _cleanup_body_fields(
        self, db: Session, cutoff_time: datetime, batch_size: int
    ) -> int:
        """压缩 request_body 和 response_body 字段到压缩字段

        逐条处理，确保每条记录都正确更新
        """
        from sqlalchemy import null, update

        total_compressed = 0
        no_progress_count = 0  # 连续无进展计数
        processed_ids: set = set()  # 记录已处理的 ID，防止重复处理

        while True:
            batch_db = create_session()
            try:
                # 1. 查询需要压缩的记录
                # 注意：排除已经是 NULL 或 JSON null 的记录
                records = (
                    batch_db.query(Usage.id, Usage.request_body, Usage.response_body)
                    .filter(Usage.created_at < cutoff_time)
                    .filter((Usage.request_body.isnot(None)) | (Usage.response_body.isnot(None)))
                    .limit(batch_size)
                    .all()
                )

                if not records:
                    break

                # 过滤掉实际值为 None 的记录（JSON null 被解析为 Python None）
                valid_records = [
                    (rid, req, resp)
                    for rid, req, resp in records
                    if req is not None or resp is not None
                ]

                if not valid_records:
                    # 所有记录都是 JSON null，需要清理它们
                    logger.warning(
                        f"检测到 {len(records)} 条记录的 body 字段为 JSON null，进行清理"
                    )
                    for record_id, _, _ in records:
                        batch_db.execute(
                            update(Usage)
                            .where(Usage.id == record_id)
                            .values(request_body=null(), response_body=null())
                        )
                    batch_db.commit()
                    continue

                # 检测是否有重复的 ID（说明更新未生效）
                current_ids = {r[0] for r in valid_records}
                repeated_ids = current_ids & processed_ids
                if repeated_ids:
                    logger.error(
                        f"检测到重复处理的记录 ID: {list(repeated_ids)[:5]}...，"
                        "说明数据库更新未生效，终止循环"
                    )
                    break

                batch_success = 0

                # 2. 逐条更新（确保每条都正确处理）
                for record_id, req_body, resp_body in valid_records:
                    try:
                        # 使用 null() 确保设置的是 SQL NULL 而不是 JSON null
                        result = batch_db.execute(
                            update(Usage)
                            .where(Usage.id == record_id)
                            .values(
                                request_body=null(),
                                response_body=null(),
                                request_body_compressed=(
                                    compress_json(req_body) if req_body else None
                                ),
                                response_body_compressed=(
                                    compress_json(resp_body) if resp_body else None
                                ),
                            )
                        )
                        if result.rowcount > 0:
                            batch_success += 1
                            processed_ids.add(record_id)
                    except Exception as e:
                        logger.warning(f"压缩记录 {record_id} 失败: {e}")
                        continue

                batch_db.commit()

                # 3. 检查是否有实际进展
                if batch_success == 0:
                    no_progress_count += 1
                    if no_progress_count >= 3:
                        logger.error(
                            f"压缩 body 字段连续 {no_progress_count} 批无进展，"
                            "终止循环以避免死循环"
                        )
                        break
                else:
                    no_progress_count = 0  # 重置计数

                total_compressed += batch_success
                logger.debug(
                    f"已压缩 {batch_success} 条记录的 body 字段，累计 {total_compressed} 条"
                )

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(f"压缩 body 字段失败: {e}")
                try:
                    batch_db.rollback()
                except Exception:
                    pass
                break
            finally:
                batch_db.close()

        return total_compressed

    async def _cleanup_compressed_fields(
        self, db: Session, cutoff_time: datetime, batch_size: int
    ) -> int:
        """清理压缩字段（90天后删除压缩的body）

        每批使用短生命周期 session，避免 ORM 缓存问题
        """
        from sqlalchemy import null, update

        total_cleaned = 0

        while True:
            batch_db = create_session()
            try:
                # 查询需要清理压缩字段的记录
                records_to_clean = (
                    batch_db.query(Usage.id)
                    .filter(Usage.created_at < cutoff_time)
                    .filter(
                        (Usage.request_body_compressed.isnot(None))
                        | (Usage.response_body_compressed.isnot(None))
                    )
                    .limit(batch_size)
                    .all()
                )

                if not records_to_clean:
                    break

                record_ids = [r.id for r in records_to_clean]

                # 批量更新，使用 null() 确保设置 SQL NULL
                result = batch_db.execute(
                    update(Usage)
                    .where(Usage.id.in_(record_ids))
                    .values(
                        request_body_compressed=null(),
                        response_body_compressed=null(),
                    )
                )

                rows_updated = result.rowcount
                batch_db.commit()

                if rows_updated == 0:
                    logger.warning("清理压缩字段: rowcount=0，可能存在问题")
                    break

                total_cleaned += rows_updated
                logger.debug(f"已清理 {rows_updated} 条记录的压缩字段，累计 {total_cleaned} 条")

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(f"清理压缩字段失败: {e}")
                try:
                    batch_db.rollback()
                except Exception:
                    pass
                break
            finally:
                batch_db.close()

        return total_cleaned

    async def _cleanup_header_fields(
        self, db: Session, cutoff_time: datetime, batch_size: int
    ) -> int:
        """清理 request_headers, response_headers 和 provider_request_headers 字段

        每批使用短生命周期 session，避免 ORM 缓存问题
        """
        from sqlalchemy import null, update

        total_cleaned = 0

        while True:
            batch_db = create_session()
            try:
                # 先查询需要清理的记录ID（分批）
                records_to_clean = (
                    batch_db.query(Usage.id)
                    .filter(Usage.created_at < cutoff_time)
                    .filter(
                        (Usage.request_headers.isnot(None))
                        | (Usage.response_headers.isnot(None))
                        | (Usage.provider_request_headers.isnot(None))
                    )
                    .limit(batch_size)
                    .all()
                )

                if not records_to_clean:
                    break

                record_ids = [r.id for r in records_to_clean]

                # 批量更新，使用 null() 确保设置 SQL NULL
                result = batch_db.execute(
                    update(Usage)
                    .where(Usage.id.in_(record_ids))
                    .values(
                        request_headers=null(),
                        response_headers=null(),
                        provider_request_headers=null(),
                    )
                )

                rows_updated = result.rowcount
                batch_db.commit()

                if rows_updated == 0:
                    logger.warning("清理 header 字段: rowcount=0，可能存在问题")
                    break

                total_cleaned += rows_updated
                logger.debug(f"已清理 {rows_updated} 条记录的 header 字段，累计 {total_cleaned} 条")

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(f"清理 header 字段失败: {e}")
                try:
                    batch_db.rollback()
                except Exception:
                    pass
                break
            finally:
                batch_db.close()

        return total_cleaned

    async def _delete_old_records(self, db: Session, cutoff_time: datetime, batch_size: int) -> int:
        """删除过期的完整记录"""
        total_deleted = 0

        while True:
            try:
                # 查询要删除的记录ID（分批）
                records_to_delete = (
                    db.query(Usage.id)
                    .filter(Usage.created_at < cutoff_time)
                    .limit(batch_size)
                    .all()
                )

                if not records_to_delete:
                    break

                record_ids = [r.id for r in records_to_delete]

                # 执行删除
                result = db.execute(
                    delete(Usage)
                    .where(Usage.id.in_(record_ids))
                    .execution_options(synchronize_session=False)
                )

                rows_deleted = result.rowcount
                db.commit()

                total_deleted += rows_deleted
                logger.debug(f"已删除 {rows_deleted} 条过期记录，累计 {total_deleted} 条")

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.exception(f"删除过期记录失败: {e}")
                try:
                    db.rollback()
                except Exception:
                    pass
                break

        return total_deleted

    async def _perform_oauth_token_refresh(self) -> None:
        """
        主动刷新即将过期的 OAuth token

        策略：
        - 查找所有 auth_type='oauth' 且 is_active=True 的 Key
        - 检查 auth_config 中的 expires_at，如果在 24 小时内过期则刷新
        - 使用 refresh_token 换取新的 access_token
        - 更新数据库中的 token 信息
        """
        import json
        import time

        from src.core.crypto import crypto_service
        from src.core.provider_oauth_utils import enrich_auth_config, post_oauth_token
        from src.core.provider_templates.fixed_providers import FIXED_PROVIDERS
        from src.core.provider_templates.types import ProviderType
        from src.models.database import ProviderAPIKey

        # 检查配置开关
        check_db = create_session()
        try:
            if not SystemConfigService.get_config(check_db, "enable_oauth_token_refresh", True):
                logger.info("OAuth Token 自动刷新已禁用，跳过任务")
                return
        finally:
            check_db.close()

        logger.info("开始执行 OAuth Token 刷新任务...")

        db = create_session()
        refreshed_count = 0
        failed_count = 0
        skipped_count = 0

        try:
            # 查找所有活跃的 OAuth 类型 Key
            oauth_keys = (
                db.query(ProviderAPIKey)
                .filter(
                    ProviderAPIKey.auth_type == "oauth",
                    ProviderAPIKey.is_active == True,  # noqa: E712
                )
                .all()
            )

            if not oauth_keys:
                logger.info("没有找到需要刷新的 OAuth Key")
                return

            logger.info("找到 {} 个 OAuth Key，开始检查过期状态...", len(oauth_keys))

            now = int(time.time())
            # 24 小时内过期的都刷新（含提前量）
            refresh_window = 24 * 3600
            # 提前 1 小时执行刷新
            refresh_advance = 1 * 3600
            refresh_threshold = now + refresh_window + refresh_advance

            for key in oauth_keys:
                try:
                    # 解密 auth_config
                    if not key.auth_config:
                        skipped_count += 1
                        continue

                    try:
                        decrypted_config = crypto_service.decrypt(key.auth_config)
                        token_meta = json.loads(decrypted_config)
                    except Exception:
                        logger.warning("Key {} auth_config 解密失败，跳过", key.id)
                        skipped_count += 1
                        continue

                    expires_at = token_meta.get("expires_at")
                    refresh_token = token_meta.get("refresh_token")
                    provider_type = str(token_meta.get("provider_type") or "")

                    # 检查是否需要刷新
                    if expires_at is None:
                        skipped_count += 1
                        continue

                    try:
                        expires_at_int = int(expires_at)
                    except (ValueError, TypeError):
                        skipped_count += 1
                        continue

                    if expires_at_int > refresh_threshold:
                        # 还没到刷新时间
                        skipped_count += 1
                        continue

                    if not refresh_token or not provider_type:
                        logger.warning(
                            "Key {} 缺少 refresh_token 或 provider_type，无法刷新", key.id
                        )
                        skipped_count += 1
                        continue

                    # 获取 provider 模板
                    try:
                        provider_type_enum = ProviderType(provider_type)
                    except ValueError:
                        logger.warning("Key {} 未知的 provider_type: {}", key.id, provider_type)
                        skipped_count += 1
                        continue

                    template = FIXED_PROVIDERS.get(provider_type_enum)
                    if not template or not template.oauth:
                        logger.warning("Key {} provider {} 不支持 OAuth", key.id, provider_type)
                        skipped_count += 1
                        continue

                    # 获取代理配置
                    proxy_config = None
                    if key.provider and key.provider.endpoints:
                        for endpoint in key.provider.endpoints:
                            if endpoint.proxy:
                                proxy_config = endpoint.proxy
                                break

                    # 执行刷新
                    token_url = template.oauth.token_url
                    is_json = "anthropic.com" in token_url

                    if is_json:
                        body = {
                            "grant_type": "refresh_token",
                            "client_id": template.oauth.client_id,
                            "refresh_token": str(refresh_token),
                        }
                        headers = {
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        }
                        data = None
                        json_body = body
                    else:
                        form = {
                            "grant_type": "refresh_token",
                            "client_id": template.oauth.client_id,
                            "refresh_token": str(refresh_token),
                        }
                        if template.oauth.client_secret:
                            form["client_secret"] = template.oauth.client_secret
                        headers = {
                            "Content-Type": "application/x-www-form-urlencoded",
                            "Accept": "application/json",
                        }
                        data = form
                        json_body = None

                    logger.info(
                        "刷新 Key {} ({}) 的 OAuth token，当前过期时间: {}",
                        key.id,
                        key.name,
                        datetime.fromtimestamp(expires_at_int, tz=timezone.utc).isoformat(),
                    )

                    resp = await post_oauth_token(
                        provider_type=provider_type,
                        token_url=token_url,
                        headers=headers,
                        data=data,
                        json_body=json_body,
                        proxy_config=proxy_config,
                        timeout_seconds=30.0,
                    )

                    if 200 <= resp.status_code < 300:
                        token = resp.json()
                        access_token = str(token.get("access_token") or "")
                        new_refresh_token = str(token.get("refresh_token") or "")
                        expires_in = token.get("expires_in")
                        new_expires_at = None

                        try:
                            if expires_in is not None:
                                new_expires_at = int(time.time()) + int(expires_in)
                        except Exception:
                            new_expires_at = None

                        if access_token:
                            # 更新 token_meta
                            token_meta["token_type"] = token.get("token_type")
                            if new_refresh_token:
                                token_meta["refresh_token"] = new_refresh_token
                            token_meta["expires_at"] = new_expires_at
                            token_meta["scope"] = token.get("scope")
                            token_meta["updated_at"] = int(time.time())

                            # 提取额外信息
                            token_meta = await enrich_auth_config(
                                provider_type=provider_type,
                                auth_config=token_meta,
                                token_response=token,
                                access_token=access_token,
                                proxy_config=proxy_config,
                            )

                            # 更新数据库
                            encrypted_token = crypto_service.encrypt(access_token)
                            encrypted_config = crypto_service.encrypt(json.dumps(token_meta))

                            key.api_key = encrypted_token
                            key.auth_config = encrypted_config
                            # 刷新成功，清除失效标记
                            key.oauth_invalid_at = None
                            key.oauth_invalid_reason = None
                            db.commit()

                            refreshed_count += 1
                            new_expires_str = (
                                datetime.fromtimestamp(new_expires_at, tz=timezone.utc).isoformat()
                                if new_expires_at
                                else "unknown"
                            )
                            logger.info(
                                "Key {} ({}) OAuth token 刷新成功，新过期时间: {}",
                                key.id,
                                key.name,
                                new_expires_str,
                            )
                        else:
                            failed_count += 1
                            logger.warning(
                                "Key {} ({}) 刷新响应中没有 access_token", key.id, key.name
                            )
                    else:
                        failed_count += 1
                        # 解析错误原因
                        error_reason = "HTTP {}".format(resp.status_code)
                        try:
                            error_body = resp.json()
                            if "error" in error_body:
                                error_reason = str(
                                    error_body.get("error_description") or error_body.get("error")
                                )
                        except Exception:
                            error_reason = (
                                resp.text[:100] if resp.text else "HTTP {}".format(resp.status_code)
                            )

                        # 标记为失效（400/401/403 通常表示永久性错误）
                        if resp.status_code in (400, 401, 403):
                            key.oauth_invalid_at = datetime.now(timezone.utc)
                            key.oauth_invalid_reason = error_reason
                            db.commit()
                            logger.warning(
                                "Key {} ({}) OAuth token 刷新失败，已标记为失效: {}",
                                key.id,
                                key.name,
                                error_reason,
                            )
                        else:
                            logger.warning(
                                "Key {} ({}) OAuth token 刷新失败，状态码: {}，响应: {}",
                                key.id,
                                key.name,
                                resp.status_code,
                                resp.text[:200],
                            )

                except Exception as e:
                    failed_count += 1
                    logger.exception("Key {} OAuth token 刷新出错: {}", key.id, e)
                    try:
                        db.rollback()
                    except Exception:
                        pass

                # 避免请求过于频繁
                await asyncio.sleep(1)

        except Exception as e:
            logger.exception("OAuth Token 刷新任务执行出错: {}", e)
        finally:
            db.close()

        logger.info(
            "OAuth Token 刷新任务完成: 刷新 {} 个，失败 {} 个，跳过 {} 个",
            refreshed_count,
            failed_count,
            skipped_count,
        )


# 全局单例
_maintenance_scheduler = None


def get_maintenance_scheduler() -> MaintenanceScheduler:
    """获取维护调度器单例"""
    global _maintenance_scheduler
    if _maintenance_scheduler is None:
        _maintenance_scheduler = MaintenanceScheduler()
    return _maintenance_scheduler


# 兼容旧名称（deprecated）
def get_cleanup_scheduler() -> MaintenanceScheduler:
    """获取维护调度器单例（已废弃，请使用 get_maintenance_scheduler）"""
    return get_maintenance_scheduler()


CleanupScheduler = MaintenanceScheduler  # 兼容旧名称
