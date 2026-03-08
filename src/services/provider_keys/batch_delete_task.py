"""Pool Key 批量删除异步任务。

接口立即返回 task_id，后台执行删除，前端轮询进度。
任务状态存储在 Redis 中，支持多 worker 进程共享。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable

import redis.asyncio as aioredis
from sqlalchemy import delete as sa_delete

from src.clients.redis_client import get_redis_client
from src.core.logger import logger

# 任务状态
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# 任务完成后保留时长（秒），也用作 Redis key 的 TTL
_TASK_RETAIN_SECONDS = 600

# 删除 ProviderAPIKey 时的分批大小
_DELETE_BATCH_SIZE = 500

# Redis key 前缀
_REDIS_KEY_PREFIX = "batch_delete_task"

# 持有后台 asyncio.Task 引用，防止 GC 回收（进程内部，不需要跨进程共享）
_running_tasks: set[asyncio.Task[None]] = set()


def _task_key(task_id: str) -> str:
    return f"{_REDIS_KEY_PREFIX}:{task_id}"


class BatchDeleteTaskInfo:
    """任务状态数据对象（从 Redis 反序列化）。"""

    __slots__ = ("task_id", "provider_id", "status", "total", "deleted", "message")

    def __init__(
        self,
        task_id: str,
        provider_id: str,
        status: str = STATUS_PENDING,
        total: int = 0,
        deleted: int = 0,
        message: str = "",
    ):
        self.task_id = task_id
        self.provider_id = provider_id
        self.status = status
        self.total = total
        self.deleted = deleted
        self.message = message

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "provider_id": self.provider_id,
            "status": self.status,
            "total": self.total,
            "deleted": self.deleted,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BatchDeleteTaskInfo:
        return cls(
            task_id=data["task_id"],
            provider_id=data["provider_id"],
            status=data.get("status", STATUS_PENDING),
            total=int(data.get("total", 0)),
            deleted=int(data.get("deleted", 0)),
            message=data.get("message", ""),
        )


async def _save_task(
    task: BatchDeleteTaskInfo,
    ttl: int = _TASK_RETAIN_SECONDS,
    r: aioredis.Redis | None = None,
) -> None:
    """将任务状态写入 Redis。"""
    if r is None:
        r = await get_redis_client(require_redis=False)
    if not r:
        return
    try:
        await r.setex(_task_key(task.task_id), ttl, json.dumps(task.to_dict()))
    except Exception as e:
        logger.warning("Failed to save batch delete task to Redis: {}", e)


async def _load_task(task_id: str, r: aioredis.Redis | None = None) -> BatchDeleteTaskInfo | None:
    """从 Redis 加载任务状态。"""
    if r is None:
        r = await get_redis_client(require_redis=False)
    if not r:
        return None
    try:
        data = await r.get(_task_key(task_id))
        if data is None:
            return None
        return BatchDeleteTaskInfo.from_dict(json.loads(data))
    except Exception as e:
        logger.warning("Failed to load batch delete task from Redis: {}", e)
        return None


async def _update_task_field(
    task_id: str, r: aioredis.Redis | None = None, **fields: object
) -> None:
    """局部更新任务状态字段（read-modify-write）。"""
    if r is None:
        r = await get_redis_client(require_redis=False)
    if not r:
        return
    task = await _load_task(task_id, r=r)
    if task is None:
        return
    for k, v in fields.items():
        setattr(task, k, v)
    # 已完成/失败的任务只保留 _TASK_RETAIN_SECONDS
    if task.status in (STATUS_COMPLETED, STATUS_FAILED):
        await _save_task(task, ttl=_TASK_RETAIN_SECONDS, r=r)
    else:
        # 运行中的任务使用更长的 TTL，防止超长任务过期
        await _save_task(task, ttl=_TASK_RETAIN_SECONDS * 2, r=r)


async def submit_batch_delete(provider_id: str, key_ids: list[str]) -> str:
    """提交批量删除任务，返回 task_id。

    Raises:
        RuntimeError: Redis 不可用时无法追踪任务状态。
    """
    r = await get_redis_client(require_redis=False)
    if not r:
        raise RuntimeError("Redis is required for batch delete tasks but is not available")

    task_id = uuid.uuid4().hex[:16]
    task = BatchDeleteTaskInfo(
        task_id=task_id,
        provider_id=provider_id,
        total=len(key_ids),
    )
    await _save_task(task, ttl=_TASK_RETAIN_SECONDS * 2, r=r)

    bg = asyncio.create_task(
        _run_batch_delete(task_id, provider_id, key_ids),
        name=f"batch-delete-{task_id}",
    )
    _running_tasks.add(bg)
    bg.add_done_callback(_running_tasks.discard)
    return task_id


async def get_batch_delete_task(task_id: str) -> BatchDeleteTaskInfo | None:
    return await _load_task(task_id)


def _sync_delete(
    provider_id: str,
    key_ids: list[str],
    progress_callback: Callable[[int], None] | None = None,
) -> int:
    """在线程中执行的同步删除逻辑，避免阻塞事件循环。

    每批 cleanup + DELETE + commit 作为独立事务，减少锁持有时间。
    """
    from src.database import create_session
    from src.models.database import ProviderAPIKey
    from src.services.provider_keys.key_side_effects import cleanup_key_references

    db = create_session()
    try:
        affected = 0
        for i in range(0, len(key_ids), _DELETE_BATCH_SIZE):
            batch = key_ids[i : i + _DELETE_BATCH_SIZE]
            try:
                cleanup_key_references(db, batch)
                result = db.execute(
                    sa_delete(ProviderAPIKey).where(
                        ProviderAPIKey.provider_id == provider_id,
                        ProviderAPIKey.id.in_(batch),
                    )
                )
                rowcount = getattr(result, "rowcount", 0) or 0
                affected += int(rowcount)
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
                raise
            # 通过回调通知进度（在事件循环线程中更新 Redis）
            if progress_callback is not None:
                progress_callback(affected)
        return affected
    finally:
        try:
            db.close()
        except Exception:
            pass


async def _run_batch_delete(
    task_id: str,
    provider_id: str,
    key_ids: list[str],
) -> None:
    r = await get_redis_client(require_redis=False)
    await _update_task_field(task_id, r=r, status=STATUS_RUNNING)

    # 用于从工作线程安全地触发 Redis 进度更新（按时间限频，最多每 2 秒写一次）
    loop = asyncio.get_running_loop()
    last_report_time = [0.0]
    _PROGRESS_INTERVAL = 2.0

    def on_progress(current: int) -> None:
        now = time.monotonic()
        if now - last_report_time[0] < _PROGRESS_INTERVAL:
            return
        last_report_time[0] = now
        try:
            asyncio.run_coroutine_threadsafe(
                _update_task_field(task_id, r=r, deleted=current), loop
            )
        except RuntimeError:
            pass

    try:
        affected = await asyncio.to_thread(_sync_delete, provider_id, key_ids, on_progress)

        # 副作用（缓存失效等）是异步操作，在事件循环中执行
        if affected > 0:
            try:
                from src.database import get_db_context
                from src.services.provider_keys.key_side_effects import (
                    run_delete_key_side_effects,
                )

                with get_db_context() as db:
                    await run_delete_key_side_effects(
                        db=db,
                        provider_id=provider_id,
                        deleted_key_allowed_models=None,
                    )
            except Exception as exc:
                logger.error("batch delete side effects failed: {}", exc)

        await _update_task_field(
            task_id,
            r=r,
            status=STATUS_COMPLETED,
            deleted=affected,
            message=f"{affected} keys deleted",
        )
        logger.info(
            "[BATCH_DELETE_TASK] completed task={} provider={} total={} deleted={}",
            task_id,
            provider_id[:8],
            len(key_ids),
            affected,
        )

    except asyncio.CancelledError:
        await _update_task_field(
            task_id,
            r=r,
            status=STATUS_FAILED,
            message="task cancelled (shutdown)",
        )
        logger.warning(
            "[BATCH_DELETE_TASK] cancelled task={} provider={}",
            task_id,
            provider_id[:8],
        )
    except Exception as exc:
        await _update_task_field(
            task_id,
            r=r,
            status=STATUS_FAILED,
            message=str(exc),
        )
        logger.error(
            "[BATCH_DELETE_TASK] failed task={} provider={}: {}",
            task_id,
            provider_id[:8],
            exc,
        )
