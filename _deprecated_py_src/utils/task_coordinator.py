"""分布式任务协调器，确保仅有一个 worker 执行特定任务

锁清理策略：
- 单实例模式（默认）：启动时使用原子操作清理旧锁并获取新锁
- 多实例模式：使用 NX 选项竞争锁，依赖 TTL 处理异常退出

使用方式：
- 默认行为：启动时清理旧锁（适用于单机部署）
- 多实例部署：设置 SINGLE_INSTANCE_MODE=false 禁用启动清理
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from src.core.logger import logger

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows 环境
    fcntl = None


class StartupTaskCoordinator:
    """利用 Redis 或文件锁，保证任务只在单个进程/实例中运行"""

    # 类级别标记：在当前进程中是否已尝试过启动清理
    # 注意：这在 fork 模式下每个 worker 都是独立的
    _startup_cleanup_attempted = False

    def __init__(self, redis_client: Any | None = None, lock_dir: str | None = None) -> None:
        self.redis = redis_client
        self._tokens: dict[str, str] = {}
        self._file_handles: dict[str, object] = {}
        self._refresh_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock_lost_callbacks: dict[str, Callable[[str], Awaitable[None] | None]] = {}
        self._lock_dir = pathlib.Path(lock_dir or os.getenv("TASK_LOCK_DIR", "./.locks"))
        if not self._lock_dir.exists():
            self._lock_dir.mkdir(parents=True, exist_ok=True)
        # 单实例模式：启动时清理旧锁（适用于单机部署，避免残留锁问题）
        self._single_instance_mode = os.getenv("SINGLE_INSTANCE_MODE", "true").lower() == "true"

    def _redis_key(self, name: str) -> str:
        return f"task_lock:{name}"

    async def acquire(self, name: str, ttl: int | None = None) -> bool:
        ttl = ttl or int(os.getenv("TASK_COORDINATOR_LOCK_TTL", "86400"))

        if self.redis:
            token = str(uuid.uuid4())
            try:
                if self._single_instance_mode:
                    # 单实例模式：使用 Lua 脚本原子性地"清理旧锁 + 竞争获取"
                    # 只有当锁不存在或成功获取时才返回 1
                    # 这样第一个执行的 worker 会清理旧锁并获取，后续 worker 会正常竞争
                    script = """
                    local key = KEYS[1]
                    local token = ARGV[1]
                    local ttl = tonumber(ARGV[2])
                    local startup_key = KEYS[1] .. ':startup'

                    -- 检查是否已有 worker 执行过启动清理
                    local cleaned = redis.call('GET', startup_key)
                    if not cleaned then
                        -- 第一个 worker：删除旧锁，标记已清理
                        redis.call('DEL', key)
                        redis.call('SET', startup_key, '1', 'EX', 60)
                    end

                    -- 尝试获取锁（NX 模式）
                    local result = redis.call('SET', key, token, 'NX', 'EX', ttl)
                    if result then
                        return 1
                    end
                    return 0
                    """
                    result = await self.redis.eval(
                        script, 2, self._redis_key(name), self._redis_key(name), token, ttl
                    )
                    if result == 1:
                        self._tokens[name] = token
                        self._start_refresh_task(name, ttl)
                        logger.info(f"任务 {name} 通过 Redis 锁独占执行")
                        return True
                    return False
                else:
                    # 多实例模式：直接使用 NX 选项竞争锁
                    acquired = await self.redis.set(self._redis_key(name), token, nx=True, ex=ttl)
                    if acquired:
                        self._tokens[name] = token
                        self._start_refresh_task(name, ttl)
                        logger.info(f"任务 {name} 通过 Redis 锁独占执行")
                        return True
                    return False
            except Exception as exc:  # pragma: no cover - Redis 异常回退
                logger.warning(f"Redis 锁获取失败，回退到文件锁: {exc}")

        return await self._acquire_file_lock(name)

    async def release(self, name: str) -> Any:
        refresh_task = self._refresh_tasks.pop(name, None)
        if refresh_task is not None:
            refresh_task.cancel()
        self._lock_lost_callbacks.pop(name, None)

        if self.redis and name in self._tokens:
            token = self._tokens.pop(name)
            script = """
            if redis.call('GET', KEYS[1]) == ARGV[1] then
                return redis.call('DEL', KEYS[1])
            end
            return 0
            """
            try:
                await self.redis.eval(script, 1, self._redis_key(name), token)
            except Exception as exc:  # pragma: no cover
                logger.warning(f"释放 Redis 锁失败: {exc}")

        handle = self._file_handles.pop(name, None)
        if handle and fcntl:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()

    async def _acquire_file_lock(self, name: str) -> bool:
        if fcntl is None:
            # 在不支持 fcntl 的平台上退化为单进程锁
            if name in self._file_handles:
                return False
            self._file_handles[name] = object()
            logger.warning("操作系统不支持文件锁，任务锁仅在当前进程生效")
            return True

        lock_path = self._lock_dir / f"{name}.lock"
        handle = open(lock_path, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._file_handles[name] = handle
            logger.info(f"任务 {name} 使用文件锁独占执行")
            return True
        except BlockingIOError:
            handle.close()
            return False

    def register_lock_lost_callback(
        self,
        name: str,
        callback: Callable[[str], Awaitable[None] | None],
    ) -> None:
        self._lock_lost_callbacks[name] = callback

    def _start_refresh_task(self, name: str, ttl: int) -> None:
        if not self.redis:
            return

        existing_task = self._refresh_tasks.pop(name, None)
        if existing_task is not None:
            existing_task.cancel()

        self._refresh_tasks[name] = asyncio.create_task(self._refresh_lock_loop(name, ttl))

    async def _refresh_lock(self, name: str, ttl: int) -> bool:
        token = self._tokens.get(name)
        if not self.redis or token is None:
            return False

        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
        end
        return 0
        """
        result = await self.redis.eval(script, 1, self._redis_key(name), token, ttl)
        return result == 1

    async def _refresh_lock_loop(self, name: str, ttl: int) -> None:
        interval = max(1, ttl // 3)
        max_consecutive_failures = 5
        consecutive_failures = 0
        try:
            while name in self._tokens:
                await asyncio.sleep(interval)
                if name not in self._tokens:
                    break
                try:
                    refreshed = await self._refresh_lock(name, ttl)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"续租任务锁 {name} 连续失败 {consecutive_failures} 次，视为锁丢失: {exc}"
                        )
                        self._tokens.pop(name, None)
                        await self._notify_lock_lost(name)
                        break
                    backoff = min(interval, 2**consecutive_failures)
                    logger.warning(
                        f"续租任务锁 {name} 失败 ({consecutive_failures}/{max_consecutive_failures})，"
                        f"{backoff}s 后重试: {exc}"
                    )
                    await asyncio.sleep(backoff)
                    continue

                consecutive_failures = 0
                if not refreshed:
                    logger.warning(f"任务 {name} 的 Redis 锁已失效，停止续租")
                    self._tokens.pop(name, None)
                    await self._notify_lock_lost(name)
                    break
        except asyncio.CancelledError:
            raise

    async def _notify_lock_lost(self, name: str) -> None:
        callback = self._lock_lost_callbacks.pop(name, None)
        if callback is None:
            return

        try:
            result = callback(name)
            if result is not None:
                await result
        except Exception as exc:  # pragma: no cover - 回调失败仅记录日志
            logger.exception("任务 {} 的失锁回调执行失败: {}", name, exc)


async def ensure_singleton_task(
    name: str, redis_client: Any | None = None, ttl: int | None = None
) -> Any:
    """便捷协程，返回 (coordinator, acquired)"""

    coordinator = StartupTaskCoordinator(redis_client)
    acquired = await coordinator.acquire(name, ttl=ttl)
    return coordinator, acquired
