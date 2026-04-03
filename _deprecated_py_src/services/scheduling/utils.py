"""
调度器工具函数

从 CacheAwareScheduler 提取的静态工具方法。
"""

from __future__ import annotations

import hashlib

from sqlalchemy.orm import Session


def affinity_hash(affinity_key: str, identifier: str) -> int:
    """基于 affinity_key 和标识符的确定性哈希（用于同优先级内分散负载均衡）"""
    return int(hashlib.sha256(f"{affinity_key}:{identifier}".encode()).hexdigest()[:16], 16)


def release_db_connection_before_await(db: Session) -> None:
    """
    Best-effort: end a read-only transaction before awaiting async I/O.

    This scheduler does a lot of async work (cache/Redis) mixed with sync SQLAlchemy reads.
    If a SELECT has already started a transaction, the pooled connection can remain checked
    out while we await, causing pool pressure under concurrency.

    Safety:
    - Only commits when the Session has no ORM pending changes.
    - Temporarily disables expire_on_commit to keep already-loaded ORM objects usable.
    """
    try:
        if db is None:
            return
        has_pending_changes = bool(db.new) or bool(db.dirty) or bool(db.deleted)
        if has_pending_changes:
            return
        if not db.in_transaction():
            return

        original_expire_on_commit = getattr(db, "expire_on_commit", True)
        db.expire_on_commit = False
        try:
            db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.expire_on_commit = original_expire_on_commit
    except Exception:
        # Never let this optimization break scheduling
        return
