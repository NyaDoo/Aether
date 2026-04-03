"""
任务服务层（Phase2/Phase3）

统一任务框架相关的应用层入口：
- 候选域能力：`services.candidate.CandidateService`（resolve/record 等）
- 终态结算：`services.task.service.TaskService.finalize_video_task`
- 统一门面：`services.task.service.TaskService`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # NOTE: keep package import lightweight; avoid importing heavy modules on submodule imports
    from .service import TaskService as TaskService

__all__ = [
    "TaskService",
]


def __getattr__(name: str) -> type:  # pragma: no cover
    if name == "TaskService":
        from .service import TaskService

        return TaskService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(__all__)
