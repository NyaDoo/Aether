"""
任务服务层（Phase2）

统一任务框架相关的应用层入口：
- 候选提交阶段：`services.candidate.CandidateService`
- 终态结算：`services.task.application.TaskApplicationService`
"""

from .application import TaskApplicationService

__all__ = [
    "TaskApplicationService",
]
