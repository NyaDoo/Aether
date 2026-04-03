"""
旧 executor 命名兼容入口。

当前主实现已经迁到 `execution_runtime_plan.py`。
"""

from src.services.request.execution_runtime_plan import (
    ExecutionPlan,
    ExecutionPlanBody,
    ExecutionPlanTimeouts,
    ExecutionProxySnapshot,
    PreparedExecutionPlan,
    build_execution_plan_body,
    build_proxy_snapshot,
    is_remote_contract_eligible,
    is_remote_execution_runtime_contract_eligible,
    is_remote_execution_runtime_proxy_supported,
    is_remote_proxy_supported,
    should_bypass_remote_execution_runtime,
    should_bypass_remote_execution_runtime_url,
    should_bypass_remote_executor,
    should_bypass_remote_executor_url,
)

__all__ = [
    "ExecutionPlan",
    "ExecutionPlanBody",
    "ExecutionPlanTimeouts",
    "ExecutionProxySnapshot",
    "PreparedExecutionPlan",
    "build_execution_plan_body",
    "build_proxy_snapshot",
    "is_remote_contract_eligible",
    "is_remote_execution_runtime_contract_eligible",
    "is_remote_execution_runtime_proxy_supported",
    "is_remote_proxy_supported",
    "should_bypass_remote_execution_runtime",
    "should_bypass_remote_execution_runtime_url",
    "should_bypass_remote_executor",
    "should_bypass_remote_executor_url",
]
