"""
Orchestration 模块

提供请求编排相关的组件：
- CandidateResolver: 候选解析器，负责获取和排序可用的 Provider 组合
- RequestDispatcher: 请求分发器，负责执行单个候选请求
- ErrorClassifier: 错误分类器，负责错误分类（纯逻辑，无副作用）
- ErrorHandlerService: 错误处理服务，负责错误后的副作用（缓存失效、健康记录等）
"""

from .candidate_resolver import CandidateResolver
from .error_classifier import ErrorAction, ErrorClassifier
from .error_handler import ErrorHandlerService
from .request_dispatcher import RequestDispatcher

__all__ = [
    "CandidateResolver",
    "RequestDispatcher",
    "ErrorClassifier",
    "ErrorHandlerService",
    "ErrorAction",
]
