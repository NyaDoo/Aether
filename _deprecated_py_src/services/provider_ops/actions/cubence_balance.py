"""
Cubence 余额查询操作
"""

from typing import Any

from src.services.provider_ops.actions.balance import BalanceAction
from src.services.provider_ops.types import BalanceInfo


class CubenceBalanceAction(BalanceAction):
    """
    Cubence 专用余额查询

    特点：
    - 余额单位直接是美元
    - 支持窗口限额查询（5小时/每周）
    - Cookie 失效时返回友好的错误提示
    """

    display_name = "查询余额（含窗口限额）"
    description = "查询账户余额和窗口限额信息"

    _cookie_auth = True

    def _parse_balance(self, data: Any) -> BalanceInfo:
        """解析 Cubence 余额信息"""
        # Cubence 响应格式：data.balance 和 data.subscription_limits
        response_data = data.get("data", {}) if isinstance(data, dict) else {}
        balance_data = response_data.get("balance", {})
        subscription_limits = response_data.get("subscription_limits", {})

        # 余额信息（单位直接是美元）
        total_available = balance_data.get("total_balance_dollar")
        normal_balance = balance_data.get("normal_balance_dollar")
        subscription_balance = balance_data.get("subscription_balance_dollar")
        charity_balance = balance_data.get("charity_balance_dollar")

        # 窗口限额信息
        extra: dict[str, Any] = {}

        # 5小时窗口限额
        five_hour = subscription_limits.get("five_hour", {})
        if five_hour:
            extra["five_hour_limit"] = {
                "limit": five_hour.get("limit"),
                "used": five_hour.get("used"),
                "remaining": five_hour.get("remaining"),
                "resets_at": five_hour.get("resets_at"),
            }

        # 每周窗口限额
        weekly = subscription_limits.get("weekly", {})
        if weekly:
            extra["weekly_limit"] = {
                "limit": weekly.get("limit"),
                "used": weekly.get("used"),
                "remaining": weekly.get("remaining"),
                "resets_at": weekly.get("resets_at"),
            }

        # 余额组成
        if normal_balance is not None:
            extra["normal_balance"] = normal_balance
        if subscription_balance is not None:
            extra["subscription_balance"] = subscription_balance
        if charity_balance is not None:
            extra["charity_balance"] = charity_balance

        return self._create_balance_info(
            total_available=total_available,
            currency=self.config.get("currency", "USD"),
            extra=extra if extra else None,
        )
