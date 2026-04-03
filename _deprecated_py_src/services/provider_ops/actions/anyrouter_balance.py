"""
Anyrouter 余额查询操作（含自动签到）
"""

from typing import Any

import httpx

from src.core.logger import logger
from src.services.provider_ops.actions.balance import BalanceAction
from src.services.provider_ops.types import BalanceInfo


class AnyrouterBalanceAction(BalanceAction):
    """
    Anyrouter 余额查询

    特点：
    - 查询余额前始终自动签到
    - 签到端点为 /api/user/sign_in
    - Cookie 失效时返回友好的错误提示
    - quota 单位是 1/500000 美元（与 New API 相同）
    """

    display_name = "查询余额（含自动签到）"
    description = "查询账户余额，同时自动签到"

    _cookie_auth = True

    def _parse_balance(self, data: Any) -> BalanceInfo:
        """解析余额信息"""
        user_data = data.get("data", {}) if isinstance(data, dict) else {}
        quota_divisor = self.config.get("quota_divisor", 500000)

        # 注意：Anyrouter 中 quota 是剩余额度（total_available），不是总额度
        raw_quota = self._to_float(user_data.get("quota"))
        raw_used = self._to_float(user_data.get("used_quota"))

        total_available = raw_quota / quota_divisor if raw_quota is not None else None
        total_used = raw_used / quota_divisor if raw_used is not None else None

        return self._create_balance_info(
            total_available=total_available,
            total_used=total_used,
            currency=self.config.get("currency", "USD"),
        )

    async def _do_checkin(self, client: httpx.AsyncClient) -> dict[str, Any] | None:
        """
        执行自动签到（始终执行）

        Returns:
            签到结果字典，包含 success 和 message 字段
        """
        checkin_endpoint = self.config.get("checkin_endpoint", "/api/user/sign_in")
        site = client.base_url.host or str(client.base_url)

        try:
            response = await client.post(checkin_endpoint)

            try:
                data = response.json()
                success = data.get("success", False)
                message = data.get("message", "")

                if success:
                    logger.debug(f"[{site}] 签到成功: {message}")
                    return {"success": True, "message": message or "签到成功"}
                else:
                    already_indicators = ["已签到", "已签", "今日已", "already"]
                    is_already = any(ind in message.lower() for ind in already_indicators)
                    if is_already:
                        logger.debug(f"[{site}] 今日已签到: {message}")
                        return {"success": None, "message": message or "今日已签到"}
                    else:
                        logger.debug(f"[{site}] 签到失败: {message}")
                        return {"success": False, "message": message or "签到失败"}
            except Exception as e:
                logger.debug(f"[{site}] 签到响应解析失败: {e}")
                return {"success": False, "message": "响应解析失败"}

        except Exception as e:
            logger.debug(f"[{site}] 签到异常: {e}")
            return {"success": False, "message": str(e)}
