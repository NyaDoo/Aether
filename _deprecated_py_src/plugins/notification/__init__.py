"""
通知插件
"""

from .base import Notification, NotificationLevel, NotificationPlugin
from .email import EmailNotificationPlugin
from .webhook import WebhookNotificationPlugin

__all__ = [
    "NotificationPlugin",
    "NotificationLevel",
    "Notification",
    "WebhookNotificationPlugin",
    "EmailNotificationPlugin",
]
