from __future__ import annotations

import inspect
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from src.core.logger import logger
from src.models.database import ApiKey, RequestCandidate, User

from .gateway_contract import GatewayStreamReportRequest, GatewaySyncReportRequest


def _gateway_module() -> Any:
    from . import gateway as gateway_module

    return gateway_module
