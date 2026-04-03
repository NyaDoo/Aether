from __future__ import annotations

import ipaddress

from fastapi import HTTPException, Request


def ensure_loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError(host)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="loopback access only") from exc
