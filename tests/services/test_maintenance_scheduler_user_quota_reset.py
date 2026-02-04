from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.services.system.maintenance_scheduler import MaintenanceScheduler


@pytest.mark.asyncio
async def test_user_quota_reset_disabled(monkeypatch):
    scheduler = MaintenanceScheduler()

    mock_db = MagicMock()
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.create_session",
        lambda: mock_db,
    )

    def fake_get_config(cls, db, key, default=None):
        if key == "enable_user_quota_reset":
            return False
        return default

    mock_set_config = MagicMock()
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.SystemConfigService.get_config",
        classmethod(fake_get_config),
    )
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.SystemConfigService.set_config",
        mock_set_config,
    )

    await scheduler._perform_user_quota_reset()

    assert not mock_db.query.called
    assert not mock_db.commit.called
    assert not mock_set_config.called


@pytest.mark.asyncio
async def test_user_quota_reset_not_due_skips(monkeypatch):
    scheduler = MaintenanceScheduler()

    mock_db = MagicMock()
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.create_session",
        lambda: mock_db,
    )

    last_reset_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    def fake_get_config(cls, db, key, default=None):
        if key == "enable_user_quota_reset":
            return True
        if key == "user_quota_reset_interval_days":
            return 2
        if key == "user_quota_last_reset_at":
            return last_reset_at
        return default

    mock_set_config = MagicMock()
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.SystemConfigService.get_config",
        classmethod(fake_get_config),
    )
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.SystemConfigService.set_config",
        mock_set_config,
    )

    await scheduler._perform_user_quota_reset()

    assert not mock_db.query.called
    assert not mock_db.commit.called
    assert not mock_set_config.called


@pytest.mark.asyncio
async def test_user_quota_reset_due_runs(monkeypatch):
    scheduler = MaintenanceScheduler()

    mock_db = MagicMock()
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.create_session",
        lambda: mock_db,
    )

    last_reset_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()

    def fake_get_config(cls, db, key, default=None):
        if key == "enable_user_quota_reset":
            return True
        if key == "user_quota_reset_interval_days":
            return 2
        if key == "user_quota_last_reset_at":
            return last_reset_at
        return default

    mock_set_config = MagicMock()
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.SystemConfigService.get_config",
        classmethod(fake_get_config),
    )
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.SystemConfigService.set_config",
        mock_set_config,
    )

    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.update.return_value = 7
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query

    await scheduler._perform_user_quota_reset()

    mock_db.query.assert_called_once()
    mock_filter.update.assert_called_once()
    _, update_kwargs = mock_filter.update.call_args
    assert update_kwargs["synchronize_session"] is False
    mock_db.commit.assert_called_once()
    mock_set_config.assert_called_once()


@pytest.mark.asyncio
async def test_user_quota_reset_invalid_interval_defaults_to_1(monkeypatch):
    scheduler = MaintenanceScheduler()

    mock_db = MagicMock()
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.create_session",
        lambda: mock_db,
    )

    def fake_get_config(cls, db, key, default=None):
        if key == "enable_user_quota_reset":
            return True
        if key == "user_quota_reset_interval_days":
            return "abc"
        if key == "user_quota_last_reset_at":
            return None
        return default

    mock_set_config = MagicMock()
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.SystemConfigService.get_config",
        classmethod(fake_get_config),
    )
    monkeypatch.setattr(
        "src.services.system.maintenance_scheduler.SystemConfigService.set_config",
        mock_set_config,
    )

    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.update.return_value = 1
    mock_query.filter.return_value = mock_filter
    mock_db.query.return_value = mock_query

    await scheduler._perform_user_quota_reset()

    mock_db.commit.assert_called_once()
    mock_set_config.assert_called_once()
