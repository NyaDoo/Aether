import pytest

from src.core.cache_service import CacheService
from src.models.database import GlobalModel, Model
from src.services.cache.model_cache import ModelCacheService


class _FakeQuery:
    def __init__(self, *, first_result=None, all_result=None, on_all=None):
        self._first_result = first_result
        self._all_result = all_result if all_result is not None else []
        self._on_all = on_all

    def join(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        return self._first_result

    def all(self):
        if self._on_all:
            self._on_all()
        return self._all_result


class _FakeSession:
    def __init__(self, *, direct_match: GlobalModel):
        self._direct_match = direct_match

    def query(self, *entities):
        if entities == (GlobalModel,):
            return _FakeQuery(first_result=self._direct_match)

        # 如果 direct match 命中，不应再走 provider_model_name 分支
        if entities == (Model, GlobalModel):
            raise AssertionError("provider_model_name query should not run when direct match exists")

        raise AssertionError(f"Unexpected query entities: {entities}")


@pytest.mark.asyncio
async def test_resolve_global_model_prefers_direct_match(monkeypatch) -> None:
    async def _fake_get(_key: str):
        return None

    async def _fake_set(_key: str, _value, ttl_seconds: int = 60):  # noqa: ARG001
        return True

    monkeypatch.setattr(CacheService, "get", staticmethod(_fake_get))
    monkeypatch.setattr(CacheService, "set", staticmethod(_fake_set))

    global_model = GlobalModel(
        id="gm-1",
        name="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        supported_capabilities=[],
        config={},
        default_tiered_pricing=None,
        default_price_per_request=None,
        is_active=True,
    )
    db = _FakeSession(direct_match=global_model)

    resolved = await ModelCacheService.resolve_global_model_by_name_or_alias(db, global_model.name)
    assert resolved is global_model

