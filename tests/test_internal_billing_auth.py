from types import SimpleNamespace

import anyio
import pytest
from fastapi import HTTPException

from mint_server.routes import internal as internal_route


class _StubUsageStore:
    def __init__(self):
        self.query_account_id = None

    async def query_logs(self, since=None, account_id=None, limit=100, offset=0):
        self.query_account_id = account_id
        return [], 0, False

    async def get_account_summary(self, account_id):
        return {"total_quantity": 7, "charge_item_totals": {"sampling": 7}}


def test_internal_usage_logs_legacy_user_falls_back_to_user_id(monkeypatch):
    usage_store = _StubUsageStore()

    async def _get_usage_store():
        return usage_store

    monkeypatch.setattr(internal_route, "get_usage_store", _get_usage_store)

    request = SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "aaaaaaaaaaaaaaaaaaaaaaaa"}))
    response = anyio.run(internal_route.get_usage_logs, request, None, 100, 0)

    assert response.count == 0
    assert usage_store.query_account_id == "aaaaaaaaaaaaaaaaaaaaaaaa"


def test_internal_usage_summary_allows_legacy_admin_without_account_id(monkeypatch):
    usage_store = _StubUsageStore()

    async def _get_usage_store():
        return usage_store

    monkeypatch.setattr(internal_route, "get_usage_store", _get_usage_store)

    request = SimpleNamespace(
        state=SimpleNamespace(user_data={"user_id": "admin", "user_role": "admin", "is_admin": True})
    )
    response = anyio.run(internal_route.get_usage_summary, "bbbbbbbbbbbbbbbbbbbbbbbb", request)

    assert response.total_quantity == 7
    assert response.charge_item_totals == {"sampling": 7}


def test_internal_usage_summary_rejects_other_legacy_user_account(monkeypatch):
    usage_store = _StubUsageStore()

    async def _get_usage_store():
        return usage_store

    monkeypatch.setattr(internal_route, "get_usage_store", _get_usage_store)

    request = SimpleNamespace(state=SimpleNamespace(user_data={"user_id": "aaaaaaaaaaaaaaaaaaaaaaaa"}))

    with pytest.raises(HTTPException) as exc:
        anyio.run(internal_route.get_usage_summary, "bbbbbbbbbbbbbbbbbbbbbbbb", request)

    assert exc.value.status_code == 403
