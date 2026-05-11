from __future__ import annotations

import asyncio


def test_checkpoint_index_pg_connection_disables_asyncpg_statement_cache(monkeypatch) -> None:
    from tinker_server import checkpoint_index

    calls = []

    class _Asyncpg:
        async def connect(self, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(checkpoint_index, "_checkpoint_index_pg_dsn", lambda: "postgres://mint@pg/mint")
    monkeypatch.setattr(checkpoint_index, "_checkpoint_index_timeout_s", lambda: 2.0)
    monkeypatch.setattr(checkpoint_index, "_import_asyncpg", lambda: _Asyncpg())

    asyncio.run(checkpoint_index._connect())

    assert calls == [
        {
            "dsn": "postgres://mint@pg/mint",
            "command_timeout": 5.0,
            "statement_cache_size": 0,
            "server_settings": {"application_name": "tinker_server_checkpoint_index"},
        }
    ]
